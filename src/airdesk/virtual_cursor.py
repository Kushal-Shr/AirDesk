"""Stage 4: move a safe on-screen cursor with the unlocked left index finger."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import cv2
import mediapipe as mp
import numpy as np

from .control_panel import build_control_panel_status
from .hand_landmarks import (
    CAMERA_INDEX,
    MODEL_PATH,
    create_landmarker,
    draw_hand,
    normalized_to_pixel,
)
from .hand_lock import (
    FINGER_JOINTS,
    FistMetrics,
    HandLockState,
    classify_fist,
    finger_is_curled,
    joint_angle_degrees,
)


WINDOW_NAME = "AirDesk - Virtual Cursor"
ACTIVE_MARGIN_X = 0.20
ACTIVE_MARGIN_TOP = 0.28
ACTIVE_MARGIN_BOTTOM = 0.12
CURSOR_SMOOTHING = 0.50
MIN_HANDEDNESS_CONFIDENCE = 0.65
# This camera reports mirrored MediaPipe handedness for the user's setup.
# Start corrected so anatomical LEFT and RIGHT work without pressing H.
DEFAULT_SWAP_HANDEDNESS = True
CAPTURE_WIDTH = 640
CAPTURE_HEIGHT = 480
CAPTURE_FPS = 30
PANEL_STATUS_INTERVAL_SECONDS = 0.20


def configure_camera(camera) -> None:
    """Request a predictable low-latency camera format when the backend allows it."""
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)
    camera.set(cv2.CAP_PROP_FPS, CAPTURE_FPS)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)


class LatestFrameCapture:
    """Continuously capture frames and expose only the newest available one."""

    def __init__(self, camera) -> None:
        self._camera = camera
        self._condition = threading.Condition()
        self._latest_frame = None
        self._sequence = 0
        self._delivered_sequence = 0
        self._stopping = False
        self._capture_failed = False
        self._thread = threading.Thread(
            target=self._capture_loop,
            name="airdesk-camera",
            daemon=True,
        )

    def start(self) -> "LatestFrameCapture":
        self._thread.start()
        return self

    def _capture_loop(self) -> None:
        while True:
            with self._condition:
                if self._stopping:
                    return
            received, frame = self._camera.read()
            with self._condition:
                if self._stopping:
                    return
                if not received:
                    self._capture_failed = True
                    self._condition.notify_all()
                    return
                self._latest_frame = frame
                self._sequence += 1
                self._condition.notify_all()

    def read(self, timeout: float = 1.0):
        """Wait for a newer frame; any older unprocessed frames are discarded."""
        deadline = time.monotonic() + timeout
        with self._condition:
            while (
                self._sequence <= self._delivered_sequence
                and not self._capture_failed
                and not self._stopping
            ):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False, None
                self._condition.wait(timeout=remaining)
            if self._sequence <= self._delivered_sequence:
                return False, None
            self._delivered_sequence = self._sequence
            return True, self._latest_frame

    def release(self) -> None:
        with self._condition:
            self._stopping = True
            self._condition.notify_all()
        self._camera.release()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)


def limit_frame_resolution(frame):
    """Downscale oversized frames while preserving their aspect ratio."""
    frame_height, frame_width = frame.shape[:2]
    scale = min(CAPTURE_WIDTH / frame_width, CAPTURE_HEIGHT / frame_height, 1.0)
    if scale >= 1.0:
        return frame
    output_size = (
        max(1, round(frame_width * scale)),
        max(1, round(frame_height * scale)),
    )
    return cv2.resize(frame, output_size, interpolation=cv2.INTER_AREA)


def resolve_hand_side(category_name: str, swap_handedness: bool) -> str | None:
    """Convert MediaPipe's hand label using the current camera correction."""
    side = category_name.upper()
    if side not in {"LEFT", "RIGHT"}:
        return None
    if swap_handedness:
        return "RIGHT" if side == "LEFT" else "LEFT"
    return side


@dataclass(frozen=True)
class ActiveRectangle:
    """The central camera region mapped across the full output area."""

    left: int
    top: int
    right: int
    bottom: int

    @classmethod
    def from_frame(cls, frame_width: int, frame_height: int) -> "ActiveRectangle":
        return cls(
            left=int(frame_width * ACTIVE_MARGIN_X),
            top=int(frame_height * ACTIVE_MARGIN_TOP),
            right=int(frame_width * (1.0 - ACTIVE_MARGIN_X)) - 1,
            bottom=int(frame_height * (1.0 - ACTIVE_MARGIN_BOTTOM)) - 1,
        )


@dataclass(frozen=True)
class PointingMetrics:
    """Measurements for the deliberate left index-only pointer pose."""

    is_pointing: bool
    index_extended: bool
    other_fingers_curled: int
    thumb_folded: bool


def classify_index_point(landmarks) -> PointingMetrics:
    """Require one extended index finger while every other finger stays down."""
    points = np.array(
        [(landmark.x, landmark.y, landmark.z) for landmark in landmarks],
        dtype=np.float32,
    )
    wrist = points[0]

    index_angle = joint_angle_degrees(points[5], points[6], points[8])
    index_tip_to_wrist = np.linalg.norm(points[8] - wrist)
    index_pip_to_wrist = np.linalg.norm(points[6] - wrist)
    index_extended = bool(
        index_angle > 155.0
        and index_tip_to_wrist > index_pip_to_wrist * 1.12
    )

    other_fingers_curled = sum(
        finger_is_curled(points, wrist, mcp_id, pip_id, tip_id)
        for mcp_id, pip_id, tip_id in FINGER_JOINTS[1:]
    )

    palm_center = points[[0, 5, 9, 13, 17]].mean(axis=0)
    palm_size = max(float(np.linalg.norm(points[9] - wrist)), 1e-6)
    thumb_compactness = float(np.linalg.norm(points[4] - palm_center) / palm_size)
    thumb_folded = thumb_compactness < 1.25

    is_pointing = index_extended and other_fingers_curled == 3 and thumb_folded
    return PointingMetrics(
        is_pointing=is_pointing,
        index_extended=index_extended,
        other_fingers_curled=other_fingers_curled,
        thumb_folded=thumb_folded,
    )


@dataclass
class CursorSmoother:
    """Apply exponential smoothing to mapped cursor coordinates."""

    amount: float = CURSOR_SMOOTHING
    x: float | None = None
    y: float | None = None

    def update(self, target: tuple[int, int]) -> tuple[int, int]:
        target_x, target_y = target
        if self.x is None or self.y is None:
            self.x = float(target_x)
            self.y = float(target_y)
        else:
            self.x += self.amount * (target_x - self.x)
            self.y += self.amount * (target_y - self.y)
        return round(self.x), round(self.y)

    def reset(self) -> None:
        self.x = None
        self.y = None


def map_to_preview(
    point: tuple[int, int],
    active_area: ActiveRectangle,
    frame_width: int,
    frame_height: int,
) -> tuple[int, int]:
    """Map and clamp a point from the active rectangle to the full preview."""
    point_x, point_y = point
    mapped_x = np.interp(
        point_x,
        (active_area.left, active_area.right),
        (0, frame_width - 1),
    )
    mapped_y = np.interp(
        point_y,
        (active_area.top, active_area.bottom),
        (0, frame_height - 1),
    )
    return round(float(mapped_x)), round(float(mapped_y))


APPLE_BLUE = (255, 132, 10)
APPLE_GREEN = (88, 209, 48)
APPLE_ORANGE = (10, 149, 255)
APPLE_RED = (69, 68, 255)
GLASS_WHITE = (246, 246, 246)


def _rounded_rectangle(image, start, end, color, radius=16, thickness=-1) -> None:
    """Draw a rounded rectangle using only portable OpenCV primitives."""
    x1, y1 = start
    x2, y2 = end
    radius = max(1, min(radius, (x2 - x1) // 2, (y2 - y1) // 2))
    if thickness < 0:
        cv2.rectangle(image, (x1 + radius, y1), (x2 - radius, y2), color, -1)
        cv2.rectangle(image, (x1, y1 + radius), (x2, y2 - radius), color, -1)
        for center in (
            (x1 + radius, y1 + radius),
            (x2 - radius, y1 + radius),
            (x1 + radius, y2 - radius),
            (x2 - radius, y2 - radius),
        ):
            cv2.circle(image, center, radius, color, -1, cv2.LINE_AA)
        return
    cv2.line(image, (x1 + radius, y1), (x2 - radius, y1), color, thickness, cv2.LINE_AA)
    cv2.line(image, (x1 + radius, y2), (x2 - radius, y2), color, thickness, cv2.LINE_AA)
    cv2.line(image, (x1, y1 + radius), (x1, y2 - radius), color, thickness, cv2.LINE_AA)
    cv2.line(image, (x2, y1 + radius), (x2, y2 - radius), color, thickness, cv2.LINE_AA)
    for center, start_angle in (
        ((x1 + radius, y1 + radius), 180),
        ((x2 - radius, y1 + radius), 270),
        ((x2 - radius, y2 - radius), 0),
        ((x1 + radius, y2 - radius), 90),
    ):
        cv2.ellipse(
            image,
            center,
            (radius, radius),
            0,
            start_angle,
            start_angle + 90,
            color,
            thickness,
            cv2.LINE_AA,
        )


def _draw_glass_card(frame, start, end, radius=20, tint=(28, 28, 30)) -> None:
    """Blur camera content beneath an adaptive, softly bordered control card."""
    frame_height, frame_width = frame.shape[:2]
    x1, y1 = max(0, start[0]), max(0, start[1])
    x2, y2 = min(frame_width - 1, end[0]), min(frame_height - 1, end[1])
    if x2 <= x1 or y2 <= y1:
        return
    region = frame[y1:y2, x1:x2]
    blurred = cv2.GaussianBlur(region, (0, 0), 7.0)
    tint_layer = np.full_like(region, tint)
    glass = cv2.addWeighted(blurred, 0.64, tint_layer, 0.36, 0)
    mask = np.zeros(region.shape[:2], dtype=np.uint8)
    _rounded_rectangle(mask, (0, 0), (x2 - x1 - 1, y2 - y1 - 1), 255, radius, -1)
    region[mask > 0] = glass[mask > 0]
    _rounded_rectangle(frame, (x1, y1), (x2, y2), (205, 205, 210), radius, 1)
    cv2.line(
        frame,
        (x1 + radius, y1 + 1),
        (x2 - radius, y1 + 1),
        (245, 245, 248),
        1,
        cv2.LINE_AA,
    )


def _draw_status_pill(frame, text, origin, color) -> None:
    text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.46, 1)
    x, y = origin
    width = text_size[0] + 24
    overlay = frame.copy()
    _rounded_rectangle(overlay, (x, y), (x + width, y + 26), color, 13, -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)
    cv2.putText(
        frame,
        text,
        (x + 12, y + 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        GLASS_WHITE,
        1,
        cv2.LINE_AA,
    )


def draw_active_area(frame, active_area: ActiveRectangle, enabled: bool) -> None:
    color = APPLE_BLUE if enabled else (150, 150, 155)
    _rounded_rectangle(
        frame,
        (active_area.left, active_area.top),
        (active_area.right, active_area.bottom),
        color,
        18,
        2,
    )
    label = "POINTER AREA"
    _draw_status_pill(
        frame,
        label,
        (active_area.left + 10, active_area.top + 10),
        (78, 78, 82),
    )


def draw_status_panel(
    frame,
    lock_states: dict[str, HandLockState],
    seen_sides: set[str],
    cursor_visible: bool,
    fps: float,
    swap_handedness: bool,
    system_control_available: bool = False,
    system_control_active: bool = False,
    control_panel_available: bool = False,
    left_action_status: str | None = None,
    right_action_status: str | None = None,
) -> None:
    panel_right = min(frame.shape[1] - 12, 470)
    _draw_glass_card(frame, (12, 12), (panel_right, 180), radius=22)

    if lock_states["LEFT"].locked:
        left_status = "LOCKED"
    elif cursor_visible:
        left_status = "POINTER"
    else:
        left_status = "READY"
    if not lock_states["LEFT"].locked and left_action_status:
        left_status = left_action_status

    right_status = "LOCKED" if lock_states["RIGHT"].locked else "READY"
    if not lock_states["RIGHT"].locked and right_action_status:
        right_status = right_action_status
    left_suffix = " (NOT SEEN)" if "LEFT" not in seen_sides else ""
    right_suffix = " (NOT SEEN)" if "RIGHT" not in seen_sides else ""

    airdesk_status = "LIVE" if system_control_active else "STOPPED"
    cv2.putText(
        frame,
        "AirDesk",
        (28, 44),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.76,
        GLASS_WHITE,
        2,
        cv2.LINE_AA,
    )
    status_color = APPLE_GREEN if system_control_active else APPLE_RED
    _draw_status_pill(frame, airdesk_status, (panel_right - 112, 23), status_color)

    left_color = APPLE_RED if lock_states["LEFT"].locked else APPLE_BLUE
    right_color = APPLE_RED if lock_states["RIGHT"].locked else APPLE_BLUE
    cv2.putText(
        frame,
        f"LEFT   {left_status}{left_suffix}",
        (28, 78),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        left_color,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"RIGHT  {right_status}{right_suffix}",
        (28, 108),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        right_color,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "DESKTOP  •  HANDS AUTO-CORRECTED",
        (28, 136),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        (205, 205, 210),
        1,
        cv2.LINE_AA,
    )

    controls = "Q: quit"
    if control_panel_available:
        controls += " | P: guide"
    if system_control_available:
        controls += " | M: mouse | Esc: STOP"
    cv2.putText(
        frame,
        f"FPS: {fps:.1f} | {controls}",
        (28, 163),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (220, 220, 224),
        1,
        cv2.LINE_AA,
    )


def draw_virtual_cursor(
    frame,
    raw_index_position: tuple[int, int],
    cursor_position: tuple[int, int],
) -> None:
    """Distinguish the raw fingertip sample from the mapped virtual cursor."""
    cv2.circle(frame, raw_index_position, 6, APPLE_ORANGE, -1, cv2.LINE_AA)
    cv2.line(
        frame,
        raw_index_position,
        cursor_position,
        (185, 185, 190),
        1,
        cv2.LINE_AA,
    )
    cv2.circle(frame, cursor_position, 16, GLASS_WHITE, 3, cv2.LINE_AA)
    cv2.circle(frame, cursor_position, 10, APPLE_BLUE, -1, cv2.LINE_AA)


def draw_glass_message(frame, text: str, baseline_y: int, *, emphasized=False) -> None:
    """Render transient diagnostics as compact glass control capsules."""
    scale = 0.56 if emphasized else 0.46
    thickness = 2 if emphasized else 1
    text_size, _ = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness
    )
    width = min(text_size[0] + 30, frame.shape[1] - 24)
    left = max(12, (frame.shape[1] - width) // 2)
    top = max(4, baseline_y - 25)
    _draw_glass_card(frame, (left, top), (left + width, baseline_y + 8), radius=16)
    color = APPLE_BLUE if emphasized else (230, 230, 234)
    cv2.putText(
        frame,
        text,
        (left + 15, baseline_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def run_virtual_cursor(
    system_mouse=None,
    window_name: str = WINDOW_NAME,
    mode_controller=None,
    control_panel=None,
) -> int:
    """Own controllers for the entire run, including early startup failures."""
    try:
        return _run_virtual_cursor(system_mouse, window_name, mode_controller, control_panel)
    except KeyboardInterrupt:
        print("AirDesk closed.")
        return 0
    except Exception as error:
        print(f"AirDesk stopped: {error}")
        return 1
    finally:
        for resource in (system_mouse, mode_controller, control_panel):
            if resource is not None:
                try:
                    resource.close()
                except Exception as error:
                    print(f"AirDesk cleanup warning: {error}")


def _run_virtual_cursor(system_mouse, window_name, mode_controller, control_panel) -> int:
    """Run Stage 4, optionally with the explicit Stage 5 mouse adapter."""
    if not MODEL_PATH.exists():
        print("The Hand Landmarker model is missing.")
        print("Run: python scripts/download_hand_model.py")
        return 1

    camera_device = cv2.VideoCapture(CAMERA_INDEX)
    if not camera_device.isOpened():
        print(
            "AirDesk could not open the camera. Check macOS Camera permission "
            "for the app running Python, then try again."
        )
        camera_device.release()
        return 1
    configure_camera(camera_device)
    camera = LatestFrameCapture(camera_device).start()

    lock_states = {"LEFT": HandLockState(), "RIGHT": HandLockState()}
    cursor_smoother = CursorSmoother()
    swap_handedness = DEFAULT_SWAP_HANDEDNESS
    start_time = time.perf_counter()
    previous_time = start_time
    previous_timestamp_ms = -1
    smoothed_fps = 0.0
    preview_window_open = False
    last_panel_status_at = 0.0
    last_frame_total_ms = 0.0

    try:
        with create_landmarker() as landmarker:
            while True:
                frame_started_at = time.perf_counter()
                frame_received, frame = camera.read()
                if not frame_received:
                    print("AirDesk stopped because it could not read a camera frame.")
                    return 1

                frame = cv2.flip(frame, 1)
                frame = limit_frame_resolution(frame)
                frame_height, frame_width = frame.shape[:2]
                active_area = ActiveRectangle.from_frame(frame_width, frame_height)

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                media_pipe_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb_frame,
                )
                timestamp_ms = int((time.perf_counter() - start_time) * 1000)
                timestamp_ms = max(timestamp_ms, previous_timestamp_ms + 1)
                previous_timestamp_ms = timestamp_ms
                result = landmarker.detect_for_video(media_pipe_image, timestamp_ms)

                current_time = time.perf_counter()
                # Stop before moving the pointer or polling completed OCR.
                escape_monitor = getattr(system_mouse, "escape_monitor", None)
                if escape_monitor is not None and escape_monitor.consume_escape():
                    system_mouse.disable("global Esc pressed")
                    cursor_smoother.reset()
                    if mode_controller is not None:
                        if mode_controller.is_overlay_active:
                            mode_controller._set_mode(
                                "DESKTOP", system_mouse, resume_desktop=False
                            )
                        mode_controller.emergency_stop()
                frame_seconds = current_time - previous_time
                previous_time = current_time
                instant_fps = 1.0 / frame_seconds if frame_seconds > 0 else 0.0
                smoothed_fps = (
                    instant_fps
                    if smoothed_fps == 0.0
                    else (0.9 * smoothed_fps) + (0.1 * instant_fps)
                )

                observations: dict[str, tuple[float, object, FistMetrics]] = {}
                for hand_index, landmarks in enumerate(result.hand_landmarks):
                    if hand_index >= len(result.handedness) or not result.handedness[hand_index]:
                        continue
                    category = result.handedness[hand_index][0]
                    score = float(category.score)
                    side = resolve_hand_side(category.category_name, swap_handedness)
                    if side is None or score < MIN_HANDEDNESS_CONFIDENCE:
                        continue

                    observation = (
                        score,
                        landmarks,
                        classify_fist(landmarks),
                    )
                    existing = observations.get(side)
                    if existing is None or observation[0] > existing[0]:
                        observations[side] = observation

                seen_sides = set(observations)
                for side, state in lock_states.items():
                    observation = observations.get(side)
                    if observation is None:
                        state.update(False, False, current_time)
                    else:
                        _, _, metrics = observation
                        state.update(metrics.is_closed, True, current_time)

                if mode_controller is not None:
                    mode_controller.update(
                        observations,
                        lock_states,
                        current_time,
                        frame_width,
                        frame_height,
                        system_mouse,
                        all_hands=result.hand_landmarks,
                    )
                    overlay_active = getattr(
                        mode_controller,
                        "is_overlay_active",
                        mode_controller.is_whiteboard,
                    )
                    if overlay_active:
                        # The preview must disappear so the already-focused app
                        # remains visible beneath the non-activating overlay.
                        if preview_window_open:
                            cv2.destroyWindow(window_name)
                            preview_window_open = False
                        if control_panel is not None:
                            if (
                                current_time - last_panel_status_at
                                >= PANEL_STATUS_INTERVAL_SECONDS
                            ):
                                control_panel.update_status(
                                    build_control_panel_status(
                                        system_enabled=bool(
                                            system_mouse is not None
                                            and system_mouse.enabled
                                        ),
                                        paused=bool(
                                            getattr(system_mouse, "paused", False)
                                        ),
                                        left_locked=lock_states["LEFT"].locked,
                                        right_locked=lock_states["RIGHT"].locked,
                                        cursor_visible=False,
                                        writing_active=True,
                                        left_action=None,
                                        right_action=None,
                                        fps=smoothed_fps,
                                        processing_ms=last_frame_total_ms,
                                    )
                                )
                                last_panel_status_at = current_time
                            control_panel.pump()
                        else:
                            mode_controller.pump_overlay()
                        last_frame_total_ms = (
                            time.perf_counter() - frame_started_at
                        ) * 1000.0
                        continue

                for side, (_, landmarks, _) in observations.items():
                    text_row = 205 if side == "LEFT" else 235
                    draw_hand(frame, landmarks, side.title(), text_row)

                raw_index_position = None
                cursor_position = None
                left_observation = observations.get("LEFT")
                if left_observation is not None and not lock_states["LEFT"].locked:
                    left_landmarks = left_observation[1]
                    pointing = classify_index_point(left_landmarks)
                    if pointing.is_pointing:
                        raw_index_position = normalized_to_pixel(
                            left_landmarks[8], frame_width, frame_height
                        )
                        target_position = map_to_preview(
                            raw_index_position,
                            active_area,
                            frame_width,
                            frame_height,
                        )
                        cursor_position = cursor_smoother.update(target_position)
                    else:
                        cursor_smoother.reset()
                else:
                    cursor_smoother.reset()

                # Put the real pointer at this frame's newest smoothed position
                # before processing a right-hand click. This matters for small
                # targets such as macOS menu-bar icons.
                if (
                    cursor_position is not None
                    and system_mouse is not None
                    and system_mouse.enabled
                ):
                    system_mouse.move_from_preview(
                        cursor_position,
                        (frame_width, frame_height),
                    )

                if system_mouse is not None and hasattr(system_mouse, "update_gestures"):
                    system_mouse.update_gestures(
                        observations,
                        lock_states,
                        current_time,
                        cursor_position=cursor_position,
                        preview_size=(frame_width, frame_height),
                    )

                draw_active_area(
                    frame,
                    active_area,
                    enabled=cursor_position is not None,
                )
                draw_status_panel(
                    frame,
                    lock_states,
                    seen_sides,
                    cursor_visible=cursor_position is not None,
                    fps=smoothed_fps,
                    swap_handedness=swap_handedness,
                    system_control_available=system_mouse is not None,
                    system_control_active=(
                        system_mouse is not None and system_mouse.enabled
                    ),
                    control_panel_available=control_panel is not None,
                    left_action_status=(
                        getattr(system_mouse, "left_status", None)
                        if system_mouse is not None
                        else None
                    ),
                    right_action_status=(
                        getattr(system_mouse, "right_status", None)
                        if system_mouse is not None
                        else None
                    ),
                )
                if raw_index_position is not None and cursor_position is not None:
                    draw_virtual_cursor(frame, raw_index_position, cursor_position)

                feedback_text = (
                    getattr(system_mouse, "feedback_text", "")
                    if system_mouse is not None
                    else ""
                )
                diagnostic_text = (
                    getattr(system_mouse, "diagnostic_text", "")
                    if system_mouse is not None
                    else ""
                )
                if diagnostic_text:
                    draw_glass_message(
                        frame,
                        diagnostic_text,
                        frame_height - 62,
                    )
                if feedback_text:
                    draw_glass_message(
                        frame,
                        feedback_text,
                        frame_height - 27,
                        emphasized=True,
                    )

                if mode_controller is not None:
                    mode_controller.draw_desktop_overlay(frame)

                if control_panel is not None:
                    if (
                        current_time - last_panel_status_at
                        >= PANEL_STATUS_INTERVAL_SECONDS
                    ):
                        control_panel.update_status(
                            build_control_panel_status(
                                system_enabled=bool(
                                    system_mouse is not None and system_mouse.enabled
                                ),
                                paused=bool(getattr(system_mouse, "paused", False)),
                                left_locked=lock_states["LEFT"].locked,
                                right_locked=lock_states["RIGHT"].locked,
                                cursor_visible=cursor_position is not None,
                                left_action=(
                                    getattr(system_mouse, "left_status", None)
                                    if system_mouse is not None
                                    else None
                                ),
                                right_action=(
                                    getattr(system_mouse, "right_status", None)
                                    if system_mouse is not None
                                    else None
                                ),
                                fps=smoothed_fps,
                                processing_ms=last_frame_total_ms,
                            )
                        )
                        last_panel_status_at = current_time
                    control_panel.pump()

                cv2.imshow(window_name, frame)
                preview_window_open = True
                key = cv2.waitKey(1) & 0xFF
                last_frame_total_ms = (
                    time.perf_counter() - frame_started_at
                ) * 1000.0
                if key in (ord("q"), ord("Q")):
                    break
                if key == 27:
                    if system_mouse is not None:
                        system_mouse.disable("Esc pressed")
                    cursor_smoother.reset()
                if (
                    key in (ord("p"), ord("P"))
                    and control_panel is not None
                ):
                    control_panel.toggle_visibility()
                if key in (ord("m"), ord("M")) and system_mouse is not None:
                    system_mouse.toggle()
                    cursor_smoother.reset()
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break
    finally:
        try:
            if system_mouse is not None:
                system_mouse.disable("camera loop stopped")
        finally:
            camera.release()
            cv2.destroyAllWindows()

    return 0


def main() -> int:
    """Run the Stage 4 preview with no possible system mouse output."""
    return run_virtual_cursor()


if __name__ == "__main__":
    raise SystemExit(main())
