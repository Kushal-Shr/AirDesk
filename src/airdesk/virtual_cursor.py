"""Stage 4: move a safe on-screen cursor with the unlocked left index finger."""

from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import mediapipe as mp
import numpy as np

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
CURSOR_SMOOTHING = 0.22
MIN_HANDEDNESS_CONFIDENCE = 0.65


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


def draw_active_area(frame, active_area: ActiveRectangle, enabled: bool) -> None:
    color = (0, 255, 0) if enabled else (130, 130, 130)
    cv2.rectangle(
        frame,
        (active_area.left, active_area.top),
        (active_area.right, active_area.bottom),
        color,
        2,
    )
    cv2.putText(
        frame,
        "LEFT-HAND ACTIVE AREA",
        (active_area.left + 8, active_area.top + 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        cv2.LINE_AA,
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
    left_action_status: str | None = None,
    right_action_status: str | None = None,
) -> None:
    overlay = frame.copy()
    panel_right = min(frame.shape[1] - 10, 500)
    cv2.rectangle(overlay, (10, 10), (panel_right, 205), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)

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

    airdesk_status = "ACTIVE" if system_control_active else "SAFE PREVIEW"
    rows = (
        (f"LEFT: {left_status}{left_suffix}", lock_states["LEFT"].locked),
        (f"RIGHT: {right_status}{right_suffix}", lock_states["RIGHT"].locked),
        ("MODE: DESKTOP", False),
        (f"AIRDESK: {airdesk_status}", system_control_active),
        (
            f"HAND LABELS: {'SWAPPED' if swap_handedness else 'NORMAL'} (H: toggle)",
            False,
        ),
    )
    for row_index, (text, is_locked) in enumerate(rows):
        color = (0, 80, 255) if is_locked else (0, 255, 0)
        cv2.putText(
            frame,
            text,
            (24, 36 + row_index * 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.64,
            color,
            2,
            cv2.LINE_AA,
        )

    controls = "Q: quit | H: hands"
    if system_control_available:
        controls = "Q: quit | H: hands | M: mouse | Esc: SAFE"
    cv2.putText(
        frame,
        f"FPS: {fps:.1f} | {controls}",
        (24, 197),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.47,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def draw_virtual_cursor(
    frame,
    raw_index_position: tuple[int, int],
    cursor_position: tuple[int, int],
) -> None:
    """Distinguish the raw fingertip sample from the mapped virtual cursor."""
    cv2.circle(frame, raw_index_position, 7, (255, 0, 255), -1, cv2.LINE_AA)
    cv2.line(
        frame,
        raw_index_position,
        cursor_position,
        (150, 150, 150),
        1,
        cv2.LINE_AA,
    )
    cv2.circle(frame, cursor_position, 15, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.circle(frame, cursor_position, 10, (255, 255, 0), -1, cv2.LINE_AA)


def run_virtual_cursor(system_mouse=None, window_name: str = WINDOW_NAME) -> int:
    """Run Stage 4, optionally with the explicit Stage 5 mouse adapter."""
    if not MODEL_PATH.exists():
        print("The Hand Landmarker model is missing.")
        print("Run: python scripts/download_hand_model.py")
        return 1

    camera = cv2.VideoCapture(CAMERA_INDEX)
    if not camera.isOpened():
        print(
            "AirDesk could not open the camera. Check macOS Camera permission "
            "for the app running Python, then try again."
        )
        camera.release()
        return 1

    lock_states = {"LEFT": HandLockState(), "RIGHT": HandLockState()}
    cursor_smoother = CursorSmoother()
    swap_handedness = False
    start_time = time.perf_counter()
    previous_time = start_time
    previous_timestamp_ms = -1
    smoothed_fps = 0.0

    try:
        with create_landmarker() as landmarker:
            while True:
                frame_received, frame = camera.read()
                if not frame_received:
                    print("AirDesk stopped because it could not read a camera frame.")
                    return 1

                frame = cv2.flip(frame, 1)
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
                    system_mouse.update_gestures(observations, lock_states, current_time)

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
                    cv2.putText(
                        frame,
                        diagnostic_text,
                        (20, frame_height - 62),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.52,
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )
                if feedback_text:
                    cv2.putText(
                        frame,
                        feedback_text,
                        (max(20, frame_width // 2 - 150), frame_height - 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75,
                        (0, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )

                cv2.imshow(window_name, frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q")):
                    break
                if key == 27:
                    if system_mouse is not None:
                        system_mouse.disable("Esc pressed")
                    cursor_smoother.reset()
                if key in (ord("h"), ord("H")):
                    swap_handedness = not swap_handedness
                    # Changing hand ownership is a safety boundary: relock both
                    # sides and discard the old cursor before using the new map.
                    lock_states = {"LEFT": HandLockState(), "RIGHT": HandLockState()}
                    cursor_smoother.reset()
                    if system_mouse is not None:
                        system_mouse.disable("hand labels changed")
                    mode = "SWAPPED" if swap_handedness else "NORMAL"
                    print(f"Hand label correction: {mode}")
                if key in (ord("m"), ord("M")) and system_mouse is not None:
                    system_mouse.toggle()
                    cursor_smoother.reset()
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break
    finally:
        if system_mouse is not None:
            system_mouse.close()
        camera.release()
        cv2.destroyAllWindows()

    return 0


def main() -> int:
    """Run the Stage 4 preview with no possible system mouse output."""
    return run_virtual_cursor()


if __name__ == "__main__":
    raise SystemExit(main())
