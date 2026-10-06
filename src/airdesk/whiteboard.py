"""Stage 8 gesture state for transparent desktop air-writing."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import cv2
import numpy as np

from .air_mouse import map_preview_to_screen
from .character_recognition import EmnistCharacterRecognizer, make_writing_guide
from .desktop_controls import _landmark_points, finger_is_extended
from .hand_lock import (
    FINGER_JOINTS,
    finger_is_curled,
    joint_angle_degrees,
    thumb_is_raised,
)
from .native_overlay import MemoryInkOverlay
from .virtual_cursor import ActiveRectangle, map_to_preview


MODE_HOLD_SECONDS = 1.0
CLEAR_HOLD_SECONDS = 1.0
PEN_PINCH_THRESHOLD = 0.24
GESTURE_PINCH_THRESHOLD = 0.38
MIN_DRAW_FINGER_OPENNESS = 0.80
PEN_THICKNESS = 6
ACCEPT_HOLD_SECONDS = 0.65
SPACE_HOLD_SECONDS = 0.65
UNDO_HOLD_SECONDS = 0.65
INSERT_HOLD_SECONDS = 0.90
MODE_SWITCH_HOLD_SECONDS = 0.80
MAX_SENTENCE_CHARACTERS = 160
LINE_IDLE_SECONDS = 3.0
MIN_LINE_CONFIDENCE = 0.90
MIN_GEMINI_INSERT_CONFIDENCE = 0.90


def is_open_palm(landmarks) -> bool:
    """Require four extended fingers without a competing thumb/index pinch."""
    points = _landmark_points(landmarks)
    wrist = points[0]
    return bool(
        all(
            finger_is_extended(points, wrist, *finger_joints)
            for finger_joints in FINGER_JOINTS
        )
        and _normalized_pinch_distance(points) >= GESTURE_PINCH_THRESHOLD
    )


@dataclass(frozen=True)
class PenPoseMetrics:
    """Explain whether the strict air-writing pose is currently valid."""

    is_drawing: bool
    pinch_distance: float
    other_finger_openness: tuple[float, float, float]

    @property
    def minimum_other_openness(self) -> float:
        return min(self.other_finger_openness)


@dataclass
class AdaptivePointSmoother:
    """Suppress small landmark jitter while following deliberate moves quickly."""

    slow_amount: float = 0.18
    fast_amount: float = 0.62
    full_speed_distance: float = 55.0
    x: float | None = None
    y: float | None = None

    def update(self, target: tuple[int, int]) -> tuple[int, int]:
        target_x, target_y = target
        if self.x is None or self.y is None:
            self.x = float(target_x)
            self.y = float(target_y)
        else:
            movement = float(np.hypot(target_x - self.x, target_y - self.y))
            response = min(movement / self.full_speed_distance, 1.0)
            amount = self.slow_amount + response * (
                self.fast_amount - self.slow_amount
            )
            self.x += amount * (target_x - self.x)
            self.y += amount * (target_y - self.y)
        return round(self.x), round(self.y)

    def reset(self) -> None:
        self.x = None
        self.y = None


def _finger_openness(
    points: np.ndarray,
    wrist: np.ndarray,
    mcp_id: int,
    pip_id: int,
    tip_id: int,
) -> float:
    """Return a 0..1 extension score from joint angle and fingertip reach."""
    angle = joint_angle_degrees(points[mcp_id], points[pip_id], points[tip_id])
    angle_score = float(np.clip((angle - 120.0) / 50.0, 0.0, 1.0))
    pip_reach = max(float(np.linalg.norm(points[pip_id] - wrist)), 1e-6)
    tip_reach_ratio = float(np.linalg.norm(points[tip_id] - wrist) / pip_reach)
    reach_score = float(np.clip((tip_reach_ratio - 1.0) / 0.25, 0.0, 1.0))
    return min(angle_score, reach_score)


def _normalized_pinch_distance(points: np.ndarray) -> float:
    palm_size = max(float(np.linalg.norm(points[9] - points[0])), 1e-6)
    return float(np.linalg.norm(points[4] - points[8]) / palm_size)


def is_thumb_index_pinch(landmarks) -> bool:
    """Detect the looser thumb/index pinch used by non-writing gestures."""
    points = _landmark_points(landmarks)
    return _normalized_pinch_distance(points) < GESTURE_PINCH_THRESHOLD


def is_stroke_undo_pose(landmarks) -> bool:
    """Use a left pinch with the middle, ring, and little fingers extended."""
    points = _landmark_points(landmarks)
    wrist = points[0]
    return bool(
        _normalized_pinch_distance(points) < PEN_PINCH_THRESHOLD
        and finger_is_extended(points, wrist, *FINGER_JOINTS[1])
        and finger_is_extended(points, wrist, *FINGER_JOINTS[2])
        and finger_is_extended(points, wrist, *FINGER_JOINTS[3])
    )


def measure_pen_pose(landmarks) -> PenPoseMetrics:
    """Require a tight pinch and 80% extension of the other three fingers."""
    points = _landmark_points(landmarks)
    wrist = points[0]
    pinch_distance = _normalized_pinch_distance(points)
    openness = tuple(
        _finger_openness(points, wrist, *finger_joints)
        for finger_joints in FINGER_JOINTS[1:]
    )
    is_drawing = bool(
        pinch_distance < PEN_PINCH_THRESHOLD
        and all(score >= MIN_DRAW_FINGER_OPENNESS for score in openness)
    )
    return PenPoseMetrics(is_drawing, pinch_distance, openness)


def is_pen_pinch(landmarks) -> bool:
    """Return whether the complete strict writing pose is active."""
    return measure_pen_pose(landmarks).is_drawing


def is_thumbs_up(landmarks) -> bool:
    """Require a raised thumb while all four other fingers are curled."""
    points = _landmark_points(landmarks)
    wrist = points[0]
    fingers_curled = all(
        finger_is_curled(points, wrist, *finger_joints)
        for finger_joints in FINGER_JOINTS
    )
    return fingers_curled and thumb_is_raised(points)


def is_mode_switch_pose(landmarks) -> bool:
    """Require exactly the left index and middle fingers to be extended."""
    points = _landmark_points(landmarks)
    wrist = points[0]
    return bool(
        finger_is_extended(points, wrist, *FINGER_JOINTS[0])
        and finger_is_extended(points, wrist, *FINGER_JOINTS[1])
        and finger_is_curled(points, wrist, *FINGER_JOINTS[2])
        and finger_is_curled(points, wrist, *FINGER_JOINTS[3])
    )


@dataclass
class HoldLatch:
    """Fire once after a pose is held, then require its release."""

    duration: float
    started_at: float | None = None
    latched: bool = False
    progress: float = 0.0
    last_active_at: float | None = None
    release_grace: float = 0.18

    def reset(self) -> None:
        self.started_at = None
        self.latched = False
        self.progress = 0.0
        self.last_active_at = None

    def update(self, active: bool, now: float) -> bool:
        if not active:
            if self.last_active_at is not None and now - self.last_active_at <= self.release_grace:
                return False
            self.reset()
            return False
        self.last_active_at = now
        if self.latched:
            self.progress = 1.0
            return False
        if self.started_at is None:
            self.started_at = now
        self.progress = min((now - self.started_at) / self.duration, 1.0)
        if self.progress < 1.0:
            return False
        self.latched = True
        return True


@dataclass
class SentenceBuffer:
    """Build editable text without sending keyboard output prematurely."""

    text: str = ""
    maximum_length: int = MAX_SENTENCE_CHARACTERS

    def append_character(self, character: str) -> bool:
        if not character or len(self.text) + len(character) > self.maximum_length:
            return False
        self.text += character
        return True

    def append_space(self) -> bool:
        if not self.text or self.text.endswith(" ") or len(self.text) >= self.maximum_length:
            return False
        self.text += " "
        return True

    def undo(self) -> bool:
        if not self.text:
            return False
        self.text = self.text[:-1]
        return True

    def clear(self) -> None:
        self.text = ""


class AirWritingController:
    """Own Desktop/Air Write mode, visible overlay, and an OCR-ready canvas."""

    def __init__(self, overlay=None, recognizer=None, sentence_mode: bool = True) -> None:
        self.overlay = overlay if overlay is not None else MemoryInkOverlay()
        self.recognizer = (
            recognizer if recognizer is not None else EmnistCharacterRecognizer()
        )
        self.mode = "DESKTOP"
        self.canvas = None
        self.mode_hold = HoldLatch(MODE_HOLD_SECONDS)
        self.clear_hold = HoldLatch(CLEAR_HOLD_SECONDS)
        self.previous_pen_point = None
        self.pen_down = False
        self.canvas_strokes: list[list[tuple[int, int]]] = []
        self.pointer_smoother = AdaptivePointSmoother()
        self.pen_pose_metrics = None
        self.canvas_cleared_feedback = False
        self.open_palm_count = 0
        self.accept_hold = HoldLatch(ACCEPT_HOLD_SECONDS)
        self.space_hold = HoldLatch(SPACE_HOLD_SECONDS)
        self.undo_hold = HoldLatch(UNDO_HOLD_SECONDS)
        self.insert_hold = HoldLatch(INSERT_HOLD_SECONDS)
        self.mode_switch_hold = HoldLatch(MODE_SWITCH_HOLD_SECONDS)
        self.sentence_mode = sentence_mode
        self.sentence = SentenceBuffer()
        self.recognition_feedback = ""
        self.text_output_allowed = True
        self.recognition_mode = getattr(self.recognizer, "mode", "uppercase")
        self.writing_guide = None
        self._configure_recognition_mode()
        self._refresh_sentence_preview()

    @property
    def is_overlay_active(self) -> bool:
        return self.mode == "AIR_WRITE"

    @property
    def is_whiteboard(self) -> bool:
        """Compatibility name used by the first Stage 8 draft."""
        return self.is_overlay_active

    def ensure_canvas(self, frame_width: int, frame_height: int) -> None:
        expected_shape = (frame_height, frame_width, 3)
        if self.canvas is None:
            self.canvas = np.full(expected_shape, 255, dtype=np.uint8)
        elif self.canvas.shape != expected_shape:
            self.canvas = cv2.resize(self.canvas, (frame_width, frame_height))

    def _lift_pen(self) -> None:
        if self.pen_down or self.previous_pen_point is not None:
            self.overlay.end_stroke()
        self.previous_pen_point = None
        self.pen_down = False

    def reset_transient(self) -> None:
        self._lift_pen()
        self.pointer_smoother.reset()
        self.pen_pose_metrics = None
        self.overlay.hide_cursor()
        self.clear_hold.reset()
        self.accept_hold.reset()
        self.space_hold.reset()
        self.undo_hold.reset()
        self.insert_hold.reset()
        self.mode_switch_hold.reset()

    def _configure_recognition_mode(self) -> None:
        self.recognition_mode = getattr(self.recognizer, "mode", "uppercase")
        if getattr(self.recognizer, "preserves_position", False):
            self.writing_guide = make_writing_guide(
                self.overlay.size, self.recognition_mode.upper()
            )
            if self.is_overlay_active:
                self.overlay.set_guide(self.writing_guide)
        else:
            self.writing_guide = None
            self.overlay.clear_guide()

    def _cycle_recognition_mode(self) -> None:
        self._clear_writing()
        new_mode = self.recognizer.cycle_mode()
        self._configure_recognition_mode()
        self.recognition_feedback = f"CHARACTER MODE: {new_mode.upper()}"

    def _refresh_sentence_preview(self) -> None:
        preview = self.sentence.text
        if len(preview) > 90:
            preview = f"…{preview[-89:]}"
        self.overlay.set_preview(preview if self.sentence_mode else "")

    def _map_right_index(
        self,
        landmarks,
        frame_width: int,
        frame_height: int,
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        active_area = ActiveRectangle.from_frame(frame_width, frame_height)
        index_tip = landmarks[8]
        raw_point = (
            round(index_tip.x * frame_width),
            round(index_tip.y * frame_height),
        )
        raw_pen_point = map_to_preview(
            raw_point,
            active_area,
            frame_width,
            frame_height,
        )
        pen_point = self.pointer_smoother.update(raw_pen_point)
        screen_point = map_preview_to_screen(
            pen_point,
            (frame_width, frame_height),
            self.overlay.size,
        )
        return pen_point, screen_point

    def _set_mode(self, mode: str, system_controller) -> None:
        self.mode = mode
        self.reset_transient()
        if system_controller is not None:
            system_controller.disable(f"entered {mode.lower()} mode")
        if self.is_overlay_active:
            self.text_output_allowed = True
            if self.writing_guide is not None:
                self.overlay.set_guide(self.writing_guide)
            self.overlay.show()
        else:
            self.overlay.clear_guide()
            self.overlay.hide()
        print(f"AirDesk mode: {mode} (real system output is OFF)")

    def _update_status(self, lock_states) -> None:
        right = "DRAW" if self.pen_down else (
            "LOCKED" if lock_states["RIGHT"].locked else "READY"
        )
        if self.clear_hold.progress > 0:
            left = f"CLEAR {round(self.clear_hold.progress * 100)}%"
        else:
            left = "LOCKED" if lock_states["LEFT"].locked else "READY"
        if self.accept_hold.progress > 0 and not self.accept_hold.latched:
            right = f"CHAR {round(self.accept_hold.progress * 100)}%"
        if self.insert_hold.progress > 0 and not self.insert_hold.latched:
            right = f"INSERT {round(self.insert_hold.progress * 100)}%"
        if self.space_hold.progress > 0 and not self.space_hold.latched:
            left = f"SPACE {round(self.space_hold.progress * 100)}%"
        if self.undo_hold.progress > 0 and not self.undo_hold.latched:
            left = f"UNDO {round(self.undo_hold.progress * 100)}%"
        if self.mode_switch_hold.progress > 0 and not self.mode_switch_hold.latched:
            left = f"MODE {round(self.mode_switch_hold.progress * 100)}%"
        feedback = f"  |  {self.recognition_feedback}" if self.recognition_feedback else ""
        controls = (
            "R👍 char | L👍 space | 👍👍 insert | L✌ mode | L pinch undo"
            if self.sentence_mode
            else "👍 save"
        )
        self.overlay.set_status(
            f"AIR WRITE [{self.recognition_mode.upper()}] | L: {left} | "
            f"R: {right} | {controls}{feedback}"
        )

    def _clear_writing(self) -> None:
        self.canvas.fill(255)
        self.canvas_strokes.clear()
        self.overlay.clear()
        self._lift_pen()

    def _draw_pen_point(
        self,
        pen_point: tuple[int, int],
        screen_point: tuple[int, int],
    ) -> None:
        """Record one synchronized canvas/overlay point for later undo."""
        if not self.pen_down:
            self.canvas_strokes.append([])
        self.canvas_strokes[-1].append(pen_point)
        if self.previous_pen_point is not None:
            cv2.line(
                self.canvas,
                self.previous_pen_point,
                pen_point,
                (20, 20, 20),
                PEN_THICKNESS,
                cv2.LINE_AA,
            )
        self.overlay.add_point(screen_point)
        self.overlay.set_cursor(screen_point, active=True)
        self.previous_pen_point = pen_point
        self.pen_down = True

    def _rebuild_canvas_from_strokes(self) -> None:
        """Re-rasterize retained vector strokes after removing the newest one."""
        self.canvas.fill(255)
        for stroke in self.canvas_strokes:
            for start, end in zip(stroke, stroke[1:]):
                cv2.line(
                    self.canvas,
                    start,
                    end,
                    (20, 20, 20),
                    PEN_THICKNESS,
                    cv2.LINE_AA,
                )

    def _undo_last_ink_stroke(self) -> bool:
        """Undo one complete pen-down-to-pen-up stroke."""
        self._lift_pen()
        while self.canvas_strokes and not self.canvas_strokes[-1]:
            self.canvas_strokes.pop()
        if not self.canvas_strokes:
            return False
        self.canvas_strokes.pop()
        self.overlay.undo_last_stroke()
        self._rebuild_canvas_from_strokes()
        return True

    def _recognize_and_commit(self, _system_controller) -> None:
        """Recognize one character and append it to the sentence preview."""
        try:
            if self.writing_guide is not None:
                result = self.recognizer.recognize_strokes(
                    self.overlay.strokes, self.writing_guide
                )
            else:
                result = self.recognizer.recognize(self.canvas)
        except Exception as error:
            self.recognition_feedback = f"OCR ERROR: {error}"
            return
        if result is None:
            self.recognition_feedback = "NOT RECOGNIZED — clear and retry"
            return
        if self.sentence.append_character(result.text):
            self.recognition_feedback = (
                f"ADDED: {result.text} ({round(result.confidence * 100)}%)"
            )
            self._clear_writing()
            self._refresh_sentence_preview()
        else:
            self.recognition_feedback = "SENTENCE LIMIT REACHED"

    def _append_space(self) -> None:
        if self.sentence.append_space():
            self.recognition_feedback = "SPACE ADDED"
            self._refresh_sentence_preview()
        else:
            self.recognition_feedback = "SPACE NOT ADDED"

    def _undo_sentence(self) -> None:
        if self.sentence.undo():
            self.recognition_feedback = "LAST CHARACTER REMOVED"
            self._refresh_sentence_preview()
        else:
            self.recognition_feedback = "SENTENCE IS EMPTY"

    def _insert_sentence(self, system_controller) -> None:
        if not self.text_output_allowed:
            self.recognition_feedback = "TEXT OUTPUT STOPPED BY ESC — re-enter Air Write"
            return
        text = self.sentence.text.rstrip()
        if not text:
            self.recognition_feedback = "SENTENCE IS EMPTY"
            return
        if system_controller.commit_text(text):
            self.recognition_feedback = f"INSERTED: {text}"
            self.sentence.clear()
            self._refresh_sentence_preview()
        else:
            self.recognition_feedback = "COULD NOT TYPE — sentence kept for retry"

    def update(
        self,
        observations,
        lock_states,
        now: float,
        frame_width: int,
        frame_height: int,
        system_controller,
        all_hands=None,
    ) -> None:
        self.ensure_canvas(frame_width, frame_height)
        left = observations.get("LEFT")
        right = observations.get("RIGHT")
        left_open = left is not None and is_open_palm(left[1])
        right_open = right is not None and is_open_palm(right[1])
        if all_hands is None:
            all_hands = [item[1] for item in observations.values()]
        self.open_palm_count = sum(is_open_palm(hand) for hand in all_hands)
        both_open = self.open_palm_count >= 2

        if self.mode_hold.update(both_open, now):
            new_mode = "DESKTOP" if self.is_overlay_active else "AIR_WRITE"
            self._set_mode(new_mode, system_controller)

        if not self.is_overlay_active:
            self.pointer_smoother.reset()
            self.pen_pose_metrics = None
            self.overlay.hide_cursor()
            return
        if both_open:
            self._lift_pen()
            self.pointer_smoother.reset()
            self.pen_pose_metrics = None
            self.overlay.hide_cursor()
            self.clear_hold.update(False, now)
            self._update_status(lock_states)
            return

        mapped_right_index = None
        if right is not None and not lock_states["RIGHT"].locked:
            mapped_right_index = self._map_right_index(
                right[1], frame_width, frame_height
            )
            self.overlay.set_cursor(mapped_right_index[1], active=False)
        else:
            self.pointer_smoother.reset()
            self.pen_pose_metrics = None
            self.overlay.hide_cursor()

        clear_requested = left_open and not right_open
        self.canvas_cleared_feedback = False
        if self.clear_hold.update(clear_requested, now):
            self._clear_writing()
            self.recognition_feedback = "CLEARED"
            self.canvas_cleared_feedback = True
        if clear_requested:
            self.accept_hold.update(False, now)
            self.space_hold.update(False, now)
            self.undo_hold.update(False, now)
            self.insert_hold.update(False, now)
            self.mode_switch_hold.update(False, now)
            self._lift_pen()
            self._update_status(lock_states)
            return

        right_thumbs_up = (
            right is not None
            and not lock_states["RIGHT"].locked
            and is_thumbs_up(right[1])
        )
        left_thumbs_up = (
            left is not None
            and not lock_states["LEFT"].locked
            and is_thumbs_up(left[1])
        )
        both_thumbs_up = self.sentence_mode and right_thumbs_up and left_thumbs_up
        mode_switch_pose = (
            self.sentence_mode
            and hasattr(self.recognizer, "cycle_mode")
            and left is not None
            and not lock_states["LEFT"].locked
            and is_mode_switch_pose(left[1])
        )
        accept_pose = right_thumbs_up and (
            not self.sentence_mode or not left_thumbs_up
        )
        space_pose = self.sentence_mode and left_thumbs_up and not right_thumbs_up
        undo_pose = (
            self.sentence_mode
            and left is not None
            and not lock_states["LEFT"].locked
            and not left_thumbs_up
            and not right_thumbs_up
            and not mode_switch_pose
            and is_thumb_index_pinch(left[1])
        )

        mode_switch_fired = self.mode_switch_hold.update(mode_switch_pose, now)
        insert_fired = self.insert_hold.update(both_thumbs_up, now)
        accept_fired = self.accept_hold.update(accept_pose, now)
        space_fired = self.space_hold.update(space_pose, now)
        undo_fired = self.undo_hold.update(undo_pose, now)
        if mode_switch_fired:
            self._cycle_recognition_mode()
        elif insert_fired:
            self._lift_pen()
            self._insert_sentence(system_controller)
        elif accept_fired:
            self._lift_pen()
            self._recognize_and_commit(system_controller)
        elif space_fired:
            self._lift_pen()
            self._append_space()
        elif undo_fired:
            self._lift_pen()
            self._undo_sentence()
        if mode_switch_pose or both_thumbs_up or accept_pose or space_pose or undo_pose:
            self._lift_pen()
            self.overlay.hide_cursor()
            self._update_status(lock_states)
            return

        if right is None or lock_states["RIGHT"].locked:
            self.pen_pose_metrics = None
            self._lift_pen()
            self._update_status(lock_states)
            return

        self.pen_pose_metrics = measure_pen_pose(right[1])
        if not self.pen_pose_metrics.is_drawing:
            self._lift_pen()
            self._update_status(lock_states)
            return

        pen_point, screen_point = mapped_right_index
        self._draw_pen_point(pen_point, screen_point)
        self._update_status(lock_states)

    def draw_desktop_overlay(self, frame) -> None:
        if self.mode_hold.latched:
            text = "RELEASE BOTH PALMS TO RE-ARM"
        elif self.mode_hold.progress > 0.0:
            text = (
                f"OPEN PALMS: {self.open_palm_count}/2 | "
                f"HOLD {round(self.mode_hold.progress * 100)}% -> AIR WRITE"
            )
        else:
            text = f"OPEN PALMS: {self.open_palm_count}/2 | HOLD TWO FOR AIR WRITE"
        cv2.putText(
            frame, text, (20, frame.shape[0] - 92), cv2.FONT_HERSHEY_SIMPLEX,
            0.65, (0, 255, 255), 2, cv2.LINE_AA,
        )

    def pump_overlay(self) -> None:
        self.overlay.pump()

    def emergency_stop(self) -> None:
        """Prevent confirmed text output after the global Esc safety stop."""
        self.text_output_allowed = False
        self.accept_hold.reset()
        self.space_hold.reset()
        self.undo_hold.reset()
        self.insert_hold.reset()
        self.mode_switch_hold.reset()
        self._lift_pen()
        self.overlay.hide_cursor()
        self.recognition_feedback = "SAFE — ESC; re-enter Air Write to re-arm"

    def close(self) -> None:
        self.overlay.close()


def make_line_writing_guide(screen_size: tuple[int, int]):
    """Create a wide single-line region with a visible baseline."""
    screen_width, screen_height = screen_size
    guide_x = round(screen_width * 0.06)
    guide_width = round(screen_width * 0.88)
    guide_height = min(round(screen_height * 0.38), 360)
    guide_y = (screen_height - guide_height) // 2
    baseline_y = guide_y + round(guide_height * 0.78)
    return (
        guide_x,
        guide_y,
        guide_width,
        guide_height,
        baseline_y,
        "MIXED SENTENCE — WRITE LEFT TO RIGHT",
    )


class WholeLineWritingController(AirWritingController):
    """Recognize and insert one complete mixed line after a quiet period."""

    def __init__(
        self,
        overlay,
        recognizer,
        idle_seconds: float = LINE_IDLE_SECONDS,
        gemini_reviewer=None,
    ):
        super().__init__(overlay=overlay, recognizer=recognizer, sentence_mode=False)
        self.recognition_mode = "mixed line"
        self.writing_guide = make_line_writing_guide(self.overlay.size)
        self.idle_seconds = idle_seconds
        self.last_ink_at = None
        self.has_line_ink = False
        self.last_inserted_text = ""
        self.gemini_reviewer = gemini_reviewer
        self._gemini_executor = (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="airdesk-gemini")
            if gemini_reviewer is not None
            else None
        )
        self._gemini_future = None
        self._gemini_revision = 0
        self._gemini_future_revision = None
        self._gemini_local_text = ""
        self.last_gemini_result = None

    def _cancel_gemini_review(self) -> None:
        self._gemini_revision += 1
        if self._gemini_future is not None:
            self._gemini_future.cancel()
        self._gemini_future = None
        self._gemini_future_revision = None
        self._gemini_local_text = ""

    def _clear_writing(self) -> None:
        if hasattr(self, "_gemini_revision"):
            self._cancel_gemini_review()
        super()._clear_writing()
        self.last_ink_at = None
        self.has_line_ink = False

    def _start_gemini_review(self, local_result) -> bool:
        if self.gemini_reviewer is None or self._gemini_executor is None:
            return False
        self._cancel_gemini_review()
        revision = self._gemini_revision
        canvas = self.canvas.copy()
        predictions = tuple(getattr(self.recognizer, "last_predictions", ()))
        self._gemini_local_text = local_result.text
        self._gemini_future_revision = revision
        self._gemini_future = self._gemini_executor.submit(
            self.gemini_reviewer.review,
            canvas,
            local_result.text,
            predictions,
        )
        self.recognition_feedback = (
            f"GEMINI REVIEWING LOW-CONFIDENCE LOCAL: {local_result.text}"
        )
        return True

    def _poll_gemini_review(self, system_controller) -> None:
        future = self._gemini_future
        if future is None or not future.done():
            return
        future_revision = self._gemini_future_revision
        local_text = self._gemini_local_text
        self._gemini_future = None
        self._gemini_future_revision = None
        self._gemini_local_text = ""
        if future_revision != self._gemini_revision:
            return
        try:
            result = future.result()
        except Exception as error:
            self.recognition_feedback = f"GEMINI UNAVAILABLE — LOCAL KEPT: {error}"
            return
        self.last_gemini_result = result
        self.overlay.set_preview(result.text)
        if not self.text_output_allowed:
            self.recognition_feedback = (
                "GEMINI RESULT READY — INSERT BLOCKED BY ESC; line kept"
            )
            return
        if result.confidence < MIN_GEMINI_INSERT_CONFIDENCE:
            self.recognition_feedback = (
                f"GEMINI LOW CONFIDENCE {round(result.confidence * 100)}% — "
                "PREVIEW ONLY"
            )
            return
        if system_controller.commit_text(result.text):
            self.last_inserted_text = result.text
            action = "AGREED AND INSERTED" if result.text == local_text else "CORRECTED AND INSERTED"
            self.recognition_feedback = (
                f"GEMINI {action}: {result.text} "
                f"({round(result.confidence * 100)}%)"
            )
            self._clear_writing()
        else:
            self.recognition_feedback = "GEMINI READY — COULD NOT TYPE; line kept"

    def _recognize_and_auto_insert(self, system_controller) -> None:
        self.last_ink_at = None
        if not self.text_output_allowed:
            self.recognition_feedback = "AUTO-INSERT BLOCKED BY ESC — re-enter Air Write"
            return
        try:
            result = self.recognizer.recognize(self.canvas)
        except Exception as error:
            self.recognition_feedback = f"LINE OCR ERROR: {error}"
            return
        if result is None:
            self.recognition_feedback = "LINE NOT RECOGNIZED — left palm clears"
            return
        self.overlay.set_preview(result.text)
        if result.confidence < MIN_LINE_CONFIDENCE:
            if not self._start_gemini_review(result):
                self.recognition_feedback = (
                    f"LOW CONFIDENCE {round(result.confidence * 100)}% — not inserted"
                )
            return
        if system_controller.commit_text(result.text):
            self.last_inserted_text = result.text
            self.recognition_feedback = (
                f"INSERTED: {result.text} ({round(result.confidence * 100)}%)"
            )
            self._clear_writing()
        else:
            self.recognition_feedback = "COULD NOT TYPE — line kept for retry"

    def _update_line_status(self, lock_states, now: float) -> None:
        if self.pen_down:
            right = "DRAW"
        elif lock_states["RIGHT"].locked:
            right = "LOCKED"
        elif self.pen_pose_metrics is not None:
            pinch = self.pen_pose_metrics.pinch_distance
            openness = round(self.pen_pose_metrics.minimum_other_openness * 100)
            right = f"AIM pinch={pinch:.2f} fingers={openness}%"
        else:
            right = "READY"
        if self.undo_hold.progress > 0 and not self.undo_hold.latched:
            left = f"UNDO {round(self.undo_hold.progress * 100)}%"
        elif lock_states["LEFT"].locked:
            left = "LOCKED"
        else:
            left = "PALM CLEAR | PINCH+3 UP UNDO"
        countdown = ""
        if self.has_line_ink and self.last_ink_at is not None and not self.pen_down:
            remaining = max(self.idle_seconds - (now - self.last_ink_at), 0.0)
            countdown = f" | AUTO-INSERT {remaining:.1f}s"
        feedback = f" | {self.recognition_feedback}" if self.recognition_feedback else ""
        self.overlay.set_status(
            f"MIXED LINE | L: {left} | R: {right} | write left→right"
            f"{countdown}{feedback}"
        )

    def update(
        self,
        observations,
        lock_states,
        now: float,
        frame_width: int,
        frame_height: int,
        system_controller,
        all_hands=None,
    ) -> None:
        self.ensure_canvas(frame_width, frame_height)
        self._poll_gemini_review(system_controller)
        left = observations.get("LEFT")
        right = observations.get("RIGHT")
        left_open = left is not None and is_open_palm(left[1])
        right_open = right is not None and is_open_palm(right[1])
        if all_hands is None:
            all_hands = [item[1] for item in observations.values()]
        self.open_palm_count = sum(is_open_palm(hand) for hand in all_hands)
        both_open = self.open_palm_count >= 2

        if self.mode_hold.update(both_open, now):
            new_mode = "DESKTOP" if self.is_overlay_active else "AIR_WRITE"
            self._set_mode(new_mode, system_controller)
        if not self.is_overlay_active:
            if self._gemini_future is not None:
                self._cancel_gemini_review()
            self.pointer_smoother.reset()
            self.pen_pose_metrics = None
            self.overlay.hide_cursor()
            return
        if both_open:
            self._lift_pen()
            self.pointer_smoother.reset()
            self.pen_pose_metrics = None
            self.overlay.hide_cursor()
            self.clear_hold.update(False, now)
            self._update_line_status(lock_states, now)
            return

        mapped_right_index = None
        if right is not None and not lock_states["RIGHT"].locked:
            mapped_right_index = self._map_right_index(
                right[1], frame_width, frame_height
            )
            self.overlay.set_cursor(mapped_right_index[1], active=False)
        else:
            self.pointer_smoother.reset()
            self.pen_pose_metrics = None
            self.overlay.hide_cursor()

        clear_requested = left_open and not right_open
        if self.clear_hold.update(clear_requested, now):
            self._clear_writing()
            self.overlay.set_preview("")
            self.recognition_feedback = "LINE CLEARED"
        if clear_requested:
            self.undo_hold.update(False, now)
            self._lift_pen()
            self._update_line_status(lock_states, now)
            return

        undo_pose = bool(
            left is not None
            and not lock_states["LEFT"].locked
            and is_stroke_undo_pose(left[1])
        )
        undo_fired = self.undo_hold.update(undo_pose, now)
        if undo_pose:
            self._lift_pen()
            if undo_fired:
                if self._gemini_future is not None:
                    self._cancel_gemini_review()
                if self._undo_last_ink_stroke():
                    self.has_line_ink = bool(self.canvas_strokes)
                    self.last_ink_at = now if self.has_line_ink else None
                    self.recognition_feedback = "LAST STROKE UNDONE"
                else:
                    self.recognition_feedback = "NOTHING TO UNDO"
            self._update_line_status(lock_states, now)
            return

        if right is not None and not lock_states["RIGHT"].locked:
            self.pen_pose_metrics = measure_pen_pose(right[1])
        else:
            self.pen_pose_metrics = None
        drawing = bool(
            self.pen_pose_metrics is not None
            and self.pen_pose_metrics.is_drawing
        )
        if drawing:
            if self._gemini_future is not None:
                self._cancel_gemini_review()
            pen_point, screen_point = mapped_right_index
            self._draw_pen_point(pen_point, screen_point)
            self.has_line_ink = True
            self.last_ink_at = now
            self.recognition_feedback = ""
            self._update_line_status(lock_states, now)
            return

        self._lift_pen()
        if (
            self.has_line_ink
            and self.last_ink_at is not None
            and now - self.last_ink_at >= self.idle_seconds
        ):
            self._recognize_and_auto_insert(system_controller)
        self._update_line_status(lock_states, now)

    def close(self) -> None:
        self._cancel_gemini_review()
        if self._gemini_executor is not None:
            self._gemini_executor.shutdown(wait=False, cancel_futures=True)
        super().close()


# Preserve imports from the first Stage 8 draft.
WhiteboardController = AirWritingController
