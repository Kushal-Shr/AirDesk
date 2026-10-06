"""Stage 8 gesture state for transparent desktop air-writing."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .air_mouse import map_preview_to_screen
from .character_recognition import EmnistCharacterRecognizer, make_writing_guide
from .desktop_controls import _landmark_points, finger_is_extended
from .hand_lock import FINGER_JOINTS, finger_is_curled, thumb_is_raised
from .native_overlay import MemoryInkOverlay
from .virtual_cursor import ActiveRectangle, map_to_preview


MODE_HOLD_SECONDS = 1.0
CLEAR_HOLD_SECONDS = 1.0
PEN_PINCH_THRESHOLD = 0.38
PEN_THICKNESS = 6
ACCEPT_HOLD_SECONDS = 0.65


def is_open_palm(landmarks) -> bool:
    """Use four extended non-thumb fingers for a robust open-palm pose."""
    points = _landmark_points(landmarks)
    wrist = points[0]
    return all(
        finger_is_extended(points, wrist, *finger_joints)
        for finger_joints in FINGER_JOINTS
    )


def is_pen_pinch(landmarks) -> bool:
    """Use a palm-size-normalized right thumb/index distance for pen down."""
    points = _landmark_points(landmarks)
    palm_size = max(float(np.linalg.norm(points[9] - points[0])), 1e-6)
    distance = float(np.linalg.norm(points[4] - points[8]) / palm_size)
    return distance < PEN_PINCH_THRESHOLD


def is_thumbs_up(landmarks) -> bool:
    """Require a raised thumb while all four other fingers are curled."""
    points = _landmark_points(landmarks)
    wrist = points[0]
    fingers_curled = all(
        finger_is_curled(points, wrist, *finger_joints)
        for finger_joints in FINGER_JOINTS
    )
    return fingers_curled and thumb_is_raised(points)


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


class AirWritingController:
    """Own Desktop/Air Write mode, visible overlay, and an OCR-ready canvas."""

    def __init__(self, overlay=None, recognizer=None) -> None:
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
        self.canvas_cleared_feedback = False
        self.open_palm_count = 0
        self.accept_hold = HoldLatch(ACCEPT_HOLD_SECONDS)
        self.recognition_feedback = ""
        self.text_output_allowed = True
        self.recognition_mode = getattr(self.recognizer, "mode", "uppercase")
        self.writing_guide = None
        if getattr(self.recognizer, "preserves_position", False):
            self.writing_guide = make_writing_guide(
                self.overlay.size, self.recognition_mode.upper()
            )

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
        self.overlay.hide_cursor()
        self.clear_hold.reset()
        self.accept_hold.reset()

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
        pen_point = map_to_preview(
            raw_point,
            active_area,
            frame_width,
            frame_height,
        )
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
            right = f"ACCEPT {round(self.accept_hold.progress * 100)}%"
        feedback = f"  |  {self.recognition_feedback}" if self.recognition_feedback else ""
        self.overlay.set_status(
            f"AIR WRITE [{self.recognition_mode.upper()}] | L: {left} | "
            f"R: {right} | 👍 accept | palms: desktop{feedback}"
        )

    def _clear_writing(self) -> None:
        self.canvas.fill(255)
        self.overlay.clear()
        self._lift_pen()

    def _recognize_and_commit(self, system_controller) -> None:
        if not self.text_output_allowed:
            self.recognition_feedback = "TEXT OUTPUT STOPPED BY ESC — re-enter Air Write"
            return
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
        if system_controller.commit_text(result.text):
            self.recognition_feedback = (
                f"INSERTED: {result.text} ({round(result.confidence * 100)}%)"
            )
            self._clear_writing()
        else:
            self.recognition_feedback = "COULD NOT TYPE — check Accessibility"

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
            self.overlay.hide_cursor()
            return
        if both_open:
            self._lift_pen()
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
            self.overlay.hide_cursor()

        clear_requested = left_open and not right_open
        self.canvas_cleared_feedback = False
        if self.clear_hold.update(clear_requested, now):
            self._clear_writing()
            self.recognition_feedback = "CLEARED"
            self.canvas_cleared_feedback = True
        if clear_requested:
            self.accept_hold.update(False, now)
            self._lift_pen()
            self._update_status(lock_states)
            return

        thumbs_up = (
            right is not None
            and not lock_states["RIGHT"].locked
            and is_thumbs_up(right[1])
        )
        if self.accept_hold.update(thumbs_up, now):
            self._lift_pen()
            self._recognize_and_commit(system_controller)
        if thumbs_up:
            self._lift_pen()
            self.overlay.hide_cursor()
            self._update_status(lock_states)
            return

        if right is None or lock_states["RIGHT"].locked or not is_pen_pinch(right[1]):
            self._lift_pen()
            self._update_status(lock_states)
            return

        pen_point, screen_point = mapped_right_index
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
        self._lift_pen()
        self.overlay.hide_cursor()
        self.recognition_feedback = "SAFE — ESC; re-enter Air Write to re-arm"

    def close(self) -> None:
        self.overlay.close()


# Preserve imports from the first Stage 8 draft.
WhiteboardController = AirWritingController
