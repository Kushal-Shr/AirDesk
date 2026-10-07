"""Stage 7: three-finger right-hand swipes for macOS shortcuts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .desktop_controls import (
    DesktopControlController,
    DwellLatch,
    MacEscapeMonitor,
    _landmark_points,
    finger_is_extended,
)
from .hand_lock import FINGER_JOINTS, finger_is_curled
from .shortcut_config import (
    MISSION_CONTROL_KEYS,
    NEXT_APP_KEYS,
    PREVIOUS_APP_KEYS,
)
from .virtual_cursor import run_virtual_cursor


WINDOW_NAME = "AirDesk - Three-Finger Shortcuts"

SWIPE_UP = "SWIPE_UP"
SWIPE_RIGHT = "SWIPE_RIGHT"
SWIPE_LEFT = "SWIPE_LEFT"

SWIPE_DWELL_SECONDS = 0.08
SWIPE_DISTANCE_THRESHOLD = 0.48
SWIPE_DIRECTION_DOMINANCE = 1.20
SWIPE_COOLDOWN_SECONDS = 0.45
ENTER_HOLD_SECONDS = 0.30

SHORTCUT_KEYS = {
    SWIPE_UP: MISSION_CONTROL_KEYS,
    SWIPE_RIGHT: NEXT_APP_KEYS,
    SWIPE_LEFT: PREVIOUS_APP_KEYS,
}

SWIPE_LABELS = {
    SWIPE_UP: "MISSION CONTROL",
    SWIPE_RIGHT: "NEXT APP",
    SWIPE_LEFT: "PREVIOUS APP",
}


@dataclass(frozen=True)
class SwipeMetrics:
    is_swipe_pose: bool
    palm_x: float
    palm_y: float
    palm_size: float


def classify_swipe_pose(landmarks) -> SwipeMetrics:
    """Require index, middle, ring up with thumb and little finger down."""
    points = _landmark_points(landmarks)
    wrist = points[0]
    palm_size = max(float(np.linalg.norm(points[9] - wrist)), 1e-6)

    first_three_extended = all(
        finger_is_extended(points, wrist, *finger_joints)
        for finger_joints in FINGER_JOINTS[:3]
    )
    little_curled = finger_is_curled(points, wrist, *FINGER_JOINTS[3])

    palm_ids = [0, 5, 9, 13, 17]
    palm_center = points[palm_ids].mean(axis=0)
    thumb_compactness = float(np.linalg.norm(points[4] - palm_center) / palm_size)
    thumb_folded = thumb_compactness < 1.25

    return SwipeMetrics(
        is_swipe_pose=first_three_extended and little_curled and thumb_folded,
        palm_x=float(palm_center[0]),
        palm_y=float(palm_center[1]),
        palm_size=palm_size,
    )


def classify_enter_pose(landmarks) -> bool:
    """Require only the right index finger up, with thumb folded."""
    points = _landmark_points(landmarks)
    wrist = points[0]
    palm_size = max(float(np.linalg.norm(points[9] - wrist)), 1e-6)
    extended = tuple(
        finger_is_extended(points, wrist, *finger_joints)
        for finger_joints in FINGER_JOINTS
    )
    other_fingers_curled = all(
        finger_is_curled(points, wrist, *finger_joints)
        for finger_joints in FINGER_JOINTS[1:]
    )
    palm_center = points[[0, 5, 9, 13, 17]].mean(axis=0)
    thumb_folded = float(np.linalg.norm(points[4] - palm_center) / palm_size) < 1.25
    return bool(
        extended[0]
        and extended[1:] == (False, False, False)
        and other_fingers_curled
        and thumb_folded
    )


@dataclass
class SwipeGestureDetector:
    """Recognize one dominant-direction swipe, then require pose release."""

    pose_started_at: float | None = None
    start_x: float | None = None
    start_y: float | None = None
    active: bool = False
    latched: bool = False
    blocked_until_release: bool = False
    cooldown_until: float = 0.0
    last_horizontal: float = 0.0
    last_upward: float = 0.0

    def reset(self, require_release: bool = False) -> None:
        self.pose_started_at = None
        self.start_x = None
        self.start_y = None
        self.active = False
        self.latched = False
        self.blocked_until_release = require_release
        self.last_horizontal = 0.0
        self.last_upward = 0.0

    def update(self, metrics: SwipeMetrics, now: float) -> str | None:
        if self.blocked_until_release:
            if not metrics.is_swipe_pose:
                self.blocked_until_release = False
            return None

        if self.latched:
            if not metrics.is_swipe_pose:
                self.reset()
            return None

        if not metrics.is_swipe_pose:
            self.reset()
            return None

        if self.pose_started_at is None:
            self.pose_started_at = now
            self.start_x = metrics.palm_x
            self.start_y = metrics.palm_y
            return None

        if not self.active:
            if now - self.pose_started_at < SWIPE_DWELL_SECONDS:
                return None
            self.active = True
            self.start_x = metrics.palm_x
            self.start_y = metrics.palm_y
            return None

        if self.start_x is None or self.start_y is None:
            return None

        horizontal = (metrics.palm_x - self.start_x) / metrics.palm_size
        upward = (self.start_y - metrics.palm_y) / metrics.palm_size
        self.last_horizontal = horizontal
        self.last_upward = upward

        action = None
        if (
            upward > SWIPE_DISTANCE_THRESHOLD
            and upward > abs(horizontal) * SWIPE_DIRECTION_DOMINANCE
        ):
            action = SWIPE_UP
        elif (
            abs(horizontal) > SWIPE_DISTANCE_THRESHOLD
            and abs(horizontal) > abs(upward) * SWIPE_DIRECTION_DOMINANCE
        ):
            action = SWIPE_RIGHT if horizontal > 0 else SWIPE_LEFT

        if action is None:
            return None

        # Consume the held swipe even during cooldown so it cannot fire later.
        self.latched = True
        if now < self.cooldown_until:
            return None
        self.cooldown_until = now + SWIPE_COOLDOWN_SECONDS
        return action


class ShortcutControlController(DesktopControlController):
    """Add Stage 7 swipes to the Stage 6 click and scroll controller."""

    def __init__(
        self,
        backend,
        escape_monitor,
        feature_config=None,
        command_palette=None,
        document_safety=None,
    ) -> None:
        super().__init__(
            backend,
            escape_monitor,
            feature_config=feature_config,
            command_palette=command_palette,
            document_safety=document_safety,
        )
        self.swipe_detector = SwipeGestureDetector()
        self.enter_detector = DwellLatch(ENTER_HOLD_SECONDS)

    def reset_actions(self, require_release: bool = False) -> None:
        super().reset_actions(require_release=require_release)
        if hasattr(self, "swipe_detector"):
            self.swipe_detector.reset(require_release=require_release)
        if hasattr(self, "enter_detector"):
            self.enter_detector.reset(require_release=require_release)

    def _show_mission_control(self) -> bool:
        """Prefer the native app so remapped/disabled shortcuts still work."""
        if not self.enabled:
            return False
        native_action = getattr(self.backend, "showMissionControl", None)
        if native_action is not None:
            return self._handle_output_error(native_action)
        return self.hotkey(*MISSION_CONTROL_KEYS)

    def update_gestures(
        self,
        observations,
        lock_states,
        now: float,
        cursor_position=None,
        preview_size=None,
    ) -> None:
        super().update_gestures(
            observations,
            lock_states,
            now,
            cursor_position=cursor_position,
            preview_size=preview_size,
        )

        if self.actions_guarded(now):
            self.swipe_detector.reset(require_release=True)
            self.enter_detector.reset(require_release=True)
            return

        right_observation = observations.get("RIGHT")
        if right_observation is None or lock_states["RIGHT"].locked:
            self.swipe_detector.reset(require_release=True)
            self.enter_detector.reset(require_release=True)
            return

        if self.right_status == "COMMAND":
            self.swipe_detector.reset(require_release=True)
            self.enter_detector.reset(require_release=True)
            return

        if classify_enter_pose(right_observation[1]):
            self.swipe_detector.reset(require_release=True)
            self.right_status = "ENTER"
            self.diagnostic_text = "RIGHT ENTER: index only"
            if self.enter_detector.update(True, now):
                self.hotkey("enter")
                self._show_feedback("ENTER", now)
            return

        self.enter_detector.update(False, now)

        metrics = classify_swipe_pose(right_observation[1])
        if not metrics.is_swipe_pose:
            self.swipe_detector.update(metrics, now)
            return

        self.right_status = "SWIPE"
        action = self.swipe_detector.update(metrics, now)
        self.diagnostic_text = (
            "RIGHT SWIPE: 3 FINGERS | "
            f"dx={self.swipe_detector.last_horizontal:+.2f} "
            f"up={self.swipe_detector.last_upward:+.2f}"
        )
        if action is not None:
            feature_key = (
                "mission_control" if action == SWIPE_UP else "app_switching"
            )
            if self._feature_enabled(feature_key):
                if action == SWIPE_UP:
                    self._show_mission_control()
                else:
                    self.hotkey(*SHORTCUT_KEYS[action])
                self._show_feedback(SWIPE_LABELS[action], now)
            else:
                self._show_feedback(f"{SWIPE_LABELS[action]} IS OFF", now)


def main() -> int:
    escape_monitor = MacEscapeMonitor()
    try:
        import pyautogui
        import Quartz

        if not Quartz.CGPreflightPostEventAccess():
            raise RuntimeError(
                "macOS is blocking synthetic shortcuts for this launcher; "
                "enable its Accessibility permission and restart it"
            )

        escape_monitor.start()
        controller = ShortcutControlController(pyautogui, escape_monitor)
    except Exception as error:
        escape_monitor.close()
        print(f"AirDesk could not start safe system shortcuts: {error}")
        print(
            "Enable Camera and Accessibility permission for the app running "
            "AirDesk, restart it, and try again."
        )
        return 1

    print("Real mouse, click, scroll, and shortcut output starts OFF.")
    print("M toggles real output. Esc disables it even if another app gains focus.")
    return run_virtual_cursor(system_mouse=controller, window_name=WINDOW_NAME)


if __name__ == "__main__":
    raise SystemExit(main())
