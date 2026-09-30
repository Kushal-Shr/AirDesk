"""Stage 7: three-finger right-hand swipes for macOS shortcuts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .desktop_controls import (
    DesktopControlController,
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

SWIPE_DWELL_SECONDS = 0.18
SWIPE_DISTANCE_THRESHOLD = 0.65
SWIPE_DIRECTION_DOMINANCE = 1.20
SWIPE_COOLDOWN_SECONDS = 0.80

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

    def __init__(self, backend, escape_monitor) -> None:
        super().__init__(backend, escape_monitor)
        self.swipe_detector = SwipeGestureDetector()

    def reset_actions(self, require_release: bool = False) -> None:
        super().reset_actions(require_release=require_release)
        if hasattr(self, "swipe_detector"):
            self.swipe_detector.reset(require_release=require_release)

    def update_gestures(self, observations, lock_states, now: float) -> None:
        super().update_gestures(observations, lock_states, now)

        right_observation = observations.get("RIGHT")
        if right_observation is None or lock_states["RIGHT"].locked:
            self.swipe_detector.reset(require_release=True)
            return

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
            self.hotkey(*SHORTCUT_KEYS[action])
            self._show_feedback(SWIPE_LABELS[action], now)


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

