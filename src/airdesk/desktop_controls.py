"""Stage 6: debounced right-hand clicks and left-hand scrolling."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import numpy as np

from .air_mouse import SystemMouseController
from .hand_lock import FINGER_JOINTS, finger_is_curled, joint_angle_degrees
from .virtual_cursor import run_virtual_cursor


WINDOW_NAME = "AirDesk - Click and Scroll"

LEFT_CLICK = "LEFT_CLICK"
RIGHT_CLICK = "RIGHT_CLICK"
NO_CLICK = "NONE"

LEFT_CLICK_DWELL_SECONDS = 0.30
RIGHT_CLICK_DWELL_SECONDS = 0.20
CLICK_COOLDOWN_SECONDS = 0.45
CLICK_POSE_GRACE_SECONDS = 0.10
CLICK_RELEASE_SECONDS = 0.10

SCROLL_DWELL_SECONDS = 0.18
SCROLL_COOLDOWN_SECONDS = 0.10
SCROLL_MOVEMENT_THRESHOLD = 0.18
SCROLL_STEPS = 3


def _landmark_points(landmarks) -> np.ndarray:
    return np.array(
        [(landmark.x, landmark.y, landmark.z) for landmark in landmarks],
        dtype=np.float32,
    )


def finger_is_extended(
    points: np.ndarray,
    wrist: np.ndarray,
    mcp_id: int,
    pip_id: int,
    tip_id: int,
) -> bool:
    """Return whether one non-thumb finger is deliberately extended."""
    angle = joint_angle_degrees(points[mcp_id], points[pip_id], points[tip_id])
    tip_distance = np.linalg.norm(points[tip_id] - wrist)
    pip_distance = np.linalg.norm(points[pip_id] - wrist)
    return bool(angle > 150.0 and tip_distance > pip_distance * 1.10)


@dataclass(frozen=True)
class ClickMetrics:
    pose: str
    index_extended: bool
    middle_extended: bool
    ring_curled: bool
    little_curled: bool
    thumb_folded: bool


def measure_click_pose(landmarks) -> ClickMetrics:
    """Classify deliberate one- and two-finger right-hand click poses."""
    points = _landmark_points(landmarks)
    wrist = points[0]
    palm_size = max(float(np.linalg.norm(points[9] - wrist)), 1e-6)

    index_extended = finger_is_extended(points, wrist, *FINGER_JOINTS[0])
    middle_extended = finger_is_extended(points, wrist, *FINGER_JOINTS[1])
    middle_curled = finger_is_curled(points, wrist, *FINGER_JOINTS[1])
    ring_curled = finger_is_curled(points, wrist, *FINGER_JOINTS[2])
    little_curled = finger_is_curled(points, wrist, *FINGER_JOINTS[3])
    palm_center = points[[0, 5, 9, 13, 17]].mean(axis=0)
    thumb_compactness = float(np.linalg.norm(points[4] - palm_center) / palm_size)
    thumb_folded = thumb_compactness < 1.25

    if (
        index_extended
        and middle_extended
        and ring_curled
        and little_curled
        and thumb_folded
    ):
        pose = RIGHT_CLICK
    elif (
        index_extended
        and middle_curled
        and ring_curled
        and little_curled
        and thumb_folded
    ):
        pose = LEFT_CLICK
    else:
        pose = NO_CLICK
    return ClickMetrics(
        pose,
        index_extended,
        middle_extended,
        ring_curled,
        little_curled,
        thumb_folded,
    )


def classify_click_pose(landmarks) -> str:
    return measure_click_pose(landmarks).pose


@dataclass
class ClickGestureDetector:
    """Require dwell, one-shot latching, release, and a click cooldown."""

    candidate: str = NO_CLICK
    candidate_started_at: float | None = None
    latched: bool = False
    blocked_until_release: bool = False
    cooldown_until: float = 0.0
    last_candidate_seen_at: float | None = None
    release_started_at: float | None = None

    def reset(self, require_release: bool = False) -> None:
        self.candidate = NO_CLICK
        self.candidate_started_at = None
        self.latched = False
        self.blocked_until_release = require_release
        self.last_candidate_seen_at = None
        self.release_started_at = None

    def update(self, pose: str, now: float) -> str | None:
        if self.blocked_until_release:
            if pose == NO_CLICK:
                if self.release_started_at is None:
                    self.release_started_at = now
                elif now - self.release_started_at >= CLICK_RELEASE_SECONDS:
                    self.blocked_until_release = False
                    self.release_started_at = None
            else:
                self.release_started_at = None
            return None

        if self.latched:
            if pose == NO_CLICK:
                if self.release_started_at is None:
                    self.release_started_at = now
                elif now - self.release_started_at >= CLICK_RELEASE_SECONDS:
                    self.latched = False
                    self.candidate = NO_CLICK
                    self.candidate_started_at = None
                    self.last_candidate_seen_at = None
                    self.release_started_at = None
            else:
                self.release_started_at = None
            return None

        if pose == NO_CLICK:
            if (
                self.candidate != NO_CLICK
                and self.last_candidate_seen_at is not None
                and now - self.last_candidate_seen_at <= CLICK_POSE_GRACE_SECONDS
            ):
                return None
            self.candidate = NO_CLICK
            self.candidate_started_at = None
            self.last_candidate_seen_at = None
            return None

        if pose != self.candidate:
            self.candidate = pose
            self.candidate_started_at = now
            self.last_candidate_seen_at = now
            return None

        self.last_candidate_seen_at = now

        dwell = (
            RIGHT_CLICK_DWELL_SECONDS
            if pose == RIGHT_CLICK
            else LEFT_CLICK_DWELL_SECONDS
        )
        if self.candidate_started_at is None or now - self.candidate_started_at < dwell:
            return None

        # Latch even during cooldown so an old held pinch cannot fire later.
        self.latched = True
        if now < self.cooldown_until:
            return None
        self.cooldown_until = now + CLICK_COOLDOWN_SECONDS
        return pose


@dataclass(frozen=True)
class ScrollMetrics:
    is_scroll_pose: bool
    hand_y: float
    palm_size: float


def classify_scroll_pose(landmarks) -> ScrollMetrics:
    """Require extended index/middle fingers and curled ring/little fingers."""
    points = _landmark_points(landmarks)
    wrist = points[0]
    palm_size = max(float(np.linalg.norm(points[9] - wrist)), 1e-6)

    extended = []
    for mcp_id, pip_id, tip_id in FINGER_JOINTS[:2]:
        extended.append(
            finger_is_extended(points, wrist, mcp_id, pip_id, tip_id)
        )

    curled = [
        finger_is_curled(points, wrist, mcp_id, pip_id, tip_id)
        for mcp_id, pip_id, tip_id in FINGER_JOINTS[2:]
    ]
    hand_y = float((points[8, 1] + points[12, 1]) / 2.0)
    return ScrollMetrics(all(extended) and all(curled), hand_y, palm_size)


@dataclass
class ScrollGestureDetector:
    """Turn deliberate vertical two-finger motion into discrete scroll steps."""

    pose_started_at: float | None = None
    anchor_y: float | None = None
    active: bool = False
    blocked_until_release: bool = False
    cooldown_until: float = 0.0

    def reset(self, require_release: bool = False) -> None:
        self.pose_started_at = None
        self.anchor_y = None
        self.active = False
        self.blocked_until_release = require_release

    def update(self, metrics: ScrollMetrics, now: float) -> int:
        if self.blocked_until_release:
            if not metrics.is_scroll_pose:
                self.blocked_until_release = False
            return 0

        if not metrics.is_scroll_pose:
            self.reset()
            return 0

        if self.pose_started_at is None:
            self.pose_started_at = now
            self.anchor_y = metrics.hand_y
            return 0

        if not self.active:
            if now - self.pose_started_at < SCROLL_DWELL_SECONDS:
                return 0
            self.active = True
            self.anchor_y = metrics.hand_y
            return 0

        if self.anchor_y is None or now < self.cooldown_until:
            return 0

        normalized_movement = (self.anchor_y - metrics.hand_y) / metrics.palm_size
        if abs(normalized_movement) < SCROLL_MOVEMENT_THRESHOLD:
            return 0

        self.anchor_y = metrics.hand_y
        self.cooldown_until = now + SCROLL_COOLDOWN_SECONDS
        return SCROLL_STEPS if normalized_movement > 0 else -SCROLL_STEPS


class MacEscapeMonitor:
    """Listen only for global Escape key-down events using macOS Quartz."""

    ESCAPE_KEY_CODE = 53

    def __init__(self) -> None:
        self._escape_event = threading.Event()
        self._ready_event = threading.Event()
        self._thread = None
        self._run_loop = None
        self._error = None
        self._quartz = None
        self._callback = None

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._run,
            name="AirDeskEscapeMonitor",
            daemon=True,
        )
        self._thread.start()
        if not self._ready_event.wait(timeout=3.0):
            raise RuntimeError("timed out while starting the global Escape monitor")
        if self._error is not None:
            raise RuntimeError(str(self._error))

    def _run(self) -> None:
        try:
            import Quartz

            self._quartz = Quartz

            def callback(_proxy, event_type, event, _context):
                if event_type == Quartz.kCGEventKeyDown:
                    key_code = Quartz.CGEventGetIntegerValueField(
                        event,
                        Quartz.kCGKeyboardEventKeycode,
                    )
                    if key_code == self.ESCAPE_KEY_CODE:
                        self._escape_event.set()
                return event

            self._callback = callback
            event_mask = Quartz.CGEventMaskBit(Quartz.kCGEventKeyDown)
            event_tap = Quartz.CGEventTapCreate(
                Quartz.kCGSessionEventTap,
                Quartz.kCGHeadInsertEventTap,
                Quartz.kCGEventTapOptionListenOnly,
                event_mask,
                callback,
                None,
            )
            if event_tap is None:
                raise RuntimeError(
                    "global Escape monitoring was denied; enable Accessibility "
                    "permission for the app running AirDesk"
                )

            source = Quartz.CFMachPortCreateRunLoopSource(None, event_tap, 0)
            self._run_loop = Quartz.CFRunLoopGetCurrent()
            Quartz.CFRunLoopAddSource(
                self._run_loop,
                source,
                Quartz.kCFRunLoopCommonModes,
            )
            Quartz.CGEventTapEnable(event_tap, True)
            self._ready_event.set()
            Quartz.CFRunLoopRun()
        except Exception as error:
            self._error = error
            self._ready_event.set()

    def consume_escape(self) -> bool:
        if not self._escape_event.is_set():
            return False
        self._escape_event.clear()
        return True

    def close(self) -> None:
        if self._quartz is not None and self._run_loop is not None:
            self._quartz.CFRunLoopStop(self._run_loop)
        if self._thread is not None:
            self._thread.join(timeout=1.0)


class DesktopControlController(SystemMouseController):
    """Combine safe PyAutoGUI output with Stage 6 gesture state machines."""

    def __init__(self, backend, escape_monitor) -> None:
        super().__init__(backend)
        self.escape_monitor = escape_monitor
        self.click_detector = ClickGestureDetector()
        self.scroll_detector = ScrollGestureDetector()
        self.left_status = None
        self.right_status = None
        self._feedback_text = ""
        self._feedback_until = 0.0
        self.diagnostic_text = "RIGHT CLICK: waiting for hand"

    @property
    def feedback_text(self) -> str:
        if time.perf_counter() >= self._feedback_until:
            return ""
        return self._feedback_text

    def _show_feedback(self, text: str, now: float) -> None:
        mode = "ACTIVE" if self.enabled else "PREVIEW"
        self._feedback_text = f"{mode}: {text}"
        self._feedback_until = now + 0.8
        print(self._feedback_text)

    def reset_actions(self, require_release: bool = False) -> None:
        if hasattr(self, "click_detector"):
            self.click_detector.reset(require_release=require_release)
            self.scroll_detector.reset(require_release=require_release)
            self.left_status = None
            self.right_status = None

    def update_gestures(self, observations, lock_states, now: float) -> None:
        if self.escape_monitor.consume_escape():
            self.disable("global Esc pressed")
            self._show_feedback("SAFE - ESC", now)

        right_observation = observations.get("RIGHT")
        if right_observation is None or lock_states["RIGHT"].locked:
            self.click_detector.reset(require_release=True)
            self.right_status = None
            if right_observation is None:
                self.diagnostic_text = "RIGHT CLICK: hand not seen"
            else:
                locked_metrics = measure_click_pose(right_observation[1])
                self.diagnostic_text = (
                    "RIGHT CLICK: LOCKED - open first | "
                    f"index={'UP' if locked_metrics.index_extended else 'DOWN'} "
                    f"middle={'UP' if locked_metrics.middle_extended else 'DOWN'}"
                )
        else:
            click_metrics = measure_click_pose(right_observation[1])
            click_pose = click_metrics.pose
            self.right_status = "CLICK" if click_pose != NO_CLICK else None
            pose_name = click_pose.replace("_", " ")
            self.diagnostic_text = (
                "RIGHT POSE "
                f"index={'UP' if click_metrics.index_extended else 'DOWN'} "
                f"middle={'UP' if click_metrics.middle_extended else 'DOWN'} "
                f"ring={'DOWN' if click_metrics.ring_curled else 'UP'} "
                f"little={'DOWN' if click_metrics.little_curled else 'UP'} "
                f"thumb={'DOWN' if click_metrics.thumb_folded else 'UP'} "
                f"| {pose_name}"
            )
            click_action = self.click_detector.update(click_pose, now)
            if click_action == LEFT_CLICK:
                self.click("left")
                self._show_feedback("LEFT CLICK", now)
            elif click_action == RIGHT_CLICK:
                self.click("right")
                self._show_feedback("RIGHT CLICK", now)

        left_observation = observations.get("LEFT")
        if left_observation is None or lock_states["LEFT"].locked:
            self.scroll_detector.reset(require_release=True)
            self.left_status = None
        else:
            scroll_metrics = classify_scroll_pose(left_observation[1])
            self.left_status = "SCROLL" if scroll_metrics.is_scroll_pose else None
            steps = self.scroll_detector.update(scroll_metrics, now)
            if steps:
                self.scroll(steps)
                direction = "UP" if steps > 0 else "DOWN"
                self._show_feedback(f"SCROLL {direction}", now)

    def close(self) -> None:
        super().close()
        self.escape_monitor.close()


def main() -> int:
    escape_monitor = MacEscapeMonitor()
    try:
        import pyautogui
        import Quartz

        if not Quartz.CGPreflightPostEventAccess():
            raise RuntimeError(
                "macOS is blocking synthetic click events for this launcher; "
                "enable its Accessibility permission and restart it"
            )

        escape_monitor.start()
        controller = DesktopControlController(pyautogui, escape_monitor)
    except Exception as error:
        escape_monitor.close()
        print(f"AirDesk could not start safe desktop controls: {error}")
        print(
            "Enable Camera and Accessibility permission for the app running "
            "AirDesk, restart it, and try again."
        )
        return 1

    print("Real mouse, click, and scroll output starts OFF.")
    print("M toggles real output. Esc disables it even if another app gains focus.")
    return run_virtual_cursor(system_mouse=controller, window_name=WINDOW_NAME)


if __name__ == "__main__":
    raise SystemExit(main())
