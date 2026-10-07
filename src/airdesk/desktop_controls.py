"""Stage 6: debounced right-hand clicks and left-hand scrolling."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import numpy as np

from .air_mouse import SystemMouseController
from .air_mouse import map_preview_to_screen
from .command_palette import PALETTE_COMMANDS
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

SCROLL_DWELL_SECONDS = 0.08
SCROLL_COOLDOWN_SECONDS = 0.06
SCROLL_MOVEMENT_THRESHOLD = 0.12
SCROLL_STEPS = 3

PRIMARY_PINCH = "PRIMARY_PINCH"
SECONDARY_PINCH = "SECONDARY_PINCH"
NO_PINCH = "NONE"
SINGLE_CLICK_ACTION = "SINGLE_CLICK"
DOUBLE_CLICK_ACTION = "DOUBLE_CLICK"
DRAG_START_ACTION = "DRAG_START"
DRAG_END_ACTION = "DRAG_END"

PRIMARY_PINCH_THRESHOLD = 0.34
SECONDARY_PINCH_THRESHOLD = 0.20
PRIMARY_MIDDLE_CLEARANCE = 0.24
SECONDARY_PINCH_DWELL_SECONDS = 0.08
DRAG_HOLD_SECONDS = 0.32
QUICK_PINCH_MAX_SECONDS = 0.30
DOUBLE_PINCH_INTERVAL_SECONDS = 0.38
PALETTE_HOLD_SECONDS = 0.30
PALETTE_SELECT_DWELL_SECONDS = 0.05


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


@dataclass(frozen=True)
class DesktopPinchMetrics:
    """Normalized right-hand measurements used for click and palette gestures."""

    pose: str
    thumb_index_distance: float
    thumb_middle_distance: float
    four_fingers_extended: bool
    palette_pose: bool


def measure_desktop_pinch(landmarks) -> DesktopPinchMetrics:
    points = _landmark_points(landmarks)
    wrist = points[0]
    palm_size = max(float(np.linalg.norm(points[9] - wrist)), 1e-6)
    thumb_index = float(np.linalg.norm(points[4] - points[8]) / palm_size)
    thumb_middle = float(np.linalg.norm(points[4] - points[12]) / palm_size)
    four_extended = all(
        finger_is_extended(points, wrist, *finger_joints)
        for finger_joints in FINGER_JOINTS
    )
    index_extended = finger_is_extended(points, wrist, *FINGER_JOINTS[0])
    middle_extended = finger_is_extended(points, wrist, *FINGER_JOINTS[1])
    ring_curled = finger_is_curled(points, wrist, *FINGER_JOINTS[2])
    little_curled = finger_is_curled(points, wrist, *FINGER_JOINTS[3])
    palm_center = points[[0, 5, 9, 13, 17]].mean(axis=0)
    thumb_compactness = float(np.linalg.norm(points[4] - palm_center) / palm_size)
    thumb_folded = thumb_compactness < 1.25
    palette_pose = (
        index_extended
        and middle_extended
        and ring_curled
        and little_curled
        and thumb_folded
    )

    if (
        thumb_index < SECONDARY_PINCH_THRESHOLD
        and thumb_middle < SECONDARY_PINCH_THRESHOLD
    ):
        pose = SECONDARY_PINCH
    elif (
        thumb_index < PRIMARY_PINCH_THRESHOLD
        and thumb_middle > PRIMARY_MIDDLE_CLEARANCE
    ):
        pose = PRIMARY_PINCH
    else:
        pose = NO_PINCH
    return DesktopPinchMetrics(
        pose,
        thumb_index,
        thumb_middle,
        four_extended,
        palette_pose,
    )


@dataclass
class PrimaryPinchDetector:
    """Distinguish one pinch, two pinches, and a held drag without ambiguity."""

    pinching: bool = False
    pinch_started_at: float | None = None
    dragging: bool = False
    first_click_at: float | None = None
    blocked_until_release: bool = False

    def reset(self, require_release: bool = False) -> None:
        self.pinching = False
        self.pinch_started_at = None
        self.dragging = False
        self.first_click_at = None
        self.blocked_until_release = require_release

    def update(
        self,
        is_pinching: bool,
        now: float,
        *,
        allow_double: bool,
        allow_drag: bool,
    ) -> str | None:
        if self.blocked_until_release:
            if not is_pinching:
                self.blocked_until_release = False
            return None

        if is_pinching:
            if not self.pinching:
                self.pinching = True
                self.pinch_started_at = now
                if (
                    self.first_click_at is not None
                    and now - self.first_click_at > DOUBLE_PINCH_INTERVAL_SECONDS
                ):
                    self.first_click_at = None
                return None
            if (
                allow_drag
                and not self.dragging
                and self.pinch_started_at is not None
                and now - self.pinch_started_at >= DRAG_HOLD_SECONDS
            ):
                self.dragging = True
                self.first_click_at = None
                return DRAG_START_ACTION
            return None

        if self.pinching:
            duration = now - (
                self.pinch_started_at if self.pinch_started_at is not None else now
            )
            self.pinching = False
            self.pinch_started_at = None
            if self.dragging:
                self.dragging = False
                return DRAG_END_ACTION
            if duration <= QUICK_PINCH_MAX_SECONDS:
                if not allow_double:
                    return SINGLE_CLICK_ACTION
                if (
                    self.first_click_at is not None
                    and now - self.first_click_at <= DOUBLE_PINCH_INTERVAL_SECONDS
                ):
                    self.first_click_at = None
                    return DOUBLE_CLICK_ACTION
                self.first_click_at = now
                return SINGLE_CLICK_ACTION

        if (
            self.first_click_at is not None
            and now - self.first_click_at > DOUBLE_PINCH_INTERVAL_SECONDS
        ):
            self.first_click_at = None
        return None


@dataclass
class DwellLatch:
    """Fire once after a pose dwell and require a release before rearming."""

    dwell_seconds: float
    started_at: float | None = None
    latched: bool = False
    blocked_until_release: bool = False

    def reset(self, require_release: bool = False) -> None:
        self.started_at = None
        self.latched = False
        self.blocked_until_release = require_release

    def update(self, active: bool, now: float) -> bool:
        if self.blocked_until_release:
            if not active:
                self.blocked_until_release = False
            return False
        if not active:
            self.started_at = None
            self.latched = False
            return False
        if self.latched:
            return False
        if self.started_at is None:
            self.started_at = now
            return False
        if now - self.started_at < self.dwell_seconds:
            return False
        self.latched = True
        return True


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
        event_tap = None
        source = None
        try:
            import Quartz

            self._quartz = Quartz

            def callback(_proxy, event_type, event, _context):
                if event_type in (
                    Quartz.kCGEventTapDisabledByTimeout,
                    Quartz.kCGEventTapDisabledByUserInput,
                ):
                    Quartz.CGEventTapEnable(event_tap, True)
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
        finally:
            if event_tap is not None:
                Quartz.CGEventTapEnable(event_tap, False)
                if source is not None:
                    Quartz.CFRunLoopRemoveSource(
                        self._run_loop, source, Quartz.kCFRunLoopCommonModes
                    )
                Quartz.CFMachPortInvalidate(event_tap)

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
    """Combine safe output, pinch gestures, scrolling, and command selection."""

    def __init__(
        self,
        backend,
        escape_monitor,
        feature_config=None,
        command_palette=None,
        document_safety=None,
    ) -> None:
        super().__init__(backend, document_safety=document_safety)
        self.escape_monitor = escape_monitor
        self.feature_config = feature_config
        self.command_palette = command_palette
        self.primary_pinch_detector = PrimaryPinchDetector()
        self.secondary_pinch_detector = DwellLatch(SECONDARY_PINCH_DWELL_SECONDS)
        self.palette_open_detector = DwellLatch(PALETTE_HOLD_SECONDS)
        self.palette_select_detector = DwellLatch(PALETTE_SELECT_DWELL_SECONDS)
        self.scroll_detector = ScrollGestureDetector()
        self.left_status = None
        self.right_status = None
        self._feedback_text = ""
        self._feedback_until = 0.0
        self._gesture_guard_until = 0.0
        self.diagnostic_text = "RIGHT CLICK: waiting for hand"

    @property
    def feedback_text(self) -> str:
        if time.perf_counter() >= self._feedback_until:
            return ""
        return self._feedback_text

    def _show_feedback(self, text: str, now: float) -> None:
        mode = "LIVE" if self.enabled else "STOPPED"
        self._feedback_text = f"{mode}: {text}"
        self._feedback_until = now + 0.8
        print(self._feedback_text)

    def guard_actions_until(self, deadline: float) -> None:
        """Keep LIVE output armed while suppressing transitional gestures."""
        self._gesture_guard_until = max(self._gesture_guard_until, deadline)
        self.reset_actions(require_release=True)

    def actions_guarded(self, now: float) -> bool:
        return now < self._gesture_guard_until

    def _feature_enabled(self, key: str) -> bool:
        if self.feature_config is None:
            return True
        return self.feature_config.get(key)

    def _enabled_palette_keys(self) -> set[str]:
        return {
            command.key
            for command in PALETTE_COMMANDS
            if self._feature_enabled(command.key)
        }

    def _close_palette(self) -> None:
        if self.command_palette is not None:
            self.command_palette.hide()
        self.palette_select_detector.reset(require_release=True)

    def _release_drag(self) -> None:
        self.mouse_up("left")
        self.primary_pinch_detector.dragging = False

    def reset_actions(self, require_release: bool = False) -> None:
        if hasattr(self, "primary_pinch_detector"):
            self._release_drag()
            self.primary_pinch_detector.reset(require_release=require_release)
            self.secondary_pinch_detector.reset(require_release=require_release)
            self.palette_open_detector.reset(require_release=require_release)
            self.palette_select_detector.reset(require_release=require_release)
            self.scroll_detector.reset(require_release=require_release)
            self._close_palette()
            self.left_status = None
            self.right_status = None

    def _update_palette(
        self,
        metrics: DesktopPinchMetrics,
        now: float,
        cursor_position: tuple[int, int] | None,
        preview_size: tuple[int, int] | None,
    ) -> bool:
        if self.command_palette is None:
            return False

        if self.command_palette.visible:
            if not self._feature_enabled("air_command_palette"):
                self._close_palette()
                return False
            enabled_keys = self._enabled_palette_keys()
            self.command_palette.update_enabled(enabled_keys)
            screen_point = None
            if cursor_position is not None and preview_size is not None:
                screen_point = map_preview_to_screen(
                    cursor_position,
                    preview_size,
                    self.screen_size,
                )
            highlighted = self.command_palette.hit_test(screen_point)
            self.command_palette.highlight(highlighted)
            selected = self.palette_select_detector.update(
                metrics.pose == PRIMARY_PINCH,
                now,
            )
            self.right_status = "COMMAND"
            if selected and highlighted is not None:
                command = PALETTE_COMMANDS[highlighted]
                if command.key not in enabled_keys:
                    self._show_feedback(f"{command.label.upper()} IS OFF", now)
                else:
                    if command.key == "save_document":
                        succeeded = self.save_document()
                    else:
                        succeeded = self.hotkey(*command.shortcut)
                    if succeeded:
                        self._show_feedback(command.label.upper(), now)
                    self._close_palette()
                    self.primary_pinch_detector.reset(require_release=True)
            return True

        palette_pose = metrics.palette_pose and metrics.pose == NO_PINCH
        if palette_pose and (
            self.primary_pinch_detector.pinching
            or self.primary_pinch_detector.first_click_at is not None
        ):
            self.palette_open_detector.reset()
            return False
        if not self._feature_enabled("air_command_palette"):
            self.palette_open_detector.reset()
            return False
        if self.palette_open_detector.update(palette_pose, now):
            self.primary_pinch_detector.reset(require_release=False)
            self.secondary_pinch_detector.reset(require_release=False)
            self.palette_select_detector.reset(require_release=False)
            self.command_palette.show(self._enabled_palette_keys())
            self.right_status = "COMMAND"
            self._show_feedback("COMMAND PALETTE", now)
            return True
        # Until the palette actually opens, a released pinch still needs to
        # finish its click/drag. An open palm is a natural pinch release pose.
        return False

    def _update_right_hand(
        self,
        observations,
        lock_states,
        now: float,
        cursor_position: tuple[int, int] | None,
        preview_size: tuple[int, int] | None,
    ) -> None:
        right_observation = observations.get("RIGHT")
        if right_observation is None or lock_states["RIGHT"].locked:
            self._release_drag()
            self.primary_pinch_detector.reset(require_release=True)
            self.secondary_pinch_detector.reset(require_release=True)
            self.palette_open_detector.reset(require_release=True)
            self._close_palette()
            self.right_status = None
            self.diagnostic_text = (
                "RIGHT: hand not seen"
                if right_observation is None
                else "RIGHT: LOCKED - open hand to rearm"
            )
            return

        metrics = measure_desktop_pinch(right_observation[1])
        if self._update_palette(
            metrics,
            now,
            cursor_position,
            preview_size,
        ):
            # Opening the hand ends a drag immediately, even while the
            # palette's opening dwell is still in progress.
            self._release_drag()
            self.primary_pinch_detector.reset(require_release=True)
            self.diagnostic_text = "RIGHT COMMAND: use left pointer, right pinch selects"
            return

        drag_enabled = self._feature_enabled("drag_selection")
        if self.primary_pinch_detector.dragging and not drag_enabled:
            self._release_drag()
            self.primary_pinch_detector.reset(require_release=True)

        if metrics.pose == SECONDARY_PINCH:
            if self.primary_pinch_detector.dragging:
                self._release_drag()
            self.primary_pinch_detector.reset(require_release=True)
            self.right_status = "CLICK"
            if self.secondary_pinch_detector.update(True, now):
                if self._feature_enabled("right_click"):
                    self.click("right")
                    self._show_feedback("RIGHT CLICK", now)
                else:
                    self._show_feedback("RIGHT CLICK IS OFF", now)
        else:
            self.secondary_pinch_detector.update(False, now)
            action = self.primary_pinch_detector.update(
                metrics.pose == PRIMARY_PINCH,
                now,
                allow_double=self._feature_enabled("double_click"),
                allow_drag=drag_enabled,
            )
            if self.primary_pinch_detector.dragging:
                self.right_status = "DRAG"
            elif metrics.pose == PRIMARY_PINCH:
                self.right_status = "CLICK"
            else:
                self.right_status = None

            if action == SINGLE_CLICK_ACTION:
                if self._feature_enabled("left_click"):
                    self.click("left")
                    self._show_feedback("LEFT CLICK", now)
                else:
                    self._show_feedback("LEFT CLICK IS OFF", now)
            elif action == DOUBLE_CLICK_ACTION:
                if self._feature_enabled("double_click"):
                    self.second_click("left")
                    self._show_feedback("DOUBLE CLICK", now)
            elif action == DRAG_START_ACTION:
                self.mouse_down("left")
                self._show_feedback("DRAG START", now)
            elif action == DRAG_END_ACTION:
                self.mouse_up("left")
                self._show_feedback("DRAG END", now)

        self.diagnostic_text = (
            "RIGHT PINCH "
            f"index={metrics.thumb_index_distance:.2f} "
            f"middle={metrics.thumb_middle_distance:.2f} | "
            f"{metrics.pose.replace('_', ' ')}"
        )

    def update_gestures(
        self,
        observations,
        lock_states,
        now: float,
        cursor_position: tuple[int, int] | None = None,
        preview_size: tuple[int, int] | None = None,
    ) -> None:
        if self.escape_monitor.consume_escape():
            self.disable("global Esc pressed")
            self._show_feedback("SAFE - ESC", now)

        if self.actions_guarded(now):
            self.reset_actions(require_release=True)
            self.left_status = None
            self.right_status = None
            self.diagnostic_text = "LIVE: lower both hands to re-arm gestures"
            return

        self._update_right_hand(
            observations,
            lock_states,
            now,
            cursor_position,
            preview_size,
        )

        left_observation = observations.get("LEFT")
        if (
            left_observation is None
            or lock_states["LEFT"].locked
            or not self._feature_enabled("scroll")
        ):
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
        if self.command_palette is not None:
            self.command_palette.close()
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
