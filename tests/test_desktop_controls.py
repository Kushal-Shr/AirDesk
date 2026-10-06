"""Deterministic Stage 6 tests with no real mouse or keyboard output."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from airdesk.desktop_controls import (
    DOUBLE_CLICK_ACTION,
    DRAG_END_ACTION,
    DRAG_START_ACTION,
    LEFT_CLICK,
    NO_CLICK,
    NO_PINCH,
    PRIMARY_PINCH,
    RIGHT_CLICK,
    SECONDARY_PINCH,
    SINGLE_CLICK_ACTION,
    ClickGestureDetector,
    DesktopControlController,
    PrimaryPinchDetector,
    ScrollGestureDetector,
    ScrollMetrics,
    classify_click_pose,
    classify_scroll_pose,
    measure_click_pose,
    measure_desktop_pinch,
)


def make_hand(index_up=True, middle_up=True, ring_up=False, little_up=False):
    points = [SimpleNamespace(x=0.5, y=0.7, z=0.0) for _ in range(21)]
    points[0] = SimpleNamespace(x=0.5, y=0.9, z=0.0)
    points[4] = SimpleNamespace(x=0.28, y=0.55, z=0.0)
    extensions = (index_up, middle_up, ring_up, little_up)
    for finger_index, (mcp_id, pip_id, dip_id, tip_id) in enumerate(
        ((5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20))
    ):
        x = 0.38 + finger_index * 0.08
        extended = extensions[finger_index]
        points[mcp_id] = SimpleNamespace(x=x, y=0.65, z=0.0)
        points[pip_id] = SimpleNamespace(x=x, y=0.50, z=0.0)
        points[dip_id] = SimpleNamespace(x=x, y=0.35 if extended else 0.60, z=0.0)
        points[tip_id] = SimpleNamespace(x=x, y=0.20 if extended else 0.72, z=0.0)
    return points


class FakeBackend:
    class FailSafeException(Exception):
        pass

    FAILSAFE = False
    PAUSE = 1.0

    def __init__(self):
        self.clicks = []
        self.double_clicks = []
        self.button_events = []
        self.scrolls = []

    def size(self):
        return 1920, 1080

    def moveTo(self, _x, _y, _pause):
        pass

    def click(self, button, _pause):
        self.clicks.append((button, _pause))

    def doubleClick(self, button, interval, _pause):
        self.double_clicks.append((button, interval, _pause))

    def mouseDown(self, button, _pause):
        self.button_events.append(("down", button, _pause))

    def mouseUp(self, button, _pause):
        self.button_events.append(("up", button, _pause))

    def scroll(self, steps, _pause):
        self.scrolls.append((steps, _pause))


class FakeEscapeMonitor:
    def __init__(self):
        self.escape_pressed = False
        self.closed = False

    def consume_escape(self):
        result = self.escape_pressed
        self.escape_pressed = False
        return result

    def close(self):
        self.closed = True


class ClickPoseTests(unittest.TestCase):
    def test_index_only_is_left_click(self):
        hand = make_hand(True, False, False, False)
        self.assertEqual(classify_click_pose(hand), LEFT_CLICK)

    def test_index_and_middle_are_right_click(self):
        hand = make_hand(True, True, False, False)
        self.assertEqual(classify_click_pose(hand), RIGHT_CLICK)

    def test_open_hand_has_no_click(self):
        self.assertEqual(classify_click_pose(make_hand(True, True, True, True)), NO_CLICK)

    def test_fist_has_no_click(self):
        self.assertEqual(classify_click_pose(make_hand(False, False, False, False)), NO_CLICK)

    def test_extended_thumb_rejects_click_pose(self):
        hand = make_hand(True, False, False, False)
        hand[4] = SimpleNamespace(x=0.05, y=0.55, z=0.0)
        metrics = measure_click_pose(hand)
        self.assertFalse(metrics.thumb_folded)
        self.assertEqual(metrics.pose, NO_CLICK)


class ClickStateMachineTests(unittest.TestCase):
    def test_dwell_then_one_action_until_release(self):
        detector = ClickGestureDetector()
        self.assertIsNone(detector.update(LEFT_CLICK, 1.00))
        self.assertIsNone(detector.update(LEFT_CLICK, 1.29))
        self.assertEqual(detector.update(LEFT_CLICK, 1.31), LEFT_CLICK)
        self.assertIsNone(detector.update(LEFT_CLICK, 1.50))
        detector.update(NO_CLICK, 1.51)

    def test_one_brief_dropout_does_not_erase_dwell(self):
        detector = ClickGestureDetector()
        detector.update(LEFT_CLICK, 1.00)
        detector.update(NO_CLICK, 1.05)
        detector.update(LEFT_CLICK, 1.10)
        self.assertEqual(detector.update(LEFT_CLICK, 1.31), LEFT_CLICK)

    def test_switching_to_two_fingers_favors_right_click(self):
        detector = ClickGestureDetector()
        detector.update(LEFT_CLICK, 1.00)
        detector.update(RIGHT_CLICK, 1.10)
        self.assertIsNone(detector.update(RIGHT_CLICK, 1.29))
        self.assertEqual(detector.update(RIGHT_CLICK, 1.31), RIGHT_CLICK)

    def test_reset_requires_release_before_new_action(self):
        detector = ClickGestureDetector()
        detector.reset(require_release=True)
        self.assertIsNone(detector.update(LEFT_CLICK, 1.00))
        self.assertIsNone(detector.update(LEFT_CLICK, 2.00))
        detector.update(NO_CLICK, 2.01)
        detector.update(NO_CLICK, 2.12)
        detector.update(LEFT_CLICK, 2.20)
        self.assertEqual(detector.update(LEFT_CLICK, 2.51), LEFT_CLICK)


class PinchGestureTests(unittest.TestCase):
    @staticmethod
    def primary_pinch_hand():
        hand = make_hand(True, True, False, False)
        hand[4] = SimpleNamespace(x=0.39, y=0.21, z=0.0)
        return hand

    @staticmethod
    def secondary_pinch_hand():
        hand = make_hand(True, True, False, False)
        hand[4] = SimpleNamespace(x=0.43, y=0.21, z=0.0)
        hand[8] = SimpleNamespace(x=0.42, y=0.20, z=0.0)
        hand[12] = SimpleNamespace(x=0.44, y=0.20, z=0.0)
        return hand

    def test_normalized_pinch_pose_distinguishes_primary_and_secondary(self):
        self.assertEqual(
            measure_desktop_pinch(self.primary_pinch_hand()).pose,
            PRIMARY_PINCH,
        )
        self.assertEqual(
            measure_desktop_pinch(self.secondary_pinch_hand()).pose,
            SECONDARY_PINCH,
        )
        self.assertEqual(measure_desktop_pinch(make_hand()).pose, NO_PINCH)

    def test_one_quick_pinch_emits_single_click(self):
        detector = PrimaryPinchDetector()
        detector.update(True, 1.00, allow_double=False, allow_drag=True)
        self.assertEqual(
            detector.update(False, 1.15, allow_double=False, allow_drag=True),
            SINGLE_CLICK_ACTION,
        )

    def test_two_quick_pinches_emit_one_double_click(self):
        detector = PrimaryPinchDetector()
        detector.update(True, 1.00, allow_double=True, allow_drag=True)
        self.assertIsNone(
            detector.update(False, 1.10, allow_double=True, allow_drag=True)
        )
        detector.update(True, 1.22, allow_double=True, allow_drag=True)
        self.assertEqual(
            detector.update(False, 1.32, allow_double=True, allow_drag=True),
            DOUBLE_CLICK_ACTION,
        )

    def test_held_pinch_starts_drag_and_release_ends_it(self):
        detector = PrimaryPinchDetector()
        detector.update(True, 1.00, allow_double=True, allow_drag=True)
        self.assertEqual(
            detector.update(True, 1.53, allow_double=True, allow_drag=True),
            DRAG_START_ACTION,
        )
        self.assertEqual(
            detector.update(False, 1.60, allow_double=True, allow_drag=True),
            DRAG_END_ACTION,
        )

    def test_controller_sends_one_native_double_click(self):
        backend = FakeBackend()
        controller = DesktopControlController(backend, FakeEscapeMonitor())
        controller.toggle()
        unlocked = SimpleNamespace(locked=False)
        locked = SimpleNamespace(locked=True)
        locks = {"LEFT": locked, "RIGHT": unlocked}
        neutral = make_hand()
        pinch = self.primary_pinch_hand()

        controller.update_gestures({"RIGHT": (1.0, neutral, None)}, locks, 0.80)
        controller.update_gestures({"RIGHT": (1.0, pinch, None)}, locks, 1.00)
        controller.update_gestures({"RIGHT": (1.0, neutral, None)}, locks, 1.10)
        controller.update_gestures({"RIGHT": (1.0, pinch, None)}, locks, 1.22)
        controller.update_gestures({"RIGHT": (1.0, neutral, None)}, locks, 1.32)

        self.assertEqual(backend.double_clicks, [("left", 0.12, False)])
        self.assertEqual(backend.clicks, [])

    def test_lock_releases_an_active_drag(self):
        backend = FakeBackend()
        controller = DesktopControlController(backend, FakeEscapeMonitor())
        controller.toggle()
        unlocked = SimpleNamespace(locked=False)
        locked = SimpleNamespace(locked=True)
        hand = self.primary_pinch_hand()

        # Enabling ACTIVE requires one neutral frame before gestures rearm.
        controller.update_gestures(
            {"RIGHT": (1.0, make_hand(), None)},
            {"LEFT": locked, "RIGHT": unlocked},
            0.90,
        )

        controller.update_gestures(
            {"RIGHT": (1.0, hand, None)},
            {"LEFT": locked, "RIGHT": unlocked},
            1.00,
        )
        controller.update_gestures(
            {"RIGHT": (1.0, hand, None)},
            {"LEFT": locked, "RIGHT": unlocked},
            1.53,
        )
        controller.update_gestures(
            {"RIGHT": (1.0, hand, None)},
            {"LEFT": locked, "RIGHT": locked},
            1.60,
        )

        self.assertEqual(
            backend.button_events,
            [("down", "left", False), ("up", "left", False)],
        )


class ScrollTests(unittest.TestCase):
    def test_only_index_middle_pose_is_scroll(self):
        self.assertTrue(classify_scroll_pose(make_hand()).is_scroll_pose)
        self.assertFalse(
            classify_scroll_pose(make_hand(ring_up=True, little_up=True)).is_scroll_pose
        )

    def test_up_and_down_motion_emit_opposite_steps(self):
        detector = ScrollGestureDetector()
        base = ScrollMetrics(True, hand_y=0.50, palm_size=0.25)
        detector.update(base, 1.00)
        detector.update(base, 1.19)
        self.assertEqual(
            detector.update(ScrollMetrics(True, 0.44, 0.25), 1.30),
            3,
        )
        self.assertEqual(
            detector.update(ScrollMetrics(True, 0.50, 0.25), 1.41),
            -3,
        )


class ControllerSafetyTests(unittest.TestCase):
    def test_global_escape_disables_output(self):
        backend = FakeBackend()
        monitor = FakeEscapeMonitor()
        controller = DesktopControlController(backend, monitor)
        controller.toggle()
        monitor.escape_pressed = True
        locked = SimpleNamespace(locked=True)
        controller.update_gestures({}, {"LEFT": locked, "RIGHT": locked}, 1.0)
        self.assertFalse(controller.enabled)

    def test_close_stops_escape_monitor(self):
        controller = DesktopControlController(FakeBackend(), FakeEscapeMonitor())
        monitor = controller.escape_monitor
        controller.close()
        self.assertTrue(monitor.closed)


if __name__ == "__main__":
    unittest.main()
