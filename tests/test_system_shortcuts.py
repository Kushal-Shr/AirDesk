"""Stage 7 swipe tests with no real keyboard or system shortcut output."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from airdesk.shortcut_config import NEXT_APP_KEYS
from airdesk.system_shortcuts import (
    SWIPE_LEFT,
    SWIPE_RIGHT,
    SWIPE_UP,
    ShortcutControlController,
    SwipeGestureDetector,
    SwipeMetrics,
    classify_swipe_pose,
)


def make_hand(index_up=True, middle_up=True, ring_up=True, little_up=False):
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


def translated_hand(points, dx=0.0, dy=0.0):
    return [
        SimpleNamespace(x=point.x + dx, y=point.y + dy, z=point.z)
        for point in points
    ]


class FakeBackend:
    class FailSafeException(Exception):
        pass

    FAILSAFE = False
    PAUSE = 1.0

    def __init__(self):
        self.hotkeys = []

    def size(self):
        return 1920, 1080

    def moveTo(self, _x, _y, _pause):
        pass

    def click(self, _button, _pause):
        pass

    def scroll(self, _steps, _pause):
        pass

    def hotkey(self, *keys, _pause):
        self.hotkeys.append((keys, _pause))


class FakeEscapeMonitor:
    def consume_escape(self):
        return False

    def close(self):
        pass


class SwipePoseTests(unittest.TestCase):
    def test_three_fingers_are_swipe_pose(self):
        self.assertTrue(classify_swipe_pose(make_hand()).is_swipe_pose)

    def test_two_or_four_fingers_are_not_swipe_pose(self):
        self.assertFalse(
            classify_swipe_pose(make_hand(ring_up=False)).is_swipe_pose
        )
        self.assertFalse(
            classify_swipe_pose(make_hand(little_up=True)).is_swipe_pose
        )


class SwipeDetectorTests(unittest.TestCase):
    def start_detector(self):
        detector = SwipeGestureDetector()
        center = SwipeMetrics(True, 0.50, 0.50, 0.25)
        detector.update(center, 1.00)
        detector.update(center, 1.19)
        return detector

    def test_right_swipe(self):
        detector = self.start_detector()
        self.assertEqual(
            detector.update(SwipeMetrics(True, 0.68, 0.50, 0.25), 1.25),
            SWIPE_RIGHT,
        )

    def test_left_swipe(self):
        detector = self.start_detector()
        self.assertEqual(
            detector.update(SwipeMetrics(True, 0.32, 0.50, 0.25), 1.25),
            SWIPE_LEFT,
        )

    def test_up_swipe(self):
        detector = self.start_detector()
        self.assertEqual(
            detector.update(SwipeMetrics(True, 0.50, 0.32, 0.25), 1.25),
            SWIPE_UP,
        )

    def test_diagonal_motion_is_rejected(self):
        detector = self.start_detector()
        self.assertIsNone(
            detector.update(SwipeMetrics(True, 0.68, 0.32, 0.25), 1.25)
        )

    def test_held_pose_does_not_repeat(self):
        detector = self.start_detector()
        moved = SwipeMetrics(True, 0.68, 0.50, 0.25)
        self.assertEqual(detector.update(moved, 1.25), SWIPE_RIGHT)
        self.assertIsNone(detector.update(moved, 2.50))


class ShortcutControllerTests(unittest.TestCase):
    def test_active_right_swipe_emits_configured_hotkey(self):
        backend = FakeBackend()
        controller = ShortcutControlController(backend, FakeEscapeMonitor())
        controller.toggle()
        unlocked = SimpleNamespace(locked=False)
        locked = SimpleNamespace(locked=True)
        locks = {"LEFT": locked, "RIGHT": unlocked}

        # Enabling real output requires a neutral/non-swipe pose before arming.
        neutral_hand = make_hand(middle_up=False, ring_up=False)
        controller.update_gestures(
            {"RIGHT": (1.0, neutral_hand, None)}, locks, 0.80
        )

        start_hand = make_hand()
        controller.update_gestures(
            {"RIGHT": (1.0, start_hand, None)}, locks, 1.00
        )
        controller.update_gestures(
            {"RIGHT": (1.0, start_hand, None)}, locks, 1.19
        )
        controller.update_gestures(
            {"RIGHT": (1.0, translated_hand(start_hand, dx=0.18), None)},
            locks,
            1.25,
        )

        self.assertEqual(backend.hotkeys, [(NEXT_APP_KEYS, False)])


if __name__ == "__main__":
    unittest.main()
