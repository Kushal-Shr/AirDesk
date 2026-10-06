"""Stage 8 air-writing tests without camera or real system input."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from airdesk.native_overlay import MemoryInkOverlay
from airdesk.character_recognition import RecognitionResult
from airdesk.whiteboard import (
    AirWritingController,
    HoldLatch,
    is_open_palm,
    is_pen_pinch,
    is_thumbs_up,
)


def make_hand(open_palm=False, pinched=False, dx=0.0):
    points = [SimpleNamespace(x=0.5 + dx, y=0.7, z=0.0) for _ in range(21)]
    points[0] = SimpleNamespace(x=0.5 + dx, y=0.9, z=0.0)
    points[4] = SimpleNamespace(x=0.16 + dx, y=0.45, z=0.0)

    for finger_index, (mcp_id, pip_id, dip_id, tip_id) in enumerate(
        ((5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20))
    ):
        x = 0.38 + finger_index * 0.08 + dx
        points[mcp_id] = SimpleNamespace(x=x, y=0.65, z=0.0)
        points[pip_id] = SimpleNamespace(x=x, y=0.50, z=0.0)
        points[dip_id] = SimpleNamespace(
            x=x,
            y=0.35 if open_palm else 0.60,
            z=0.0,
        )
        points[tip_id] = SimpleNamespace(
            x=x,
            y=0.20 if open_palm else 0.72,
            z=0.0,
        )

    if pinched:
        points[4] = SimpleNamespace(
            x=points[8].x,
            y=points[8].y,
            z=points[8].z,
        )
    return points


def make_thumbs_up_hand():
    points = make_hand(open_palm=False)
    points[2] = SimpleNamespace(x=0.43, y=0.66, z=0.0)
    points[3] = SimpleNamespace(x=0.43, y=0.45, z=0.0)
    points[4] = SimpleNamespace(x=0.43, y=0.18, z=0.0)
    return points


class FakeSystemController:
    def __init__(self):
        self.enabled = True
        self.disable_reasons = []
        self.committed_text = []

    def disable(self, reason):
        self.enabled = False
        self.disable_reasons.append(reason)

    def commit_text(self, text):
        self.committed_text.append(text)
        return True


class FakeRecognizer:
    def __init__(self, result=RecognitionResult("A", 0.91)):
        self.result = result
        self.calls = 0

    def recognize(self, _canvas):
        self.calls += 1
        return self.result


class PoseTests(unittest.TestCase):
    def test_open_palm_requires_extended_fingers(self):
        self.assertTrue(is_open_palm(make_hand(open_palm=True)))
        self.assertFalse(is_open_palm(make_hand(open_palm=False)))

    def test_pen_pinch_uses_thumb_and_index(self):
        self.assertTrue(is_pen_pinch(make_hand(open_palm=True, pinched=True)))
        self.assertFalse(is_pen_pinch(make_hand(open_palm=True, pinched=False)))

    def test_thumbs_up_requires_raised_thumb_and_curled_fingers(self):
        self.assertTrue(is_thumbs_up(make_thumbs_up_hand()))
        self.assertFalse(is_thumbs_up(make_hand(open_palm=True)))


class HoldLatchTests(unittest.TestCase):
    def test_hold_fires_once_and_requires_release(self):
        hold = HoldLatch(duration=1.0)
        self.assertFalse(hold.update(True, 1.0))
        self.assertFalse(hold.update(True, 1.9))
        self.assertTrue(hold.update(True, 2.0))
        self.assertFalse(hold.update(True, 3.0))
        hold.update(False, 3.1)
        self.assertTrue(hold.latched)
        hold.update(False, 3.3)
        self.assertFalse(hold.latched)

    def test_brief_tracking_dropout_keeps_hold_progress(self):
        hold = HoldLatch(duration=1.0)
        hold.update(True, 1.0)
        hold.update(True, 1.5)
        hold.update(False, 1.6)
        self.assertGreaterEqual(hold.progress, 0.5)
        self.assertTrue(hold.update(True, 2.01))


class AirWritingControllerTests(unittest.TestCase):
    def setUp(self):
        self.overlay = MemoryInkOverlay(size=(1280, 800))
        self.recognizer = FakeRecognizer()
        self.controller = AirWritingController(self.overlay, self.recognizer)
        self.system = FakeSystemController()
        self.unlocked = {
            "LEFT": SimpleNamespace(locked=False),
            "RIGHT": SimpleNamespace(locked=False),
        }

    def test_both_open_palms_toggle_mode_and_disable_system_input(self):
        open_hand = make_hand(open_palm=True)
        observations = {
            "LEFT": (1.0, open_hand, None),
            "RIGHT": (1.0, open_hand, None),
        }
        self.controller.update(observations, self.unlocked, 1.0, 640, 480, self.system)
        self.controller.update(observations, self.unlocked, 2.01, 640, 480, self.system)
        self.assertTrue(self.controller.is_whiteboard)
        self.assertTrue(self.overlay.visible)
        self.assertFalse(self.system.enabled)

        # Keeping both palms held cannot immediately toggle back.
        self.controller.update(observations, self.unlocked, 3.0, 640, 480, self.system)
        self.assertTrue(self.controller.is_whiteboard)

        # Release, then perform a fresh one-second hold to return.
        self.controller.update({}, self.unlocked, 3.3, 640, 480, self.system)
        self.controller.update(observations, self.unlocked, 4.0, 640, 480, self.system)
        self.controller.update(observations, self.unlocked, 5.01, 640, 480, self.system)
        self.assertFalse(self.controller.is_whiteboard)
        self.assertFalse(self.overlay.visible)

    def test_mode_switch_ignores_duplicate_handedness_labels(self):
        open_hand = make_hand(open_palm=True)
        observations = {"LEFT": (1.0, open_hand, None)}
        all_hands = [open_hand, open_hand]
        self.controller.update(
            observations,
            self.unlocked,
            1.0,
            640,
            480,
            self.system,
            all_hands=all_hands,
        )
        self.controller.update(
            observations,
            self.unlocked,
            2.01,
            640,
            480,
            self.system,
            all_hands=all_hands,
        )
        self.assertTrue(self.controller.is_whiteboard)

    def test_pinched_right_index_draws_and_locked_hand_lifts_pen(self):
        self.controller.mode = "AIR_WRITE"
        first = make_hand(open_palm=True, pinched=True)
        second = make_hand(open_palm=True, pinched=True, dx=0.05)
        self.controller.update(
            {"RIGHT": (1.0, first, None)},
            self.unlocked,
            1.0,
            640,
            480,
            self.system,
        )
        self.controller.update(
            {"RIGHT": (1.0, second, None)},
            self.unlocked,
            1.1,
            640,
            480,
            self.system,
        )
        self.assertTrue(np.any(self.controller.canvas < 255))
        self.assertTrue(self.controller.pen_down)
        drawn_points = [point for stroke in self.overlay.strokes for point in stroke]
        self.assertGreaterEqual(len(drawn_points), 2)
        self.assertTrue(all(0 <= x < 1280 and 0 <= y < 800 for x, y in drawn_points))

        locked = dict(self.unlocked)
        locked["RIGHT"] = SimpleNamespace(locked=True)
        self.controller.update(
            {"RIGHT": (1.0, second, None)},
            locked,
            1.2,
            640,
            480,
            self.system,
        )
        self.assertFalse(self.controller.pen_down)
        self.assertIsNone(self.controller.previous_pen_point)
        self.assertEqual(self.overlay.strokes[-1], [])

    def test_left_open_palm_clears_after_hold(self):
        self.controller.mode = "AIR_WRITE"
        self.controller.ensure_canvas(640, 480)
        self.controller.canvas[250, 250] = 0
        self.overlay.add_point((100, 100))
        left_open = {"LEFT": (1.0, make_hand(open_palm=True), None)}
        self.controller.update(
            left_open, self.unlocked, 1.0, 640, 480, self.system
        )
        self.assertTrue(np.any(self.controller.canvas < 255))
        self.controller.update(
            left_open, self.unlocked, 2.01, 640, 480, self.system
        )
        self.assertTrue(np.all(self.controller.canvas == 255))
        self.assertEqual(self.overlay.strokes, [])

    def test_thumbs_up_recognizes_types_and_clears_one_character(self):
        self.controller.mode = "AIR_WRITE"
        self.controller.ensure_canvas(640, 480)
        self.controller.canvas[200:260, 300:306] = 0
        self.overlay.add_point((600, 300))
        thumbs_up = {"RIGHT": (1.0, make_thumbs_up_hand(), None)}

        self.controller.update(
            thumbs_up, self.unlocked, 1.0, 640, 480, self.system
        )
        self.controller.update(
            thumbs_up, self.unlocked, 1.66, 640, 480, self.system
        )

        self.assertEqual(self.recognizer.calls, 1)
        self.assertEqual(self.system.committed_text, ["A"])
        self.assertTrue(np.all(self.controller.canvas == 255))
        self.assertEqual(self.overlay.strokes, [])
        self.assertIn("INSERTED: A", self.controller.recognition_feedback)

        # A held confirmation cannot type the same character repeatedly.
        self.controller.update(
            thumbs_up, self.unlocked, 2.5, 640, 480, self.system
        )
        self.assertEqual(self.system.committed_text, ["A"])

    def test_escape_safety_blocks_confirmed_text(self):
        self.controller.mode = "AIR_WRITE"
        self.controller.ensure_canvas(640, 480)
        self.controller.canvas[200:260, 300:306] = 0
        self.controller.emergency_stop()
        thumbs_up = {"RIGHT": (1.0, make_thumbs_up_hand(), None)}

        self.controller.update(
            thumbs_up, self.unlocked, 1.0, 640, 480, self.system
        )
        self.controller.update(
            thumbs_up, self.unlocked, 1.66, 640, 480, self.system
        )

        self.assertEqual(self.recognizer.calls, 0)
        self.assertEqual(self.system.committed_text, [])
        self.assertIn("STOPPED BY ESC", self.controller.recognition_feedback)


if __name__ == "__main__":
    unittest.main()
