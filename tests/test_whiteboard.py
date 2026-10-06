"""Stage 8 air-writing tests without camera or real system input."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from airdesk.native_overlay import MemoryInkOverlay
from airdesk.character_recognition import RecognitionResult
from airdesk.gemini_correction import GeminiCorrectionResult
from airdesk.whiteboard import (
    AdaptivePointSmoother,
    AirWritingController,
    HoldLatch,
    SentenceBuffer,
    WholeLineWritingController,
    is_open_palm,
    is_pen_pinch,
    is_stroke_undo_pose,
    is_mode_switch_pose,
    is_thumbs_up,
    measure_pen_pose,
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


def make_mode_switch_hand():
    points = make_hand(open_palm=False)
    for dip_id, tip_id in ((7, 8), (11, 12)):
        points[dip_id] = SimpleNamespace(x=points[dip_id].x, y=0.35, z=0.0)
        points[tip_id] = SimpleNamespace(x=points[tip_id].x, y=0.20, z=0.0)
    return points


def make_stroke_undo_hand():
    return make_hand(open_palm=True, pinched=True)


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


class FakeSymbolRecognizer(FakeRecognizer):
    mode = "symbols"
    preserves_position = True

    def recognize_strokes(self, strokes, guide):
        self.calls += 1
        self.received_strokes = strokes
        self.received_guide = guide
        return self.result


class FakeMultiModeRecognizer(FakeRecognizer):
    modes = ("uppercase", "lowercase", "digits", "symbols")

    def __init__(self, mode="digits"):
        super().__init__()
        self.mode = mode

    @property
    def preserves_position(self):
        return self.mode == "symbols"

    def cycle_mode(self):
        index = self.modes.index(self.mode)
        self.mode = self.modes[(index + 1) % len(self.modes)]
        return self.mode


class FakeLineRecognizer:
    def __init__(self, text="Hi 2!", confidence=0.94):
        self.result = RecognitionResult(text, confidence)
        self.calls = 0

    def recognize(self, _canvas):
        self.calls += 1
        return self.result


class FakeGeminiReviewer:
    def __init__(self, text="Hello 2!", confidence=0.96):
        self.result = GeminiCorrectionResult(text, confidence, "visual correction")
        self.calls = []

    def review(self, canvas, local_text, predictions):
        self.calls.append((canvas.copy(), local_text, predictions))
        return self.result


class PoseTests(unittest.TestCase):
    def test_open_palm_requires_extended_fingers(self):
        self.assertTrue(is_open_palm(make_hand(open_palm=True)))
        self.assertFalse(is_open_palm(make_hand(open_palm=False)))
        self.assertFalse(is_open_palm(make_hand(open_palm=True, pinched=True)))

    def test_pen_pinch_requires_tight_pinch_and_open_other_fingers(self):
        self.assertTrue(is_pen_pinch(make_hand(open_palm=True, pinched=True)))
        self.assertFalse(is_pen_pinch(make_hand(open_palm=True, pinched=False)))
        self.assertFalse(is_pen_pinch(make_hand(open_palm=False, pinched=True)))

        metrics = measure_pen_pose(make_hand(open_palm=True, pinched=True))
        self.assertGreaterEqual(metrics.minimum_other_openness, 0.80)

    def test_thumbs_up_requires_raised_thumb_and_curled_fingers(self):
        self.assertTrue(is_thumbs_up(make_thumbs_up_hand()))
        self.assertFalse(is_thumbs_up(make_hand(open_palm=True)))

    def test_mode_switch_requires_index_and_middle_only(self):
        self.assertTrue(is_mode_switch_pose(make_mode_switch_hand()))
        self.assertFalse(is_mode_switch_pose(make_hand(open_palm=True)))
        self.assertFalse(is_mode_switch_pose(make_hand(open_palm=False)))

    def test_stroke_undo_pose_is_distinct_from_palm_and_fist(self):
        pose = make_stroke_undo_hand()
        self.assertTrue(is_stroke_undo_pose(pose))
        self.assertFalse(is_open_palm(pose))
        self.assertFalse(is_stroke_undo_pose(make_hand(open_palm=True)))


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


class AdaptivePointSmootherTests(unittest.TestCase):
    def test_small_movements_are_smoothed_and_reset_starts_fresh(self):
        smoother = AdaptivePointSmoother()
        self.assertEqual(smoother.update((100, 100)), (100, 100))
        smoothed = smoother.update((110, 100))
        self.assertGreater(smoothed[0], 100)
        self.assertLess(smoothed[0], 110)

        smoother.reset()
        self.assertEqual(smoother.update((300, 200)), (300, 200))


class SentenceBufferTests(unittest.TestCase):
    def test_character_space_and_undo(self):
        sentence = SentenceBuffer(maximum_length=5)
        self.assertFalse(sentence.append_space())
        self.assertTrue(sentence.append_character("h"))
        self.assertTrue(sentence.append_character("i"))
        self.assertTrue(sentence.append_space())
        self.assertFalse(sentence.append_space())
        self.assertTrue(sentence.undo())
        self.assertEqual(sentence.text, "hi")

    def test_length_limit_keeps_existing_text(self):
        sentence = SentenceBuffer(text="hello", maximum_length=5)
        self.assertFalse(sentence.append_character("!"))
        self.assertEqual(sentence.text, "hello")


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
        self.assertTrue(self.overlay.cursor_active)
        self.assertIsNotNone(self.overlay.cursor_point)
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
        self.assertIsNone(self.overlay.cursor_point)

    def test_released_pinch_keeps_aiming_cursor_without_drawing(self):
        self.controller.mode = "AIR_WRITE"
        pointing = make_hand(open_palm=False, pinched=False)

        self.controller.update(
            {"RIGHT": (1.0, pointing, None)},
            self.unlocked,
            1.0,
            640,
            480,
            self.system,
        )

        self.assertIsNotNone(self.overlay.cursor_point)
        self.assertFalse(self.overlay.cursor_active)
        self.assertFalse(self.controller.pen_down)
        self.assertEqual(self.overlay.strokes, [])

        self.controller.update({}, self.unlocked, 1.1, 640, 480, self.system)
        self.assertIsNone(self.overlay.cursor_point)

    def test_pinch_with_curled_other_fingers_is_aim_only(self):
        self.controller.mode = "AIR_WRITE"
        invalid_writing_pose = make_hand(open_palm=False, pinched=True)

        self.controller.update(
            {"RIGHT": (1.0, invalid_writing_pose, None)},
            self.unlocked,
            1.0,
            640,
            480,
            self.system,
        )

        self.assertIsNotNone(self.overlay.cursor_point)
        self.assertFalse(self.overlay.cursor_active)
        self.assertFalse(self.controller.pen_down)
        self.assertEqual(self.overlay.strokes, [])

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

    def test_right_thumb_buffers_character_then_both_thumbs_insert(self):
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
        self.assertEqual(self.system.committed_text, [])
        self.assertEqual(self.controller.sentence.text, "A")
        self.assertTrue(np.all(self.controller.canvas == 255))
        self.assertEqual(self.overlay.strokes, [])
        self.assertIn("ADDED: A", self.controller.recognition_feedback)

        both_thumbs = {
            "LEFT": (1.0, make_thumbs_up_hand(), None),
            "RIGHT": (1.0, make_thumbs_up_hand(), None),
        }
        self.controller.update(
            both_thumbs, self.unlocked, 3.0, 640, 480, self.system
        )
        self.controller.update(
            both_thumbs, self.unlocked, 3.91, 640, 480, self.system
        )
        self.assertEqual(self.system.committed_text, ["A "])
        self.assertEqual(self.controller.sentence.text, "")

        # A held confirmation cannot type the same character repeatedly.
        self.controller.update(
            both_thumbs, self.unlocked, 4.5, 640, 480, self.system
        )
        self.assertEqual(self.system.committed_text, ["A "])

    def test_left_thumb_adds_space_and_left_pinch_undoes(self):
        self.controller.mode = "AIR_WRITE"
        self.controller.sentence.text = "HI"
        left_thumb = {"LEFT": (1.0, make_thumbs_up_hand(), None)}
        self.controller.update(left_thumb, self.unlocked, 1.0, 640, 480, self.system)
        self.controller.update(left_thumb, self.unlocked, 1.66, 640, 480, self.system)
        self.assertEqual(self.controller.sentence.text, "HI ")

        self.controller.update({}, self.unlocked, 2.0, 640, 480, self.system)
        left_pinch = {"LEFT": (1.0, make_hand(open_palm=False, pinched=True), None)}
        self.controller.update(left_pinch, self.unlocked, 2.3, 640, 480, self.system)
        self.controller.update(left_pinch, self.unlocked, 2.96, 640, 480, self.system)
        self.assertEqual(self.controller.sentence.text, "HI")

    def test_escape_safety_blocks_confirmed_text(self):
        self.controller.mode = "AIR_WRITE"
        self.controller.ensure_canvas(640, 480)
        self.controller.sentence.text = "SAFE"
        self.controller.emergency_stop()
        both_thumbs = {
            "LEFT": (1.0, make_thumbs_up_hand(), None),
            "RIGHT": (1.0, make_thumbs_up_hand(), None),
        }

        self.controller.update(
            both_thumbs, self.unlocked, 1.0, 640, 480, self.system
        )
        self.controller.update(
            both_thumbs, self.unlocked, 1.91, 640, 480, self.system
        )

        self.assertEqual(self.recognizer.calls, 0)
        self.assertEqual(self.system.committed_text, [])
        self.assertEqual(self.controller.sentence.text, "SAFE")
        self.assertIn("STOPPED BY ESC", self.controller.recognition_feedback)

    def test_symbol_mode_uses_position_preserving_strokes(self):
        recognizer = FakeSymbolRecognizer(RecognitionResult(".", 0.93))
        controller = AirWritingController(self.overlay, recognizer)
        controller.mode = "AIR_WRITE"
        controller.ensure_canvas(640, 480)
        self.overlay.add_point((640, 650))
        thumbs_up = {"RIGHT": (1.0, make_thumbs_up_hand(), None)}

        controller.update(thumbs_up, self.unlocked, 1.0, 640, 480, self.system)
        controller.update(thumbs_up, self.unlocked, 1.66, 640, 480, self.system)

        self.assertEqual(self.system.committed_text, [])
        self.assertEqual(controller.sentence.text, ".")
        self.assertEqual(recognizer.calls, 1)
        self.assertIsNotNone(recognizer.received_guide)

    def test_left_two_finger_hold_cycles_mode_and_preserves_sentence(self):
        recognizer = FakeMultiModeRecognizer(mode="digits")
        controller = AirWritingController(self.overlay, recognizer)
        controller.mode = "AIR_WRITE"
        controller.sentence.text = "Room 2"
        controller.ensure_canvas(640, 480)
        controller.canvas[200:220, 300:305] = 0
        self.overlay.add_point((500, 300))
        mode_pose = {"LEFT": (1.0, make_mode_switch_hand(), None)}

        controller.update(mode_pose, self.unlocked, 1.0, 640, 480, self.system)
        controller.update(mode_pose, self.unlocked, 1.81, 640, 480, self.system)

        self.assertEqual(controller.recognition_mode, "symbols")
        self.assertEqual(controller.sentence.text, "Room 2")
        self.assertIsNotNone(controller.writing_guide)
        self.assertIsNotNone(self.overlay.guide)
        self.assertTrue(np.all(controller.canvas == 255))

        # Holding the pose cannot skip repeatedly; release is required.
        controller.update(mode_pose, self.unlocked, 2.8, 640, 480, self.system)
        self.assertEqual(controller.recognition_mode, "symbols")
        controller.update({}, self.unlocked, 3.1, 640, 480, self.system)
        controller.update(mode_pose, self.unlocked, 3.4, 640, 480, self.system)
        controller.update(mode_pose, self.unlocked, 4.21, 640, 480, self.system)
        self.assertEqual(controller.recognition_mode, "uppercase")
        self.assertIsNone(controller.writing_guide)
        self.assertIsNone(self.overlay.guide)


class WholeLineWritingControllerTests(unittest.TestCase):
    def setUp(self):
        self.overlay = MemoryInkOverlay(size=(1280, 800))
        self.recognizer = FakeLineRecognizer()
        self.controller = WholeLineWritingController(
            self.overlay, self.recognizer, idle_seconds=3.0
        )
        self.controller.mode = "AIR_WRITE"
        self.system = FakeSystemController()
        self.unlocked = {
            "LEFT": SimpleNamespace(locked=False),
            "RIGHT": SimpleNamespace(locked=False),
        }

    def test_complete_line_is_inserted_once_after_idle_timeout(self):
        first = make_hand(open_palm=True, pinched=True)
        second = make_hand(open_palm=True, pinched=True, dx=0.05)
        self.controller.update(
            {"RIGHT": (1.0, first, None)}, self.unlocked, 1.0, 640, 480, self.system
        )
        self.controller.update(
            {"RIGHT": (1.0, second, None)}, self.unlocked, 1.1, 640, 480, self.system
        )
        self.controller.update({}, self.unlocked, 3.9, 640, 480, self.system)
        self.assertEqual(self.system.committed_text, [])
        self.controller.update({}, self.unlocked, 4.11, 640, 480, self.system)

        self.assertEqual(self.recognizer.calls, 1)
        self.assertEqual(self.system.committed_text, ["Hi 2! "])
        self.assertEqual(self.overlay.preview, "Hi 2!")
        self.assertTrue(np.all(self.controller.canvas == 255))
        self.assertFalse(self.controller.has_line_ink)

        self.controller.update({}, self.unlocked, 8.0, 640, 480, self.system)
        self.assertEqual(self.system.committed_text, ["Hi 2! "])

    def test_left_undo_hold_removes_only_the_latest_stroke(self):
        first = make_hand(open_palm=True, pinched=True)
        first_end = make_hand(open_palm=True, pinched=True, dx=0.02)
        second = make_hand(open_palm=True, pinched=True, dx=0.12)
        second_end = make_hand(open_palm=True, pinched=True, dx=0.14)
        released = make_hand(open_palm=True, pinched=False)

        for timestamp, hand in (
            (1.0, first),
            (1.1, first_end),
            (1.2, released),
            (1.3, second),
            (1.4, second_end),
            (1.5, released),
        ):
            self.controller.update(
                {"RIGHT": (1.0, hand, None)},
                self.unlocked,
                timestamp,
                640,
                480,
                self.system,
            )

        undo = {"LEFT": (1.0, make_stroke_undo_hand(), None)}
        self.controller.update(undo, self.unlocked, 1.6, 640, 480, self.system)
        self.controller.update(undo, self.unlocked, 2.26, 640, 480, self.system)

        retained_strokes = [stroke for stroke in self.overlay.strokes if stroke]
        self.assertEqual(len(retained_strokes), 1)
        self.assertEqual(len(self.controller.canvas_strokes), 1)
        self.assertTrue(np.any(self.controller.canvas < 255))
        self.assertTrue(self.controller.has_line_ink)
        self.assertIn("LAST STROKE UNDONE", self.controller.recognition_feedback)

        # Holding the same pose cannot remove another stroke until it is released.
        self.controller.update(undo, self.unlocked, 3.0, 640, 480, self.system)
        self.assertEqual(len(self.controller.canvas_strokes), 1)

    def test_escape_blocks_automatic_insertion_and_keeps_line(self):
        self.controller.ensure_canvas(640, 480)
        self.controller.canvas[200:260, 200:500] = 0
        self.controller.has_line_ink = True
        self.controller.last_ink_at = 1.0
        self.controller.emergency_stop()
        self.controller.update({}, self.unlocked, 4.1, 640, 480, self.system)

        self.assertEqual(self.recognizer.calls, 0)
        self.assertEqual(self.system.committed_text, [])
        self.assertTrue(np.any(self.controller.canvas < 255))
        self.assertIn("BLOCKED BY ESC", self.controller.recognition_feedback)

    def test_low_confidence_line_is_previewed_but_not_inserted(self):
        self.controller.recognizer = FakeLineRecognizer("Hi C!", confidence=0.78)
        self.controller.ensure_canvas(640, 480)
        self.controller.canvas[200:260, 200:500] = 0
        self.controller.has_line_ink = True
        self.controller.last_ink_at = 1.0
        self.controller.update({}, self.unlocked, 4.1, 640, 480, self.system)

        self.assertEqual(self.system.committed_text, [])
        self.assertEqual(self.overlay.preview, "Hi C!")
        self.assertTrue(np.any(self.controller.canvas < 255))
        self.assertIn("LOW CONFIDENCE", self.controller.recognition_feedback)

    def test_high_confidence_gemini_correction_is_inserted(self):
        reviewer = FakeGeminiReviewer("Hi C!", confidence=0.95)
        controller = WholeLineWritingController(
            self.overlay,
            FakeLineRecognizer("Hi c!", confidence=0.72),
            idle_seconds=3.0,
            gemini_reviewer=reviewer,
        )
        controller.mode = "AIR_WRITE"
        controller.ensure_canvas(640, 480)
        controller.canvas[200:260, 200:500] = 0
        controller.has_line_ink = True
        controller.last_ink_at = 1.0

        controller.update({}, self.unlocked, 4.1, 640, 480, self.system)
        controller._gemini_future.result(timeout=1.0)
        controller._poll_gemini_review(self.system)

        self.assertEqual(self.system.committed_text, ["Hi C! "])
        self.assertEqual(self.overlay.preview, "Hi C!")
        self.assertEqual(len(reviewer.calls), 1)
        self.assertIn("CORRECTED AND INSERTED", controller.recognition_feedback)
        self.assertFalse(controller.has_line_ink)
        self.assertTrue(np.all(controller.canvas == 255))
        controller.close()

    def test_low_confidence_gemini_result_remains_preview_only(self):
        reviewer = FakeGeminiReviewer("Hi C!", confidence=0.72)
        controller = WholeLineWritingController(
            self.overlay,
            FakeLineRecognizer("Hi c!", confidence=0.70),
            idle_seconds=3.0,
            gemini_reviewer=reviewer,
        )
        controller.mode = "AIR_WRITE"
        controller.ensure_canvas(640, 480)
        controller.canvas[200:260, 200:500] = 0
        controller.has_line_ink = True
        controller.last_ink_at = 1.0

        controller.update({}, self.unlocked, 4.1, 640, 480, self.system)
        controller._gemini_future.result(timeout=1.0)
        controller._poll_gemini_review(self.system)

        self.assertEqual(self.system.committed_text, [])
        self.assertEqual(self.overlay.preview, "Hi C!")
        self.assertIn("PREVIEW ONLY", controller.recognition_feedback)
        self.assertTrue(controller.has_line_ink)
        controller.close()

    def test_escape_blocks_completed_gemini_insertion(self):
        reviewer = FakeGeminiReviewer("Hi C!", confidence=0.98)
        controller = WholeLineWritingController(
            self.overlay,
            FakeLineRecognizer("Hi c!", confidence=0.70),
            idle_seconds=3.0,
            gemini_reviewer=reviewer,
        )
        controller.mode = "AIR_WRITE"
        controller.ensure_canvas(640, 480)
        controller.canvas[200:260, 200:500] = 0
        controller.has_line_ink = True
        controller.last_ink_at = 1.0

        controller.update({}, self.unlocked, 4.1, 640, 480, self.system)
        controller._gemini_future.result(timeout=1.0)
        controller.emergency_stop()
        controller._poll_gemini_review(self.system)

        self.assertEqual(self.system.committed_text, [])
        self.assertTrue(controller.has_line_ink)
        self.assertIn("BLOCKED BY ESC", controller.recognition_feedback)
        controller.close()


if __name__ == "__main__":
    unittest.main()
