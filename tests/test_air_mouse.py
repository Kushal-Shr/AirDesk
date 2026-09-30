"""Tests for Stage 5 that never access the real macOS mouse."""

from __future__ import annotations

import unittest

from airdesk.air_mouse import SystemMouseController, map_preview_to_screen


class FakeMouseBackend:
    class FailSafeException(Exception):
        pass

    FAILSAFE = False
    PAUSE = 1.0

    def __init__(self):
        self.moves = []
        self.clicks = []
        self.scrolls = []
        self.raise_fail_safe = False

    def size(self):
        return 1920, 1080

    def moveTo(self, x, y, _pause):
        if self.raise_fail_safe:
            raise self.FailSafeException
        self.moves.append((x, y, _pause))

    def click(self, button, _pause):
        self.clicks.append((button, _pause))

    def scroll(self, steps, _pause):
        self.scrolls.append((steps, _pause))


class InvalidScreenBackend(FakeMouseBackend):
    def size(self):
        return 0, 0


class ScreenMappingTests(unittest.TestCase):
    def test_preview_edges_map_to_screen_edges(self):
        self.assertEqual(
            map_preview_to_screen((0, 0), (1000, 500), (1920, 1080)),
            (0, 0),
        )
        self.assertEqual(
            map_preview_to_screen((999, 499), (1000, 500), (1920, 1080)),
            (1919, 1079),
        )

    def test_outside_coordinates_are_clamped(self):
        self.assertEqual(
            map_preview_to_screen((-20, 800), (1000, 500), (1920, 1080)),
            (0, 1079),
        )


class SystemMouseControllerTests(unittest.TestCase):
    def setUp(self):
        self.backend = FakeMouseBackend()
        self.controller = SystemMouseController(self.backend)

    def test_starts_disabled_and_does_not_move(self):
        self.assertFalse(self.controller.enabled)
        self.assertFalse(self.controller.move_from_preview((500, 250), (1000, 500)))
        self.assertEqual(self.backend.moves, [])

    def test_invalid_screen_size_refuses_to_start(self):
        with self.assertRaises(RuntimeError):
            SystemMouseController(InvalidScreenBackend())

    def test_toggle_enables_movement(self):
        self.controller.toggle()
        self.assertTrue(self.controller.move_from_preview((999, 499), (1000, 500)))
        self.assertEqual(self.backend.moves, [(1919, 1079, False)])

    def test_fail_safe_disables_output(self):
        self.controller.toggle()
        self.backend.raise_fail_safe = True
        self.assertFalse(self.controller.move_from_preview((500, 250), (1000, 500)))
        self.assertFalse(self.controller.enabled)

    def test_explicit_disable_stops_output(self):
        self.controller.toggle()
        self.controller.disable("test")
        self.assertFalse(self.controller.enabled)

    def test_click_and_scroll_require_active_control(self):
        self.assertFalse(self.controller.click("left"))
        self.assertFalse(self.controller.scroll(3))
        self.controller.toggle()
        self.assertTrue(self.controller.click("right"))
        self.assertTrue(self.controller.scroll(-3))
        self.assertEqual(self.backend.clicks, [("right", False)])
        self.assertEqual(self.backend.scrolls, [(-3, False)])


if __name__ == "__main__":
    unittest.main()
