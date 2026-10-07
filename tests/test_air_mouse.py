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
        self.hotkeys = []
        self.writes = []
        self.button_events = []
        self.double_clicks = []
        self.raise_fail_safe = False

    def size(self):
        return 1920, 1080

    def moveTo(self, x, y, _pause):
        if self.raise_fail_safe:
            raise self.FailSafeException
        self.moves.append((x, y, _pause))

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

    def hotkey(self, *keys, _pause):
        self.hotkeys.append((keys, _pause))

    def write(self, text, interval, _pause):
        self.writes.append((text, interval, _pause))


class InvalidScreenBackend(FakeMouseBackend):
    def size(self):
        return 0, 0


class FakeDocumentSafety:
    def __init__(self):
        self.insertions = []
        self.closed = False

    def after_text_insert(self, text):
        self.insertions.append(text)

    def save_now(self, save_action):
        return save_action()

    def close(self):
        self.closed = True


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

    def test_disable_releases_held_mouse_button(self):
        self.controller.toggle()
        self.assertTrue(self.controller.mouse_down("left"))

        self.controller.disable("test")

        self.assertEqual(
            self.backend.button_events,
            [("down", "left", False), ("up", "left", False)],
        )
        self.assertFalse(self.controller.enabled)

    def test_corner_fail_safe_cannot_prevent_mouse_release(self):
        def guarded_release(button, _pause):
            if self.backend.FAILSAFE:
                raise self.backend.FailSafeException()
            self.backend.button_events.append(("up", button, _pause))

        self.controller.toggle()
        self.controller.mouse_down()
        self.backend.mouseUp = guarded_release
        self.controller.disable("corner stop")
        self.assertEqual(self.backend.button_events[-1], ("up", "left", False))
        self.assertTrue(self.backend.FAILSAFE)

    def test_moving_while_held_sends_drag_events_without_another_mouse_down(self):
        from unittest.mock import Mock
        self.backend.dragTo = Mock()
        self.controller.toggle()
        self.controller.mouse_down()
        self.controller.move_from_preview((999, 499), (1000, 500))
        self.backend.dragTo.assert_called_once_with(
            1919, 1079, duration=0, button="left", mouseDownUp=False, _pause=False
        )
        self.assertEqual(self.backend.moves, [])
        self.assertEqual(self.backend.button_events, [("down", "left", False)])

    def test_click_and_scroll_require_active_control(self):
        self.assertFalse(self.controller.click("left"))
        self.assertFalse(self.controller.scroll(3))
        self.assertFalse(self.controller.hotkey("command", "tab"))
        self.controller.toggle()
        self.assertTrue(self.controller.click("right"))
        self.assertTrue(self.controller.scroll(-3))
        self.assertTrue(self.controller.hotkey("command", "tab"))
        self.assertEqual(self.backend.clicks, [("right", False)])
        self.assertEqual(self.backend.scrolls, [(-3, False)])
        self.assertEqual(self.backend.hotkeys, [(('command', 'tab'), False)])

    def test_confirmed_text_commit_is_separate_from_mouse_toggle(self):
        self.assertFalse(self.controller.enabled)
        self.assertFalse(self.controller.commit_text(""))
        self.assertTrue(self.controller.commit_text("A"))
        self.assertEqual(self.backend.writes, [("A", 0.0, False)])

    def test_successful_text_commit_records_without_saving_focused_app(self):
        safety = FakeDocumentSafety()
        controller = SystemMouseController(self.backend, document_safety=safety)

        self.assertTrue(controller.commit_text("Recovered text "))
        self.assertEqual(safety.insertions, ["Recovered text "])
        self.assertEqual(self.backend.hotkeys, [])

    def test_manual_document_save_uses_command_s_and_closes_safety_manager(self):
        safety = FakeDocumentSafety()
        controller = SystemMouseController(self.backend, document_safety=safety)
        controller.toggle()

        self.assertTrue(controller.save_document())
        controller.close()

        self.assertEqual(self.backend.hotkeys, [(('command', 's'), False)])
        self.assertTrue(safety.closed)


if __name__ == "__main__":
    unittest.main()
