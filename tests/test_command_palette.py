import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from airdesk.command_palette import MemoryCommandPalette, PALETTE_COMMANDS
from airdesk.desktop_controls import DesktopControlController
from tests.test_desktop_controls import FakeBackend, FakeEscapeMonitor, make_hand


class EnabledConfig:
    def get(self, _key):
        return True


def primary_pinch_hand():
    hand = make_hand(True, True, False, False)
    hand[4] = SimpleNamespace(x=0.39, y=0.21, z=0.0)
    return hand


def palette_hand():
    return make_hand(True, True, False, False)


class CommandPaletteTests(unittest.TestCase):
    def test_hit_test_maps_pointer_to_command_row(self):
        palette = MemoryCommandPalette()
        palette.show({command.key for command in PALETTE_COMMANDS})

        self.assertEqual(palette.hit_test((900, 360)), 0)
        self.assertEqual(palette.hit_test((900, 396)), 1)
        self.assertIsNone(palette.hit_test((400, 360)))

    def test_two_finger_hold_opens_and_pinch_selects_highlight(self):
        backend = FakeBackend()
        backend.hotkeys = []
        backend.hotkey = lambda *keys, _pause: backend.hotkeys.append((keys, _pause))
        palette = MemoryCommandPalette()
        controller = DesktopControlController(
            backend,
            FakeEscapeMonitor(),
            feature_config=EnabledConfig(),
            command_palette=palette,
        )
        controller.toggle()
        unlocked = SimpleNamespace(locked=False)
        locked = SimpleNamespace(locked=True)
        locks = {"LEFT": locked, "RIGHT": unlocked}
        two_fingers = palette_hand()

        controller.update_gestures(
            {"RIGHT": (1.0, make_hand(True, True, True, True), None)}, locks, 0.80
        )
        controller.update_gestures(
            {"RIGHT": (1.0, two_fingers, None)}, locks, 1.00
        )
        controller.update_gestures(
            {"RIGHT": (1.0, two_fingers, None)}, locks, 1.51
        )
        self.assertTrue(palette.visible)

        pinch = primary_pinch_hand()
        cursor = (300, 160)
        controller.update_gestures(
            {"RIGHT": (1.0, pinch, None)},
            locks,
            1.60,
            cursor_position=cursor,
            preview_size=(640, 480),
        )
        controller.update_gestures(
            {"RIGHT": (1.0, pinch, None)},
            locks,
            1.71,
            cursor_position=cursor,
            preview_size=(640, 480),
        )

        self.assertEqual(backend.hotkeys, [(PALETTE_COMMANDS[0].shortcut, False)])
        self.assertFalse(palette.visible)

    def test_right_lock_closes_palette_without_action(self):
        palette = MemoryCommandPalette()
        palette.show({command.key for command in PALETTE_COMMANDS})
        controller = DesktopControlController(
            FakeBackend(),
            FakeEscapeMonitor(),
            feature_config=EnabledConfig(),
            command_palette=palette,
        )
        locked = SimpleNamespace(locked=True)

        controller.update_gestures({}, {"LEFT": locked, "RIGHT": locked}, 2.0)

        self.assertFalse(palette.visible)

    def test_save_document_command_uses_controller_save_path(self):
        backend = FakeBackend()
        backend.hotkeys = []
        backend.hotkey = lambda *keys, _pause: backend.hotkeys.append((keys, _pause))
        palette = MemoryCommandPalette()
        controller = DesktopControlController(
            backend,
            FakeEscapeMonitor(),
            feature_config=EnabledConfig(),
            command_palette=palette,
        )
        controller.toggle()
        controller.save_document = Mock(return_value=True)
        save_index = next(
            index
            for index, command in enumerate(PALETTE_COMMANDS)
            if command.key == "save_document"
        )
        palette.show({command.key for command in PALETTE_COMMANDS})
        row_y = palette.bounds[1] + 54 + save_index * 36 + 18
        cursor = (
            round(900 / 1919 * 639),
            round(row_y / 1079 * 479),
        )
        pinch = primary_pinch_hand()
        locks = {
            "LEFT": SimpleNamespace(locked=True),
            "RIGHT": SimpleNamespace(locked=False),
        }
        controller.update_gestures(
            {"RIGHT": (1.0, make_hand(True, True, True, True), None)},
            locks,
            0.9,
            cursor_position=cursor,
            preview_size=(640, 480),
        )
        controller.update_gestures(
            {"RIGHT": (1.0, pinch, None)},
            locks,
            1.0,
            cursor_position=cursor,
            preview_size=(640, 480),
        )
        controller.update_gestures(
            {"RIGHT": (1.0, pinch, None)},
            locks,
            1.1,
            cursor_position=cursor,
            preview_size=(640, 480),
        )

        command = PALETTE_COMMANDS[save_index]
        self.assertEqual(command.shortcut, ("command", "s"))
        controller.save_document.assert_called_once_with()
        self.assertEqual(backend.hotkeys, [])
        self.assertFalse(palette.visible)

    def test_last_palette_row_is_inside_default_memory_bounds(self):
        palette = MemoryCommandPalette()
        palette.show({command.key for command in PALETTE_COMMANDS})
        left, top, right, _bottom = palette.bounds
        last_y = top + 54 + (len(PALETTE_COMMANDS) - 1) * 36 + 18

        self.assertEqual(
            palette.hit_test(((left + right) // 2, last_y)),
            len(PALETTE_COMMANDS) - 1,
        )

    def test_opening_palette_releases_drag_before_dwell_completes(self):
        backend = FakeBackend()
        controller = DesktopControlController(
            backend, FakeEscapeMonitor(), command_palette=MemoryCommandPalette()
        )
        controller.toggle()
        locks = {"LEFT": SimpleNamespace(locked=True), "RIGHT": SimpleNamespace(locked=False)}
        for now, hand in (
            (0.5, make_hand(True, True, True, True)), (1.0, primary_pinch_hand()),
            (1.6, primary_pinch_hand()), (1.7, palette_hand()),
        ):
            controller.update_gestures({"RIGHT": (1.0, hand, None)}, locks, now)
        self.assertEqual(backend.button_events[-1], ("up", "left", False))
        self.assertFalse(controller.primary_pinch_detector.dragging)

    def test_releasing_pinch_to_open_palm_still_completes_single_click(self):
        backend = FakeBackend()
        controller = DesktopControlController(
            backend, FakeEscapeMonitor(), command_palette=MemoryCommandPalette()
        )
        controller.toggle()
        locks = {"LEFT": SimpleNamespace(locked=True), "RIGHT": SimpleNamespace(locked=False)}
        for now, hand in (
            (0.5, make_hand(True, True, True, True)), (1.0, primary_pinch_hand()),
            (1.1, make_hand(True, True, True, True)),
            (1.65, make_hand(True, True, True, True)),
        ):
            controller.update_gestures({"RIGHT": (1.0, hand, None)}, locks, now)
        self.assertEqual(backend.clicks, [("left", False)])

    def test_four_finger_open_hand_does_not_open_palette(self):
        palette = MemoryCommandPalette()
        controller = DesktopControlController(
            FakeBackend(), FakeEscapeMonitor(), command_palette=palette
        )
        controller.toggle()
        locks = {
            "LEFT": SimpleNamespace(locked=True),
            "RIGHT": SimpleNamespace(locked=False),
        }
        open_hand = make_hand(True, True, True, True)

        controller.update_gestures(
            {"RIGHT": (1.0, open_hand, None)}, locks, 0.5
        )
        controller.update_gestures(
            {"RIGHT": (1.0, open_hand, None)}, locks, 1.0
        )
        controller.update_gestures(
            {"RIGHT": (1.0, open_hand, None)}, locks, 2.0
        )

        self.assertFalse(palette.visible)


if __name__ == "__main__":
    unittest.main()
