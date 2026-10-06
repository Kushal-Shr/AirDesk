import unittest
from types import SimpleNamespace

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


class CommandPaletteTests(unittest.TestCase):
    def test_hit_test_maps_pointer_to_command_row(self):
        palette = MemoryCommandPalette()
        palette.show({command.key for command in PALETTE_COMMANDS})

        self.assertEqual(palette.hit_test((900, 360)), 0)
        self.assertEqual(palette.hit_test((900, 396)), 1)
        self.assertIsNone(palette.hit_test((400, 360)))

    def test_four_finger_hold_opens_and_pinch_selects_highlight(self):
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
        four_fingers = make_hand(True, True, True, True)

        controller.update_gestures(
            {"RIGHT": (1.0, make_hand(), None)}, locks, 0.80
        )
        controller.update_gestures(
            {"RIGHT": (1.0, four_fingers, None)}, locks, 1.00
        )
        controller.update_gestures(
            {"RIGHT": (1.0, four_fingers, None)}, locks, 1.51
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


if __name__ == "__main__":
    unittest.main()
