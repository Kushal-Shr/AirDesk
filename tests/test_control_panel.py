import unittest

from airdesk.control_panel import (
    ControlPanelStatus,
    MemoryControlPanel,
    build_control_panel_status,
)


class ControlPanelTests(unittest.TestCase):
    def test_status_uses_safe_preview_and_locked_states(self):
        status = build_control_panel_status(
            system_enabled=False,
            paused=False,
            left_locked=True,
            right_locked=True,
            cursor_visible=False,
            left_action=None,
            right_action=None,
            fps=30.0,
            processing_ms=12.5,
        )

        self.assertEqual(status.mode, "STOPPED")
        self.assertEqual(status.left_hand, "LOCKED")
        self.assertEqual(status.right_hand, "LOCKED")

    def test_runtime_values_map_to_panel_vocabulary(self):
        status = build_control_panel_status(
            system_enabled=True,
            paused=False,
            left_locked=False,
            right_locked=False,
            cursor_visible=True,
            left_action="SCROLL",
            right_action="SWIPE",
            fps=45.0,
            processing_ms=8.0,
        )

        self.assertEqual(status.mode, "LIVE")
        self.assertEqual(status.left_hand, "SCROLL")
        self.assertEqual(status.right_hand, "COMMAND")

    def test_paused_mode_takes_priority(self):
        status = build_control_panel_status(
            system_enabled=True,
            paused=True,
            left_locked=False,
            right_locked=False,
            cursor_visible=True,
            left_action=None,
            right_action=None,
            fps=-1.0,
            processing_ms=-2.0,
        )

        self.assertEqual(status.mode, "PAUSED")
        self.assertEqual(status.left_hand, "POINTER")
        self.assertEqual(status.fps, 0.0)
        self.assertEqual(status.processing_ms, 0.0)

    def test_air_write_is_shown_as_an_overlay_on_live_output(self):
        status = build_control_panel_status(
            system_enabled=True,
            paused=False,
            left_locked=False,
            right_locked=False,
            cursor_visible=False,
            left_action=None,
            right_action=None,
            fps=30.0,
            processing_ms=10.0,
            writing_active=True,
        )

        self.assertEqual(status.mode, "LIVE · AIR WRITE")

    def test_memory_panel_keeps_latest_status(self):
        panel = MemoryControlPanel()
        expected = ControlPanelStatus(left_hand="READY", right_hand="READY")

        panel.update_status(expected)
        panel.close()

        self.assertEqual(panel.status, expected)
        self.assertFalse(panel.visible)

    def test_guide_can_be_hidden_and_shown_again(self):
        panel = MemoryControlPanel()

        panel.toggle_visibility()
        self.assertFalse(panel.visible)
        panel.toggle_visibility()
        self.assertTrue(panel.visible)


if __name__ == "__main__":
    unittest.main()
