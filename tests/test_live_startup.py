"""The integrated launcher starts live after permissions and models pass."""

import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch

from airdesk.air_writing import main
from airdesk.native_overlay import MemoryInkOverlay
from tests.test_air_mouse import FakeMouseBackend


def test_integrated_launcher_enables_live_output_before_camera_loop():
    def run(**kwargs):
        controller = kwargs["system_mouse"]
        assert controller.enabled
        assert controller.primary_pinch_detector.blocked_until_release
        controller.close()
        kwargs["mode_controller"].close()
        return 0

    with patch.dict(sys.modules, {
        "pyautogui": FakeMouseBackend(),
        "Quartz": SimpleNamespace(CGPreflightPostEventAccess=lambda: True),
    }), patch("airdesk.air_writing.MODEL_PATH") as model, patch(
        "airdesk.air_writing.SegmentedCharacterRecognizer"
    ), patch("airdesk.air_writing.GeminiCorrectionClient.from_environment", return_value=None), patch(
        "airdesk.air_writing.MacEscapeMonitor", return_value=Mock()
    ), patch("airdesk.air_writing.NativeInkOverlay", return_value=MemoryInkOverlay()), patch(
        "airdesk.air_writing.NativeCommandPalette"
    ), patch("airdesk.air_writing.NativeControlPanel"), patch(
        "airdesk.air_writing.run_virtual_cursor", side_effect=run
    ):
        model.is_file.return_value = True
        model.stat.return_value.st_size = 100
        assert main() == 0
