"""Stage 8 launcher for transparent desktop air-writing."""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

from .desktop_controls import MacEscapeMonitor
from .gemini_correction import GeminiCorrectionClient
from .native_overlay import NativeInkOverlay
from .segmented_recognition import SegmentedCharacterRecognizer
from .system_shortcuts import ShortcutControlController
from .virtual_cursor import run_virtual_cursor
from .whiteboard import WholeLineWritingController


WINDOW_NAME = "AirDesk - Desktop Controls"
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    escape_monitor = MacEscapeMonitor()
    overlay = None
    try:
        recognizer = SegmentedCharacterRecognizer()
        recognizer.verify_ready()
        gemini_reviewer = GeminiCorrectionClient.from_environment()
        if gemini_reviewer is not None:
            gemini_reviewer.verify_ready()
        import pyautogui
        import Quartz

        if not Quartz.CGPreflightPostEventAccess():
            raise RuntimeError(
                "macOS is blocking system input for this launcher; enable its "
                "Accessibility permission and restart it"
            )
        escape_monitor.start()
        system_controller = ShortcutControlController(pyautogui, escape_monitor)
        overlay = NativeInkOverlay()
    except Exception as error:
        escape_monitor.close()
        if overlay is not None:
            overlay.close()
        print(f"AirDesk could not start Air Write mode: {error}")
        return 1

    print("AirDesk starts in DESKTOP mode with real output OFF.")
    print("Recognition mode: SEGMENTED CHARACTERS (A–Z, a–z, 0–9, symbols).")
    print("The full line is split and every character is classified independently.")
    print("A finished line is inserted automatically after three seconds without ink.")
    if gemini_reviewer is None:
        print("Gemini review: OFF (set GEMINI_API_KEY to enable safe preview review).")
    else:
        print(
            f"Gemini review: ON ({gemini_reviewer.model}); low-confidence results "
            "are previewed, never auto-inserted."
        )
    print(
        "Undo: hold a LEFT thumb-index pinch with middle, ring, and little up."
    )
    print("Hold both open palms for one second to enter or leave AIR WRITE mode.")
    print("The desktop overlay is click-through and does not activate itself.")
    return run_virtual_cursor(
        system_mouse=system_controller,
        window_name=WINDOW_NAME,
        mode_controller=WholeLineWritingController(
            overlay,
            recognizer,
            gemini_reviewer=gemini_reviewer,
        ),
    )


if __name__ == "__main__":
    raise SystemExit(main())
