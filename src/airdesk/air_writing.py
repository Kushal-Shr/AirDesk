"""Stage 8 launcher for transparent desktop air-writing."""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

from .command_palette import NativeCommandPalette
from .control_panel import NativeControlPanel
from .desktop_controls import MacEscapeMonitor
from .document_safety import DocumentSafetyManager
from .gemini_correction import GeminiCorrectionClient
from .hand_landmarks import MODEL_PATH
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
    control_panel = None
    command_palette = None
    system_controller = None
    document_safety = DocumentSafetyManager()
    try:
        if not MODEL_PATH.is_file() or MODEL_PATH.stat().st_size == 0:
            raise RuntimeError(
                "hand tracking model missing; run: python scripts/download_hand_model.py"
            )
        recognizer = SegmentedCharacterRecognizer()
        recognizer.verify_ready()
        gemini_reviewer = GeminiCorrectionClient.from_environment()
        if gemini_reviewer is not None:
            try:
                gemini_reviewer.verify_ready()
            except Exception as error:
                print(
                    "Gemini unavailable; continuing with local recognition: "
                    f"{type(error).__name__}"
                )
                gemini_reviewer = None
        import pyautogui
        import Quartz

        if not Quartz.CGPreflightPostEventAccess():
            raise RuntimeError(
                "macOS is blocking system input for this launcher; enable its "
                "Accessibility permission and restart it"
            )
        escape_monitor.start()
        overlay = NativeInkOverlay()
        command_palette = NativeCommandPalette()
        control_panel = NativeControlPanel()
        system_controller = ShortcutControlController(
            pyautogui,
            escape_monitor,
            command_palette=command_palette,
            document_safety=document_safety,
        )
        system_controller.toggle()
    except Exception as error:
        if system_controller is not None:
            system_controller.close()
        else:
            escape_monitor.close()
            if command_palette is not None:
                command_palette.close()
        if overlay is not None:
            overlay.close()
        if control_panel is not None:
            control_panel.close()
        print(f"AirDesk could not start Air Write mode: {error}")
        return 1

    print("AirDesk starts LIVE in DESKTOP mode with real mouse/keyboard output ON.")
    print("Recognition mode: SEGMENTED CHARACTERS (A–Z, a–z, 0–9, symbols).")
    print("The full line is split and every character is classified independently.")
    print("A finished line is inserted automatically after two seconds without ink.")
    if gemini_reviewer is None:
        print("Gemini review: OFF (set GEMINI_API_KEY to enable correction).")
    else:
        print(
            f"Gemini review: ON ({gemini_reviewer.model}); low-confidence results "
            "are auto-inserted when Gemini confidence is at least 90%."
        )
    print(
        "Undo: hold a LEFT thumb-index pinch with middle, ring, and little up."
    )
    print("Hold both open palms briefly to enter or leave AIR WRITE mode.")
    print("The desktop overlay is click-through and does not activate itself.")
    print("Desktop controls are live. M toggles output; P hides or shows the guide.")
    print(
        "Esc stops output and returns to the preview. "
        "Q in preview or Ctrl+C in Terminal quits."
    )
    print("Air Write text insertion is armed separately when you enter with both palms.")
    print(
        "Inserted text is recovery-logged locally and the focused document "
        "is auto-saved after one second."
    )
    print(f"Recovery history: {document_safety.recovery_path}")
    print("Right thumbs-up recognizes now or retries a retained line.")
    print("Enter: hold only the RIGHT index finger up briefly; keep the others folded.")
    print(
        "Closing Air Write saves pending ink first, then safely rearms LIVE controls."
    )
    return run_virtual_cursor(
        system_mouse=system_controller,
        window_name=WINDOW_NAME,
        mode_controller=WholeLineWritingController(
            overlay,
            recognizer,
            gemini_reviewer=gemini_reviewer,
        ),
        control_panel=control_panel,
    )


if __name__ == "__main__":
    raise SystemExit(main())
