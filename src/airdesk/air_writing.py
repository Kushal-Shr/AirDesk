"""Stage 8 launcher for transparent desktop air-writing."""

from __future__ import annotations

import argparse

from .character_recognition import PersonalCharacterRecognizer, RECOGNITION_CLASSES
from .desktop_controls import MacEscapeMonitor
from .native_overlay import NativeInkOverlay
from .system_shortcuts import ShortcutControlController
from .virtual_cursor import run_virtual_cursor
from .whiteboard import AirWritingController


WINDOW_NAME = "AirDesk - Desktop Controls"


def parse_arguments(arguments=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--recognition-mode",
        choices=tuple(RECOGNITION_CLASSES),
        default="uppercase",
        help="character set to recognize (default: uppercase)",
    )
    return parser.parse_args(arguments)


def main() -> int:
    arguments = parse_arguments()
    escape_monitor = MacEscapeMonitor()
    overlay = None
    try:
        recognizer = PersonalCharacterRecognizer(arguments.recognition_mode)
        recognizer.verify_ready()
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
    print(f"Recognition mode: {arguments.recognition_mode.upper()}.")
    print("Hold both open palms for one second to enter or leave AIR WRITE mode.")
    print("The desktop overlay is click-through and does not activate itself.")
    return run_virtual_cursor(
        system_mouse=system_controller,
        window_name=WINDOW_NAME,
        mode_controller=AirWritingController(overlay, recognizer),
    )


if __name__ == "__main__":
    raise SystemExit(main())
