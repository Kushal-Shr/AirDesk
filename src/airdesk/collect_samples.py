"""Launcher for collecting labeled personal air-writing samples."""

from __future__ import annotations

import argparse
from pathlib import Path

from .native_overlay import NativeInkOverlay
from .personal_samples import (
    LABEL_GROUPS,
    PERSONAL_DATA_PATH,
    PersonalSampleController,
    PersonalSampleStore,
)
from .virtual_cursor import run_virtual_cursor


WINDOW_NAME = "AirDesk - Personal Training"


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--group",
        choices=tuple(LABEL_GROUPS),
        default="lowercase",
        help="character group to collect (default: lowercase)",
    )
    parser.add_argument(
        "--samples-per-character",
        type=int,
        default=5,
        help="target saved samples for every character (default: 5)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PERSONAL_DATA_PATH,
    )
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    if arguments.samples_per_character < 1:
        print("--samples-per-character must be at least 1")
        return 2
    try:
        overlay = NativeInkOverlay()
        store = PersonalSampleStore(
            root=arguments.data_dir,
            group=arguments.group,
            labels=LABEL_GROUPS[arguments.group],
            samples_per_character=arguments.samples_per_character,
        )
    except Exception as error:
        print(f"AirDesk could not start Personal Training Mode: {error}")
        return 1

    if store.current_label is None:
        print("This collection target is already complete.")
        overlay.close()
        return 0
    print(f"Collecting {arguments.group} samples in {arguments.data_dir}.")
    print("Hold both open palms for one second to enter Personal Training Mode.")
    print("Draw the prompted character inside the guide, then hold thumbs-up to save.")
    print("A left open-palm hold clears the current attempt. Q quits in camera mode.")
    return run_virtual_cursor(
        system_mouse=None,
        window_name=WINDOW_NAME,
        mode_controller=PersonalSampleController(overlay, store),
    )


if __name__ == "__main__":
    raise SystemExit(main())
