"""Download the official MediaPipe Hand Landmarker model for AirDesk."""

from __future__ import annotations

import subprocess
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen


MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "hand_landmarker.task"


def main() -> int:
    if MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 0:
        print(f"Hand model already exists: {MODEL_PATH}")
        return 0

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = MODEL_PATH.with_suffix(".task.download")

    print("Downloading the official MediaPipe Hand Landmarker model...")
    try:
        with urlopen(MODEL_URL, timeout=60) as response:
            temporary_path.write_bytes(response.read())
    except (OSError, URLError) as error:
        print(f"Python download failed ({error}). Trying macOS curl...")
        try:
            subprocess.run(
                [
                    "curl",
                    "--fail",
                    "--location",
                    "--output",
                    str(temporary_path),
                    MODEL_URL,
                ],
                check=True,
            )
        except (OSError, subprocess.CalledProcessError) as curl_error:
            temporary_path.unlink(missing_ok=True)
            print(f"Download failed: {curl_error}")
            return 1

    if not temporary_path.exists() or temporary_path.stat().st_size == 0:
        temporary_path.unlink(missing_ok=True)
        print("Download failed: the model file was empty.")
        return 1

    temporary_path.replace(MODEL_PATH)

    print(f"Saved model to: {MODEL_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
