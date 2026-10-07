"""Demo preflight. Default checks never open the camera or send system input."""

from __future__ import annotations

import argparse
from pathlib import Path
import time


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", action="store_true", help="read 20 live frames; do not save them")
    parser.add_argument("--native", action="store_true", help="briefly open and close native windows")
    parser.add_argument("--gemini", action="store_true", help="send a synthetic Hi 2! image to Gemini")
    args = parser.parse_args()
    failures = []

    def check(label, action):
        try:
            detail = action()
            print(f"PASS {label}" + (f": {detail}" if detail else ""), flush=True)
        except Exception as error:
            failures.append(label)
            # Remote exceptions can contain request details; never echo keys.
            print(f"FAIL {label}: {type(error).__name__}", flush=True)

    def handwriting():
        from .segmented_recognition import SegmentedCharacterRecognizer, MERGED_CLASSES
        recognizer = SegmentedCharacterRecognizer()
        recognizer.verify_ready()
        return f"validated checkpoint loaded ({len(MERGED_CLASSES)} classes)"

    def tracking():
        import mediapipe as mp
        import numpy as np
        from .hand_landmarks import create_landmarker
        with create_landmarker() as detector:
            detector.detect_for_video(mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=np.zeros((480, 640, 3), dtype=np.uint8),
            ), 0)
        return "model loaded and inference completed"

    def accessibility():
        import Quartz
        if not Quartz.CGPreflightPostEventAccess():
            raise RuntimeError("Accessibility permission missing")
        from .desktop_controls import MacEscapeMonitor
        monitor = MacEscapeMonitor()
        try:
            monitor.start()
        finally:
            monitor.close()
        return "system input permission and global Escape monitor available"

    def camera():
        import cv2
        import mediapipe as mp
        from .hand_landmarks import CAMERA_INDEX, create_landmarker
        from .virtual_cursor import configure_camera, limit_frame_resolution
        device = cv2.VideoCapture(CAMERA_INDEX)
        try:
            if not device.isOpened():
                raise RuntimeError("Camera permission missing or device busy")
            configure_camera(device)
            with create_landmarker() as detector:
                start = time.perf_counter()
                for index in range(20):
                    received, frame = device.read()
                    if not received:
                        raise RuntimeError("Camera stopped returning frames")
                    frame = limit_frame_resolution(cv2.flip(frame, 1))
                    detector.detect_for_video(mp.Image(
                        image_format=mp.ImageFormat.SRGB,
                        data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                    ), index * 100)
                elapsed = time.perf_counter() - start
            return f"20 frames tracked at {20 / elapsed:.1f} FPS (no recordings saved)"
        finally:
            device.release()

    def native():
        from .native_overlay import NativeInkOverlay
        from .control_panel import NativeControlPanel
        from .command_palette import NativeCommandPalette
        resources = []
        try:
            for factory in (NativeInkOverlay, NativeCommandPalette, NativeControlPanel):
                resources.append(factory())
            resources[0].show()
            resources[0].add_point((200, 200))
            resources[1].show(set())
            resources[-1].pump()
            return "ink overlay, command palette, and gesture guide constructed and rendered"
        finally:
            for resource in reversed(resources):
                resource.close()

    def gemini():
        import cv2
        import numpy as np
        from dotenv import load_dotenv
        from .gemini_correction import GeminiCorrectionClient
        load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=False)
        reviewer = GeminiCorrectionClient.from_environment()
        if reviewer is None:
            raise RuntimeError("GEMINI_API_KEY not configured")
        canvas = np.full((180, 460, 3), 255, dtype=np.uint8)
        cv2.putText(canvas, "Hi 2!", (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 0, 0), 4)
        result = reviewer.review(canvas, "Hi 2!", ())
        return f"{reviewer.last_model_used}: {result.text!r}, confidence {result.confidence:.0%}"

    check("Handwriting model", handwriting)
    check("Hand tracking model", tracking)
    check("Accessibility / Escape", accessibility)
    if args.camera:
        check("Live camera", camera)
    if args.native:
        check("Native windows", native)
    if args.gemini:
        check("Gemini connection", gemini)
    print("Preflight complete." if not failures else "Fix failed checks before the demo.")
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
