"""Stage 1: display a safe, mirrored OpenCV webcam preview."""

from __future__ import annotations

import time

import cv2


WINDOW_NAME = "AirDesk - Safe Webcam Preview"
CAMERA_INDEX = 0


def draw_status(frame, fps: float) -> None:
    """Draw the Stage 1 status information on a webcam frame."""
    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "SAFE PREVIEW - Press Q to quit",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def main() -> int:
    """Open the default camera and show frames until Q is pressed."""
    camera = cv2.VideoCapture(CAMERA_INDEX)

    if not camera.isOpened():
        print(
            "AirDesk could not open the camera. Check macOS Camera permission "
            "for the app running Python, then try again."
        )
        camera.release()
        return 1

    previous_time = time.perf_counter()
    smoothed_fps = 0.0

    try:
        while True:
            frame_received, frame = camera.read()
            if not frame_received:
                print("AirDesk stopped because it could not read a camera frame.")
                return 1

            # A webcam frame is a NumPy array of BGR pixels. Flipping around the
            # vertical axis makes the preview behave like a familiar mirror.
            frame = cv2.flip(frame, 1)

            current_time = time.perf_counter()
            frame_seconds = current_time - previous_time
            previous_time = current_time
            instant_fps = 1.0 / frame_seconds if frame_seconds > 0 else 0.0

            # Blend measurements so the displayed number is easier to read.
            smoothed_fps = (
                instant_fps
                if smoothed_fps == 0.0
                else (0.9 * smoothed_fps) + (0.1 * instant_fps)
            )

            draw_status(frame, smoothed_fps)
            cv2.imshow(WINDOW_NAME, frame)

            # waitKey lets OpenCV process window events and returns a key code.
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q")):
                break

            # Also finish cleanly if the preview window's close button is used.
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        # Always release the hardware and close OpenCV windows, even on error.
        camera.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
