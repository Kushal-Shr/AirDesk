"""Stage 2: detect and draw MediaPipe's 21 hand landmarks."""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision


WINDOW_NAME = "AirDesk - Hand Landmarks"
CAMERA_INDEX = 0
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "hand_landmarker.task"

# Landmark pairs that form the palm and five fingers.
HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)

IMPORTANT_LANDMARKS = {
    0: "0 wrist",
    4: "4 thumb",
    8: "8 index",
    12: "12 middle",
}


def normalized_to_pixel(landmark, frame_width: int, frame_height: int) -> tuple[int, int]:
    """Convert a normalized landmark position into safe frame coordinates."""
    x = min(max(int(landmark.x * frame_width), 0), frame_width - 1)
    y = min(max(int(landmark.y * frame_height), 0), frame_height - 1)
    return x, y


def draw_hand(frame, landmarks, hand_name: str, text_row: int) -> None:
    """Draw one hand skeleton and its index-tip coordinate readout."""
    frame_height, frame_width = frame.shape[:2]
    points = [
        normalized_to_pixel(landmark, frame_width, frame_height)
        for landmark in landmarks
    ]

    for start_id, end_id in HAND_CONNECTIONS:
        cv2.line(frame, points[start_id], points[end_id], (255, 180, 0), 2, cv2.LINE_AA)

    for landmark_id, point in enumerate(points):
        color = (0, 255, 0) if landmark_id == 8 else (0, 120, 255)
        radius = 7 if landmark_id == 8 else 4
        cv2.circle(frame, point, radius, color, -1, cv2.LINE_AA)

    for landmark_id, label in IMPORTANT_LANDMARKS.items():
        x, y = points[landmark_id]
        cv2.putText(
            frame,
            label,
            (x + 7, y - 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    index_tip = landmarks[8]
    index_x, index_y = points[8]
    coordinate_text = (
        f"{hand_name} index: pixel=({index_x}, {index_y})  "
        f"normalized=({index_tip.x:.3f}, {index_tip.y:.3f})"
    )
    cv2.putText(
        frame,
        coordinate_text,
        (20, text_row),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )


def draw_header(frame, fps: float, hands_detected: int) -> None:
    cv2.putText(
        frame,
        f"FPS: {fps:.1f}  Hands: {hands_detected}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "SAFE PREVIEW - Press Q to quit",
        (20, 68),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def create_landmarker() -> vision.HandLandmarker:
    options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path=str(MODEL_PATH),
            delegate=BaseOptions.Delegate.CPU,
        ),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.HandLandmarker.create_from_options(options)


def main() -> int:
    if not MODEL_PATH.exists():
        print("The Hand Landmarker model is missing.")
        print("Run: python scripts/download_hand_model.py")
        return 1

    camera = cv2.VideoCapture(CAMERA_INDEX)
    if not camera.isOpened():
        print(
            "AirDesk could not open the camera. Check macOS Camera permission "
            "for the app running Python, then try again."
        )
        camera.release()
        return 1

    start_time = time.perf_counter()
    previous_time = start_time
    previous_timestamp_ms = -1
    smoothed_fps = 0.0

    try:
        with create_landmarker() as landmarker:
            while True:
                frame_received, frame = camera.read()
                if not frame_received:
                    print("AirDesk stopped because it could not read a camera frame.")
                    return 1

                frame = cv2.flip(frame, 1)

                # OpenCV supplies BGR pixels. MediaPipe expects SRGB/RGB pixels.
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                media_pipe_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb_frame,
                )

                timestamp_ms = int((time.perf_counter() - start_time) * 1000)
                timestamp_ms = max(timestamp_ms, previous_timestamp_ms + 1)
                previous_timestamp_ms = timestamp_ms
                result = landmarker.detect_for_video(media_pipe_image, timestamp_ms)

                current_time = time.perf_counter()
                frame_seconds = current_time - previous_time
                previous_time = current_time
                instant_fps = 1.0 / frame_seconds if frame_seconds > 0 else 0.0
                smoothed_fps = (
                    instant_fps
                    if smoothed_fps == 0.0
                    else (0.9 * smoothed_fps) + (0.1 * instant_fps)
                )

                for hand_index, landmarks in enumerate(result.hand_landmarks):
                    hand_name = "Hand"
                    if hand_index < len(result.handedness) and result.handedness[hand_index]:
                        hand_name = result.handedness[hand_index][0].category_name
                    draw_hand(frame, landmarks, hand_name, 105 + hand_index * 30)

                draw_header(frame, smoothed_fps, len(result.hand_landmarks))
                cv2.imshow(WINDOW_NAME, frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q")):
                    break
                if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    break
    finally:
        camera.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
