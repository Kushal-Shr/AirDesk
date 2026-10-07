"""Stage 3: classify closed fists and maintain safe per-hand locks."""

from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import mediapipe as mp
import numpy as np

from .hand_landmarks import (
    CAMERA_INDEX,
    MODEL_PATH,
    create_landmarker,
    draw_hand,
    normalized_to_pixel,
)


WINDOW_NAME = "AirDesk - Hand Locks"
UNLOCK_DELAY_SECONDS = 0.12

# Each tuple contains a finger's MCP joint, PIP joint, and fingertip.
FINGER_JOINTS = (
    (5, 6, 8),    # Index
    (9, 10, 12),  # Middle
    (13, 14, 16), # Ring
    (17, 18, 20), # Little
)


@dataclass(frozen=True)
class FistMetrics:
    """Measurements used to explain one fist-classification result."""

    is_closed: bool
    curled_fingers: int
    compactness: float


@dataclass
class HandLockState:
    """Lock immediately, but require a stable open hand before unlocking."""

    locked: bool = True
    unlock_started_at: float | None = None

    def update(self, fist_detected: bool, hand_seen: bool, now: float) -> None:
        if not hand_seen or fist_detected:
            self.locked = True
            self.unlock_started_at = None
            return

        if not self.locked:
            return

        if self.unlock_started_at is None:
            self.unlock_started_at = now
        elif now - self.unlock_started_at >= UNLOCK_DELAY_SECONDS:
            self.locked = False
            self.unlock_started_at = None


def joint_angle_degrees(first: np.ndarray, vertex: np.ndarray, last: np.ndarray) -> float:
    """Return the smaller angle formed by three points."""
    first_vector = first - vertex
    last_vector = last - vertex
    denominator = np.linalg.norm(first_vector) * np.linalg.norm(last_vector)
    if denominator < 1e-8:
        return 180.0

    cosine = np.clip(np.dot(first_vector, last_vector) / denominator, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def finger_is_curled(
    points: np.ndarray,
    wrist: np.ndarray,
    mcp_id: int,
    pip_id: int,
    tip_id: int,
) -> bool:
    """Return whether one non-thumb finger is folded toward the palm."""
    joint_angle = joint_angle_degrees(points[mcp_id], points[pip_id], points[tip_id])
    tip_to_wrist = np.linalg.norm(points[tip_id] - wrist)
    pip_to_wrist = np.linalg.norm(points[pip_id] - wrist)
    return bool(joint_angle < 125.0 or tip_to_wrist < pip_to_wrist * 1.08)


def thumb_is_raised(points: np.ndarray) -> bool:
    """Detect a straight thumb pointing upward, independent of hand side."""
    wrist = points[0]
    palm_size = max(float(np.linalg.norm(points[9] - wrist)), 1e-6)
    thumb_angle = joint_angle_degrees(points[2], points[3], points[4])
    thumb_vector = points[4] - points[2]
    upward = float(-thumb_vector[1])
    return bool(
        thumb_angle > 145.0
        and upward > palm_size * 0.75
        and upward > abs(float(thumb_vector[0])) * 1.25
    )


def classify_fist(landmarks) -> FistMetrics:
    """Classify a fist using scale-independent finger curl and compactness."""
    points = np.array(
        [(landmark.x, landmark.y, landmark.z) for landmark in landmarks],
        dtype=np.float32,
    )

    wrist = points[0]
    curled_fingers = 0

    for mcp_id, pip_id, tip_id in FINGER_JOINTS:
        if finger_is_curled(points, wrist, mcp_id, pip_id, tip_id):
            curled_fingers += 1

    palm_ids = (0, 5, 9, 13, 17)
    tip_ids = (8, 12, 16, 20)
    palm_center = points[list(palm_ids)].mean(axis=0)
    palm_size = max(float(np.linalg.norm(points[9] - wrist)), 1e-6)
    compactness = float(
        np.mean([np.linalg.norm(points[tip_id] - palm_center) for tip_id in tip_ids])
        / palm_size
    )

    # Requiring all four fingers prevents an index-point pose from becoming a fist.
    # A short unlock delay absorbs brief landmark misses on a real fist.
    # A raised thumb makes this an intentional thumbs-up, not a closed-fist
    # safety lock. All other compact four-finger poses remain locked.
    is_closed = (
        curled_fingers == 4
        and compactness < 1.35
        and not thumb_is_raised(points)
    )
    return FistMetrics(is_closed, curled_fingers, compactness)


def draw_hand_measurement(frame, landmarks, hand_name: str, metrics: FistMetrics) -> None:
    """Show why the current pose was or was not classified as a fist."""
    frame_height, frame_width = frame.shape[:2]
    wrist_x, wrist_y = normalized_to_pixel(landmarks[0], frame_width, frame_height)
    pose = "FIST" if metrics.is_closed else "OPEN"
    color = (0, 0, 255) if metrics.is_closed else (0, 255, 0)
    text = (
        f"{hand_name.upper()} {pose}  curl={metrics.curled_fingers}/4  "
        f"compact={metrics.compactness:.2f}"
    )
    cv2.putText(
        frame,
        text,
        (max(10, wrist_x - 80), max(25, wrist_y + 28)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        2,
        cv2.LINE_AA,
    )


def draw_status_panel(
    frame,
    lock_states: dict[str, HandLockState],
    seen_sides: set[str],
    fps: float,
) -> None:
    """Draw the safety state required before any controls are introduced."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (475, 205), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)

    rows = []
    for side in ("LEFT", "RIGHT"):
        status = "LOCKED" if lock_states[side].locked else "READY"
        suffix = " (NOT SEEN)" if side not in seen_sides else ""
        rows.append((f"{side}: {status}{suffix}", lock_states[side].locked))

    rows.extend(
        [
            ("MODE: DESKTOP", False),
            ("AIRDESK: SAFE PREVIEW", False),
        ]
    )

    for row_index, (text, is_locked) in enumerate(rows):
        color = (0, 80, 255) if is_locked else (0, 255, 0)
        cv2.putText(
            frame,
            text,
            (25, 42 + row_index * 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.68,
            color,
            2,
            cv2.LINE_AA,
        )

    cv2.putText(
        frame,
        f"FPS: {fps:.1f}  |  Q: quit",
        (25, 192),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


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

    lock_states = {"LEFT": HandLockState(), "RIGHT": HandLockState()}
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

                # Keep only the highest-confidence observation for each side.
                observations: dict[str, tuple[float, object, FistMetrics]] = {}
                for hand_index, landmarks in enumerate(result.hand_landmarks):
                    if hand_index >= len(result.handedness) or not result.handedness[hand_index]:
                        continue

                    category = result.handedness[hand_index][0]
                    side = category.category_name.upper()
                    if side not in lock_states:
                        continue

                    metrics = classify_fist(landmarks)
                    score = float(category.score)
                    existing = observations.get(side)
                    if existing is None or score > existing[0]:
                        observations[side] = (score, landmarks, metrics)

                seen_sides = set(observations)
                for side, state in lock_states.items():
                    observation = observations.get(side)
                    if observation is None:
                        state.update(False, False, current_time)
                        continue

                    _, landmarks, metrics = observation
                    state.update(metrics.is_closed, True, current_time)
                    draw_hand(frame, landmarks, side.title(), 235 if side == "LEFT" else 265)
                    draw_hand_measurement(frame, landmarks, side, metrics)

                draw_status_panel(frame, lock_states, seen_sides, smoothed_fps)
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
