"""Deterministic tests for Stage 4 coordinate mapping and smoothing."""

from __future__ import annotations

import time
import unittest
from types import SimpleNamespace

import numpy as np

from airdesk.virtual_cursor import (
    ActiveRectangle,
    CAPTURE_HEIGHT,
    CAPTURE_WIDTH,
    CursorSmoother,
    DEFAULT_SWAP_HANDEDNESS,
    LatestFrameCapture,
    classify_index_point,
    configure_camera,
    limit_frame_resolution,
    map_to_preview,
    resolve_hand_side,
)


class FakeCamera:
    def __init__(self, frames=None):
        self.frames = list(frames or [])
        self.set_calls = []
        self.released = False

    def set(self, property_id, value):
        self.set_calls.append((property_id, value))
        return True

    def read(self):
        if self.frames:
            return True, self.frames.pop(0)
        return False, None

    def release(self):
        self.released = True


class CapturePerformanceTests(unittest.TestCase):
    def test_camera_requests_bounded_capture_settings(self):
        camera = FakeCamera()

        configure_camera(camera)

        values = [value for _property_id, value in camera.set_calls]
        self.assertIn(CAPTURE_WIDTH, values)
        self.assertIn(CAPTURE_HEIGHT, values)
        self.assertIn(30, values)
        self.assertIn(1, values)

    def test_oversized_frame_is_downscaled_without_distortion(self):
        large = np.zeros((720, 1280, 3), dtype=np.uint8)
        small = np.zeros((240, 320, 3), dtype=np.uint8)

        self.assertEqual(limit_frame_resolution(large).shape, (360, 640, 3))
        self.assertIs(limit_frame_resolution(small), small)

    def test_latest_frame_capture_drops_stale_frames(self):
        frames = [np.full((2, 2, 3), value, dtype=np.uint8) for value in (1, 2, 3)]
        camera = FakeCamera(frames)
        capture = LatestFrameCapture(camera).start()
        time.sleep(0.02)

        received, frame = capture.read(timeout=0.1)
        capture.release()

        self.assertTrue(received)
        self.assertEqual(int(frame[0, 0, 0]), 3)
        self.assertTrue(camera.released)


def make_pose(index_up: bool, other_fingers_up: bool, thumb_out: bool = False):
    points = [SimpleNamespace(x=0.5, y=0.7, z=0.0) for _ in range(21)]
    points[0] = SimpleNamespace(x=0.5, y=0.9, z=0.0)
    points[4] = SimpleNamespace(
        x=0.12 if thumb_out else 0.46,
        y=0.62 if thumb_out else 0.70,
        z=0.0,
    )

    for finger_index, (mcp_id, pip_id, dip_id, tip_id) in enumerate(
        ((5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20))
    ):
        x = 0.38 + finger_index * 0.08
        extended = index_up if finger_index == 0 else other_fingers_up
        points[mcp_id] = SimpleNamespace(x=x, y=0.65, z=0.0)
        points[pip_id] = SimpleNamespace(x=x, y=0.50, z=0.0)
        points[dip_id] = SimpleNamespace(x=x, y=0.35 if extended else 0.60, z=0.0)
        points[tip_id] = SimpleNamespace(x=x, y=0.20 if extended else 0.72, z=0.0)
    return points


class PointingPoseTests(unittest.TestCase):
    def test_index_only_pose_is_accepted(self):
        metrics = classify_index_point(make_pose(index_up=True, other_fingers_up=False))
        self.assertTrue(metrics.is_pointing)

    def test_open_palm_is_rejected(self):
        metrics = classify_index_point(make_pose(index_up=True, other_fingers_up=True))
        self.assertFalse(metrics.is_pointing)

    def test_closed_fist_is_rejected(self):
        metrics = classify_index_point(make_pose(index_up=False, other_fingers_up=False))
        self.assertFalse(metrics.is_pointing)

    def test_extended_thumb_is_rejected(self):
        metrics = classify_index_point(
            make_pose(index_up=True, other_fingers_up=False, thumb_out=True)
        )
        self.assertFalse(metrics.is_pointing)


class HandednessCorrectionTests(unittest.TestCase):
    def test_application_starts_with_labels_reversed_for_this_camera(self):
        self.assertTrue(DEFAULT_SWAP_HANDEDNESS)

    def test_normal_labels_are_unchanged(self):
        self.assertEqual(resolve_hand_side("Left", False), "LEFT")
        self.assertEqual(resolve_hand_side("Right", False), "RIGHT")

    def test_swapped_labels_are_reversed(self):
        self.assertEqual(resolve_hand_side("Left", True), "RIGHT")
        self.assertEqual(resolve_hand_side("Right", True), "LEFT")

    def test_unknown_label_is_ignored(self):
        self.assertIsNone(resolve_hand_side("Unknown", False))


class CoordinateMappingTests(unittest.TestCase):
    def setUp(self):
        self.area = ActiveRectangle(left=200, top=100, right=800, bottom=400)

    def test_active_area_edges_map_to_preview_edges(self):
        self.assertEqual(map_to_preview((200, 100), self.area, 1000, 500), (0, 0))
        self.assertEqual(map_to_preview((800, 400), self.area, 1000, 500), (999, 499))

    def test_points_outside_active_area_are_clamped(self):
        self.assertEqual(map_to_preview((0, 900), self.area, 1000, 500), (0, 499))


class CursorSmootherTests(unittest.TestCase):
    def test_first_sample_is_used_immediately(self):
        smoother = CursorSmoother(amount=0.25)
        self.assertEqual(smoother.update((100, 200)), (100, 200))

    def test_following_samples_are_smoothed(self):
        smoother = CursorSmoother(amount=0.25)
        smoother.update((100, 200))
        self.assertEqual(smoother.update((200, 100)), (125, 175))

    def test_reset_discards_the_old_position(self):
        smoother = CursorSmoother(amount=0.25)
        smoother.update((100, 200))
        smoother.reset()
        self.assertEqual(smoother.update((500, 400)), (500, 400))


if __name__ == "__main__":
    unittest.main()
