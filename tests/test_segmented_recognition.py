"""Tests for character-by-character line segmentation."""

from __future__ import annotations

import unittest

import cv2
import numpy as np

from airdesk.segmented_recognition import prepare_segment_model_image, segment_characters


class SegmentedRecognitionTests(unittest.TestCase):
    def test_separates_characters_and_detects_large_space(self):
        canvas = np.full((180, 700), 255, dtype=np.uint8)
        cv2.putText(canvas, "Hi", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 2, 0, 5)
        cv2.putText(canvas, "2!", (330, 120), cv2.FONT_HERSHEY_SIMPLEX, 2, 0, 5)
        segments = segment_characters(canvas)

        self.assertEqual(len(segments), 4)
        self.assertFalse(segments[0].space_before)
        self.assertFalse(segments[1].space_before)
        self.assertTrue(segments[2].space_before)
        self.assertFalse(segments[3].space_before)
        self.assertTrue(all(segment.image.shape == (28, 28) for segment in segments))

    def test_dot_and_stem_with_overlapping_x_are_one_character(self):
        canvas = np.full((180, 200), 255, dtype=np.uint8)
        cv2.circle(canvas, (90, 35), 5, 0, -1)
        cv2.line(canvas, (90, 60), (90, 135), 0, 6)
        segments = segment_characters(canvas)
        self.assertEqual(len(segments), 1)

    def test_empty_canvas_has_no_segments(self):
        self.assertEqual(segment_characters(np.full((100, 200), 255, dtype=np.uint8)), [])

    def test_segment_preparation_preserves_vertical_position(self):
        ink = np.zeros((100, 20), dtype=np.uint8)
        ink[80:90, 5:15] = 255
        image = prepare_segment_model_image(ink)
        rows, _columns = np.where(image > 0)
        self.assertGreater(float(rows.mean()), 20)


if __name__ == "__main__":
    unittest.main()
