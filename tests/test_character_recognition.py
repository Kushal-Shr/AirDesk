"""Deterministic tests for one-character image preprocessing."""

from __future__ import annotations

import unittest

import cv2
import numpy as np

from airdesk.character_recognition import (
    MODEL_IMAGE_SIZE,
    OCR_IMAGE_SIZE,
    prepare_model_image,
    preprocess_character,
)


class CharacterPreprocessingTests(unittest.TestCase):
    def test_empty_canvas_has_no_character(self):
        canvas = np.full((480, 640, 3), 255, dtype=np.uint8)
        self.assertIsNone(preprocess_character(canvas))

    def test_ink_is_cropped_scaled_and_centered(self):
        canvas = np.full((480, 640, 3), 255, dtype=np.uint8)
        cv2.line(canvas, (500, 350), (520, 420), (0, 0, 0), 8)
        result = preprocess_character(canvas)

        self.assertIsNotNone(result)
        self.assertEqual(result.shape, (OCR_IMAGE_SIZE, OCR_IMAGE_SIZE))
        rows, columns = np.where(result < 255)
        self.assertGreater(len(rows), 30)
        self.assertAlmostEqual(float(rows.mean()), OCR_IMAGE_SIZE / 2, delta=20)
        self.assertAlmostEqual(float(columns.mean()), OCR_IMAGE_SIZE / 2, delta=20)

    def test_model_image_matches_emnist_polarity_and_size(self):
        canvas = np.full((480, 640, 3), 255, dtype=np.uint8)
        cv2.line(canvas, (250, 150), (300, 350), (0, 0, 0), 8)
        result = prepare_model_image(canvas)

        self.assertEqual(result.shape, (MODEL_IMAGE_SIZE, MODEL_IMAGE_SIZE))
        self.assertEqual(int(result[0, 0]), 0)
        self.assertGreater(int(result.max()), 200)


if __name__ == "__main__":
    unittest.main()
