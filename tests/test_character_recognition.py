"""Deterministic tests for one-character image preprocessing."""

from __future__ import annotations

import unittest
from pathlib import Path
import tempfile

import cv2
import numpy as np
import torch

from airdesk.character_recognition import (
    MODEL_IMAGE_SIZE,
    OCR_IMAGE_SIZE,
    PersonalCharacterRecognizer,
    prepare_model_image,
    prepare_position_model_image,
    preprocess_character,
)
from airdesk.emnist_model import EmnistLetterCNN


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

    def test_position_image_keeps_a_dot_near_the_baseline(self):
        guide = (100, 50, 400, 500, 450, "SYMBOLS")
        image = prepare_position_model_image([[(300, 425), (305, 430)]], guide)

        self.assertIsNotNone(image)
        rows, _columns = np.where(image > 0)
        self.assertGreater(float(rows.mean()), MODEL_IMAGE_SIZE * 0.65)


class PersonalRecognizerTests(unittest.TestCase):
    def _checkpoint(self, path: Path, approved: bool = True):
        model = EmnistLetterCNN(number_of_classes=10)
        torch.save(
            {
                "state_dict": model.state_dict(),
                "classes": "0123456789",
                "mode": "digits",
                "approved": approved,
            },
            path,
        )

    def test_validated_checkpoint_loads(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "digits.pt"
            self._checkpoint(path)
            recognizer = PersonalCharacterRecognizer("digits", path)
            recognizer.verify_ready()

    def test_unapproved_checkpoint_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "digits.pt"
            self._checkpoint(path, approved=False)
            recognizer = PersonalCharacterRecognizer("digits", path)
            with self.assertRaisesRegex(RuntimeError, "not passed validation"):
                recognizer.verify_ready()


if __name__ == "__main__":
    unittest.main()
