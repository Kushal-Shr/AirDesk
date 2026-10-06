"""Gemini correction tests that never make a network request."""

from __future__ import annotations

import json
import unittest
from types import SimpleNamespace

import cv2
import numpy as np

from airdesk.gemini_correction import (
    GeminiCorrectionClient,
    build_review_prompt,
    prepare_gemini_ink_image,
)
from airdesk.segmented_recognition import CharacterCandidate, CharacterPrediction


class FakeModels:
    def __init__(self, output):
        self.output = output
        self.calls = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(text=json.dumps(self.output))


class TemporaryCapacityError(RuntimeError):
    code = 503


class FallbackModels(FakeModels):
    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        if len(self.calls) == 1:
            raise TemporaryCapacityError("primary model busy")
        return SimpleNamespace(text=json.dumps(self.output))


class GeminiCorrectionTests(unittest.TestCase):
    def test_canvas_is_cropped_to_ink(self):
        canvas = np.full((300, 500, 3), 255, dtype=np.uint8)
        cv2.line(canvas, (200, 120), (260, 180), (0, 0, 0), 7)

        cropped = prepare_gemini_ink_image(canvas)

        self.assertIsNotNone(cropped)
        self.assertLess(cropped.shape[0], canvas.shape[0])
        self.assertLess(cropped.shape[1], canvas.shape[1])
        self.assertTrue(np.any(cropped < 220))

    def test_prompt_contains_local_alternatives_and_conservative_rules(self):
        predictions = (
            CharacterPrediction(
                character="C",
                confidence=0.55,
                alternatives=(
                    CharacterCandidate("C", 0.55),
                    CharacterCandidate("c", 0.40),
                ),
                space_before=False,
            ),
        )

        prompt = build_review_prompt("C", predictions)

        self.assertIn("'C':0.550", prompt)
        self.assertIn("'c':0.400", prompt)
        self.assertIn("Do not autocomplete", prompt)

    def test_review_sends_png_and_parses_structured_result(self):
        models = FakeModels(
            {"text": "Hello 2!", "confidence": 0.93, "explanation": "visual match"}
        )
        client = SimpleNamespace(models=models)
        reviewer = GeminiCorrectionClient("test-key", client=client)
        canvas = np.full((200, 500), 255, dtype=np.uint8)
        cv2.putText(canvas, "Hello", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 2, 0, 5)

        result = reviewer.review(canvas, "He11o 2!", ())

        self.assertEqual(result.text, "Hello 2!")
        self.assertAlmostEqual(result.confidence, 0.93)
        request = models.calls[0]
        self.assertEqual(request["model"], reviewer.model)
        image_part = request["contents"][1]
        self.assertEqual(image_part.inline_data.mime_type, "image/png")
        self.assertGreater(len(image_part.inline_data.data), 20)
        self.assertEqual(request["config"].response_mime_type, "application/json")
        self.assertEqual(request["config"].temperature, 0)

    def test_unsupported_output_characters_are_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "unsupported"):
            GeminiCorrectionClient._parse_response(
                json.dumps(
                    {
                        "text": "hello 🙂",
                        "confidence": 0.9,
                        "explanation": "not in the local alphabet",
                    }
                )
            )

    def test_capacity_error_retries_with_fallback_model(self):
        models = FallbackModels(
            {"text": "Hi", "confidence": 0.9, "explanation": "visual match"}
        )
        reviewer = GeminiCorrectionClient(
            "test-key",
            model="primary",
            fallback_model="fallback",
            client=SimpleNamespace(models=models),
        )
        canvas = np.full((200, 400), 255, dtype=np.uint8)
        cv2.putText(canvas, "Hi", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 2, 0, 5)

        result = reviewer.review(canvas, "Hl", ())

        self.assertEqual(result.text, "Hi")
        self.assertEqual([call["model"] for call in models.calls], ["primary", "fallback"])
        self.assertEqual(reviewer.last_model_used, "fallback")


if __name__ == "__main__":
    unittest.main()
