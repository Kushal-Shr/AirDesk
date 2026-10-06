"""Preprocess and classify one air-written uppercase letter."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .emnist_model import EmnistLetterCNN, LETTERS


OCR_IMAGE_SIZE = 256
OCR_INK_SIZE = 200
MODEL_IMAGE_SIZE = 28
MIN_INK_PIXELS = 30
MIN_CONFIDENCE = 0.25
MIN_TOP_TWO_MARGIN = 0.06
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "emnist_letters_cnn.pt"
LAST_CAPTURE_PATH = PROJECT_ROOT / "captures" / "last_character.png"


@dataclass(frozen=True)
class RecognitionResult:
    text: str
    confidence: float


def preprocess_character(canvas: np.ndarray) -> np.ndarray | None:
    """Crop the ink, center it on a square, and enlarge it consistently."""
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY) if canvas.ndim == 3 else canvas
    ink = gray < 220
    rows, columns = np.where(ink)
    if len(rows) < MIN_INK_PIXELS:
        return None

    top, bottom = int(rows.min()), int(rows.max()) + 1
    left, right = int(columns.min()), int(columns.max()) + 1
    crop = gray[top:bottom, left:right]
    height, width = crop.shape
    scale = min(OCR_INK_SIZE / max(width, 1), OCR_INK_SIZE / max(height, 1))
    resized = cv2.resize(
        crop,
        (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv2.INTER_CUBIC,
    )
    _, resized = cv2.threshold(resized, 200, 255, cv2.THRESH_BINARY)

    output = np.full((OCR_IMAGE_SIZE, OCR_IMAGE_SIZE), 255, dtype=np.uint8)
    output_y = (OCR_IMAGE_SIZE - resized.shape[0]) // 2
    output_x = (OCR_IMAGE_SIZE - resized.shape[1]) // 2
    output[
        output_y : output_y + resized.shape[0],
        output_x : output_x + resized.shape[1],
    ] = resized
    return output


def prepare_model_image(canvas: np.ndarray) -> np.ndarray | None:
    """Convert black-on-white AirDesk ink to EMNIST's white-on-black format."""
    prepared = preprocess_character(canvas)
    if prepared is None:
        return None
    resized = cv2.resize(
        prepared,
        (MODEL_IMAGE_SIZE, MODEL_IMAGE_SIZE),
        interpolation=cv2.INTER_AREA,
    )
    return 255 - resized


class EmnistCharacterRecognizer:
    """Run the trained EMNIST A–Z classifier on an AirDesk canvas."""

    def __init__(self, model_path: Path = DEFAULT_MODEL_PATH) -> None:
        self.model_path = Path(model_path)
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return self._model
        if not self.model_path.exists():
            raise RuntimeError(
                "handwriting model missing; run: "
                "PYTHONPATH=src python scripts/train_emnist_letters.py"
            )

        import torch

        checkpoint = torch.load(self.model_path, map_location="cpu", weights_only=True)
        if checkpoint.get("classes") != LETTERS:
            raise RuntimeError("the handwriting model has incompatible classes")
        model = EmnistLetterCNN()
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        self._model = model
        return model

    def recognize(self, canvas: np.ndarray) -> RecognitionResult | None:
        model_image = prepare_model_image(canvas)
        if model_image is None:
            return None

        LAST_CAPTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(LAST_CAPTURE_PATH), model_image)

        import torch

        image_tensor = torch.from_numpy(model_image.astype(np.float32) / 255.0)
        image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)
        image_tensor = (image_tensor - 0.5) / 0.5
        with torch.inference_mode():
            probabilities = torch.softmax(self._load_model()(image_tensor), dim=1)[0]
        top_confidences, top_indices = probabilities.topk(2)
        confidence_value = float(top_confidences[0].item())
        runner_up_value = float(top_confidences[1].item())
        if (
            confidence_value < MIN_CONFIDENCE
            or confidence_value - runner_up_value < MIN_TOP_TWO_MARGIN
        ):
            return None
        return RecognitionResult(LETTERS[int(top_indices[0].item())], confidence_value)


# Compatibility alias for callers created during the first OCR attempt.
MacVisionCharacterRecognizer = EmnistCharacterRecognizer
