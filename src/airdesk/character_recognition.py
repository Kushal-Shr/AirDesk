"""Preprocess and classify one air-written character in a selected mode."""

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
RECOGNITION_CLASSES = {
    "uppercase": "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
    "lowercase": "abcdefghijklmnopqrstuvwxyz",
    "digits": "0123456789",
    "symbols": ".,?!@#$%&+-_=()[]{}/:",
}
PERSONAL_MODEL_PATHS = {
    mode: PROJECT_ROOT / "models" / f"airdesk_{mode}.pt"
    for mode in RECOGNITION_CLASSES
}


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


def make_writing_guide(screen_size: tuple[int, int], label: str):
    """Create the fixed cell used when a character's position matters."""
    screen_width, screen_height = screen_size
    guide_height = min(round(screen_height * 0.62), 620)
    guide_width = min(round(guide_height * 0.78), round(screen_width * 0.55))
    guide_x = (screen_width - guide_width) // 2
    guide_y = (screen_height - guide_height) // 2
    baseline_y = guide_y + round(guide_height * 0.80)
    return guide_x, guide_y, guide_width, guide_height, baseline_y, label


def rasterize_guide_strokes(strokes, guide, image_size: int = 256):
    """Render screen-space paths relative to a fixed writing cell."""
    guide_x, guide_y, guide_width, guide_height, _baseline_y, _label = guide
    output = np.zeros((image_size, image_size), dtype=np.uint8)
    scale_x = (image_size - 1) / max(guide_width, 1)
    scale_y = (image_size - 1) / max(guide_height, 1)
    thickness = max(2, round(6 * (scale_x + scale_y) / 2))

    for stroke in strokes:
        points = [
            (
                round((float(point_x) - guide_x) * scale_x),
                round((float(point_y) - guide_y) * scale_y),
            )
            for point_x, point_y in stroke
        ]
        if len(points) == 1:
            cv2.circle(output, points[0], thickness, 255, -1, cv2.LINE_AA)
        for start, end in zip(points, points[1:]):
            cv2.line(output, start, end, 255, thickness, cv2.LINE_AA)

    if int(np.count_nonzero(output)) < MIN_INK_PIXELS:
        return None
    return output


def prepare_position_model_image(strokes, guide) -> np.ndarray | None:
    """Preserve punctuation position while producing a 28×28 model image."""
    guide_image = rasterize_guide_strokes(strokes, guide)
    if guide_image is None:
        return None
    resized = cv2.resize(
        guide_image,
        (MODEL_IMAGE_SIZE, MODEL_IMAGE_SIZE),
        interpolation=cv2.INTER_AREA,
    )
    maximum = int(resized.max())
    if maximum > 0:
        resized = np.clip(
            resized.astype(np.float32) * (255.0 / maximum), 0, 255
        ).astype(np.uint8)
    return resized


class PersonalCharacterRecognizer:
    """Run one validated mode-specific AirDesk character classifier."""

    def __init__(self, mode: str = "uppercase", model_path: Path | None = None) -> None:
        if mode not in RECOGNITION_CLASSES:
            raise ValueError(f"unsupported recognition mode: {mode}")
        self.mode = mode
        self.classes = RECOGNITION_CLASSES[mode]
        self.model_path = Path(model_path or PERSONAL_MODEL_PATHS[mode])
        self.preserves_position = mode == "symbols"
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return self._model
        if not self.model_path.exists():
            raise RuntimeError(
                f"validated {self.mode} model missing; run: "
                "PYTHONPATH=src python scripts/train_personal_models.py"
            )

        import torch

        checkpoint = torch.load(self.model_path, map_location="cpu", weights_only=True)
        if checkpoint.get("approved") is not True:
            raise RuntimeError(f"the {self.mode} model has not passed validation")
        if checkpoint.get("mode") != self.mode:
            raise RuntimeError("the handwriting model has an incompatible mode")
        if checkpoint.get("classes") != self.classes:
            raise RuntimeError("the handwriting model has incompatible classes")
        model = EmnistLetterCNN(number_of_classes=len(self.classes))
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        self._model = model
        return model

    def verify_ready(self) -> None:
        """Load and validate the checkpoint before opening the camera."""
        self._load_model()

    def _predict(self, model_image: np.ndarray) -> RecognitionResult | None:
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
        return RecognitionResult(
            self.classes[int(top_indices[0].item())], confidence_value
        )

    def recognize(self, canvas: np.ndarray) -> RecognitionResult | None:
        if self.preserves_position:
            raise RuntimeError("symbol recognition requires guide-relative strokes")
        model_image = prepare_model_image(canvas)
        return None if model_image is None else self._predict(model_image)

    def recognize_strokes(self, strokes, guide) -> RecognitionResult | None:
        if not self.preserves_position:
            raise RuntimeError(f"{self.mode} recognition uses the drawing canvas")
        model_image = prepare_position_model_image(strokes, guide)
        return None if model_image is None else self._predict(model_image)


class MultiModeCharacterRecognizer:
    """Keep all validated classifiers loaded and route to the selected mode."""

    modes = tuple(RECOGNITION_CLASSES)

    def __init__(self, initial_mode: str = "lowercase") -> None:
        if initial_mode not in self.modes:
            raise ValueError(f"unsupported recognition mode: {initial_mode}")
        self._recognizers = {
            mode: PersonalCharacterRecognizer(mode) for mode in self.modes
        }
        self.mode = initial_mode

    @property
    def current(self) -> PersonalCharacterRecognizer:
        return self._recognizers[self.mode]

    @property
    def classes(self) -> str:
        return self.current.classes

    @property
    def preserves_position(self) -> bool:
        return self.current.preserves_position

    def verify_ready(self) -> None:
        """Require every model so mode switching cannot fail halfway through text."""
        for recognizer in self._recognizers.values():
            recognizer.verify_ready()

    def cycle_mode(self) -> str:
        current_index = self.modes.index(self.mode)
        self.mode = self.modes[(current_index + 1) % len(self.modes)]
        return self.mode

    def recognize(self, canvas: np.ndarray) -> RecognitionResult | None:
        return self.current.recognize(canvas)

    def recognize_strokes(self, strokes, guide) -> RecognitionResult | None:
        return self.current.recognize_strokes(strokes, guide)


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
