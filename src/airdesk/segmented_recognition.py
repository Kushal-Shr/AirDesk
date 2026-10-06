"""Segment a complete line and classify each character independently."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .character_recognition import RecognitionResult
from .emnist_model import EmnistLetterCNN
from .personal_samples import (
    DIGIT_LABELS,
    LOWERCASE_LABELS,
    SYMBOL_LABELS,
    UPPERCASE_LABELS,
)


MERGED_CLASSES = "".join(
    UPPERCASE_LABELS + LOWERCASE_LABELS + DIGIT_LABELS + SYMBOL_LABELS
)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MERGED_MODEL_PATH = PROJECT_ROOT / "models" / "airdesk_merged_characters.pt"
LAST_SEGMENTS_PATH = PROJECT_ROOT / "captures" / "last_segments.png"
MIN_SEGMENT_INK = 8


@dataclass(frozen=True)
class CharacterSegment:
    image: np.ndarray
    x_start: int
    x_end: int
    space_before: bool


@dataclass(frozen=True)
class CharacterCandidate:
    character: str
    confidence: float


@dataclass(frozen=True)
class CharacterPrediction:
    """Retain the local model's alternatives for contextual review."""

    character: str
    confidence: float
    alternatives: tuple[CharacterCandidate, ...]
    space_before: bool


def prepare_segment_model_image(ink: np.ndarray) -> np.ndarray | None:
    """Fit white ink into 28×28 while retaining its line-relative height."""
    if ink.ndim == 3:
        ink = cv2.cvtColor(ink, cv2.COLOR_BGR2GRAY)
    if int(np.count_nonzero(ink)) < MIN_SEGMENT_INK:
        return None
    height, width = ink.shape
    scale = min(24 / max(height, 1), 24 / max(width, 1))
    resized = cv2.resize(
        ink,
        (max(round(width * scale), 1), max(round(height * scale), 1)),
        interpolation=cv2.INTER_AREA,
    )
    output = np.zeros((28, 28), dtype=np.uint8)
    x = (28 - resized.shape[1]) // 2
    y = (28 - resized.shape[0]) // 2
    output[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    maximum = int(output.max())
    if maximum:
        output = np.clip(
            output.astype(np.float32) * (255.0 / maximum), 0, 255
        ).astype(np.uint8)
    return output


def _runs(active_columns: np.ndarray) -> list[tuple[int, int]]:
    runs = []
    start = None
    for index, active in enumerate(active_columns.tolist() + [False]):
        if active and start is None:
            start = index
        elif not active and start is not None:
            runs.append((start, index))
            start = None
    return runs


def segment_characters(canvas: np.ndarray) -> list[CharacterSegment]:
    """Split a single handwritten line at its empty vertical columns."""
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY) if canvas.ndim == 3 else canvas
    ink_mask = gray < 220
    rows, columns = np.where(ink_mask)
    if len(rows) < 40:
        return []
    top = max(int(rows.min()) - 5, 0)
    bottom = min(int(rows.max()) + 6, gray.shape[0])
    left = max(int(columns.min()) - 2, 0)
    right = min(int(columns.max()) + 3, gray.shape[1])
    line_mask = ink_mask[top:bottom, left:right]
    active = line_mask.any(axis=0)

    # Join very small internal gaps caused by separately drawn strokes while
    # retaining the larger gaps between written characters.
    bridge_limit = max(2, round(line_mask.shape[0] * 0.025))
    inactive_runs = _runs(~active)
    for gap_start, gap_end in inactive_runs:
        if gap_start > 0 and gap_end < len(active) and gap_end - gap_start <= bridge_limit:
            active[gap_start:gap_end] = True

    raw_runs = _runs(active)
    valid_runs = []
    for run_start, run_end in raw_runs:
        segment_mask = line_mask[:, run_start:run_end]
        if int(np.count_nonzero(segment_mask)) >= MIN_SEGMENT_INK:
            valid_runs.append((run_start, run_end))
    if not valid_runs:
        return []

    widths = [end - start for start, end in valid_runs]
    gaps = [
        valid_runs[index][0] - valid_runs[index - 1][1]
        for index in range(1, len(valid_runs))
    ]
    typical_width = float(np.median(widths))
    typical_gap = float(np.percentile(gaps, 30)) if gaps else 0.0
    space_threshold = max(
        line_mask.shape[0] * 0.12,
        typical_width * 0.62,
        typical_gap * 2.2,
        bridge_limit + 3,
    )

    white_ink = (line_mask.astype(np.uint8) * 255)
    segments = []
    for index, (run_start, run_end) in enumerate(valid_runs):
        model_image = prepare_segment_model_image(white_ink[:, run_start:run_end])
        if model_image is None:
            continue
        gap = 0 if index == 0 else run_start - valid_runs[index - 1][1]
        segments.append(
            CharacterSegment(
                image=model_image,
                x_start=left + run_start,
                x_end=left + run_end,
                space_before=index > 0 and gap >= space_threshold,
            )
        )
    return segments


class SegmentedCharacterRecognizer:
    """Classify segmented characters, then concatenate them without NLP."""

    def __init__(self, model_path: Path = DEFAULT_MERGED_MODEL_PATH) -> None:
        self.model_path = Path(model_path)
        self._model = None
        self.last_predictions: tuple[CharacterPrediction, ...] = ()

    def _load_model(self):
        if self._model is not None:
            return self._model
        if not self.model_path.exists():
            raise RuntimeError(
                "merged character model missing; run: "
                "PYTHONPATH=src python scripts/train_merged_characters.py"
            )
        import torch

        checkpoint = torch.load(self.model_path, map_location="cpu", weights_only=True)
        if checkpoint.get("approved") is not True:
            raise RuntimeError("the merged character model has not passed validation")
        if checkpoint.get("classes") != MERGED_CLASSES:
            raise RuntimeError("the merged character model has incompatible classes")
        model = EmnistLetterCNN(number_of_classes=len(MERGED_CLASSES))
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        self._model = model
        return model

    def verify_ready(self) -> None:
        self._load_model()

    def recognize(self, canvas: np.ndarray) -> RecognitionResult | None:
        segments = segment_characters(canvas)
        if not segments:
            self.last_predictions = ()
            return None
        import torch

        images = np.stack([segment.image for segment in segments]).astype(np.float32) / 255.0
        tensor = torch.from_numpy(images).unsqueeze(1)
        tensor = (tensor - 0.5) / 0.5
        with torch.inference_mode():
            probabilities = torch.softmax(self._load_model()(tensor), dim=1)
        candidate_count = min(3, probabilities.shape[1])
        top_confidences, top_indices = probabilities.topk(candidate_count, dim=1)
        confidences = top_confidences[:, 0]
        indices = top_indices[:, 0]
        output = []
        predictions = []
        for row, (segment, index) in enumerate(zip(segments, indices.tolist())):
            if segment.space_before:
                output.append(" ")
            character = MERGED_CLASSES[index]
            output.append(character)
            alternatives = tuple(
                CharacterCandidate(
                    MERGED_CLASSES[candidate_index],
                    float(candidate_confidence),
                )
                for candidate_confidence, candidate_index in zip(
                    top_confidences[row].tolist(),
                    top_indices[row].tolist(),
                )
            )
            predictions.append(
                CharacterPrediction(
                    character=character,
                    confidence=float(confidences[row]),
                    alternatives=alternatives,
                    space_before=segment.space_before,
                )
            )
        self.last_predictions = tuple(predictions)

        LAST_SEGMENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        montage = np.concatenate([segment.image for segment in segments], axis=1)
        cv2.imwrite(str(LAST_SEGMENTS_PATH), montage)
        return RecognitionResult(
            "".join(output),
            float(confidences.mean().item()),
        )
