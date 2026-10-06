"""Personal air-writing sample collection for later model fine-tuning."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import cv2
import numpy as np

from .character_recognition import (
    make_writing_guide,
    prepare_model_image,
    rasterize_guide_strokes,
)
from .whiteboard import AirWritingController


UPPERCASE_LABELS = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
LOWERCASE_LABELS = tuple("abcdefghijklmnopqrstuvwxyz")
DIGIT_LABELS = tuple("0123456789")
SYMBOL_LABELS = (
    ".", ",", "?", "!", "@", "#", "$", "%", "&", "+", "-", "_", "=",
    "(", ")", "[", "]", "{", "}", "/", ":",
)
LABEL_GROUPS = {
    "uppercase": UPPERCASE_LABELS,
    "lowercase": LOWERCASE_LABELS,
    "digits": DIGIT_LABELS,
    "symbols": SYMBOL_LABELS,
    "all": UPPERCASE_LABELS + LOWERCASE_LABELS + DIGIT_LABELS + SYMBOL_LABELS,
}
PERSONAL_DATA_PATH = Path(__file__).resolve().parents[2] / "personal_data"


def label_key(label: str) -> str:
    """Return a filesystem-safe, unambiguous folder name for one character."""
    return f"U+{ord(label):04X}"


def normalized_strokes(strokes, guide):
    guide_x, guide_y, guide_width, guide_height, _baseline_y, _label = guide
    return [
        [
            [
                (float(point_x) - guide_x) / max(guide_width, 1),
                (float(point_y) - guide_y) / max(guide_height, 1),
            ]
            for point_x, point_y in stroke
        ]
        for stroke in strokes
        if stroke
    ]


class PersonalSampleStore:
    """Track collection progress and save labeled image/path pairs."""

    def __init__(
        self,
        root: Path,
        group: str,
        labels,
        samples_per_character: int,
    ) -> None:
        self.root = Path(root)
        self.group = group
        self.labels = tuple(labels)
        self.samples_per_character = samples_per_character
        self.pending = self._build_pending_schedule()
        self.total_goal = len(self.labels) * self.samples_per_character

    def directory_for(self, label: str) -> Path:
        return self.root / self.group / label_key(label)

    def count(self, label: str) -> int:
        directory = self.directory_for(label)
        return len(tuple(directory.glob("*_model.png"))) if directory.exists() else 0

    def _build_pending_schedule(self) -> list[str]:
        pending = []
        for round_index in range(self.samples_per_character):
            for label in self.labels:
                if self.count(label) <= round_index:
                    pending.append(label)
        return pending

    @property
    def current_label(self) -> str | None:
        return self.pending[0] if self.pending else None

    @property
    def completed_count(self) -> int:
        return self.total_goal - len(self.pending)

    def save(self, strokes, guide) -> bool:
        label = self.current_label
        if label is None:
            return False
        guide_image = rasterize_guide_strokes(strokes, guide)
        if guide_image is None:
            return False
        model_image = prepare_model_image(255 - guide_image)
        if model_image is None:
            return False

        directory = self.directory_for(label)
        directory.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        stem = directory / stamp
        model_path = Path(f"{stem}_model.png")
        guide_path = Path(f"{stem}_guide.png")
        metadata_path = Path(f"{stem}.json")
        if not cv2.imwrite(str(model_path), model_image):
            raise RuntimeError(f"could not save {model_path}")
        if not cv2.imwrite(str(guide_path), guide_image):
            raise RuntimeError(f"could not save {guide_path}")
        metadata = {
            "label": label,
            "codepoint": label_key(label),
            "group": self.group,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "model_image": model_path.name,
            "guide_image": guide_path.name,
            "strokes": normalized_strokes(strokes, guide),
        }
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        self.pending.pop(0)
        return True


class PersonalSampleController(AirWritingController):
    """Replace text insertion with labeled sample capture on thumbs-up."""

    def __init__(self, overlay, store: PersonalSampleStore) -> None:
        super().__init__(overlay=overlay)
        self.store = store
        self.guide = make_writing_guide(overlay.size, store.current_label or "✓")
        self.overlay.set_guide(self.guide)

    def _refresh_guide(self) -> None:
        label = self.store.current_label or "✓"
        self.guide = make_writing_guide(self.overlay.size, label)
        self.overlay.set_guide(self.guide)

    def _set_mode(self, mode: str, system_controller) -> None:
        super()._set_mode(mode, system_controller)
        if self.is_overlay_active:
            self._refresh_guide()

    def _recognize_and_commit(self, _system_controller) -> None:
        if self.store.current_label is None:
            self.recognition_feedback = "COLLECTION COMPLETE"
            return
        expected_label = self.store.current_label
        try:
            saved = self.store.save(self.overlay.strokes, self.guide)
        except Exception as error:
            self.recognition_feedback = f"SAVE ERROR: {error}"
            return
        if not saved:
            self.recognition_feedback = "DRAW INSIDE THE GUIDE, THEN RETRY"
            return
        self._clear_writing()
        self.recognition_feedback = f"SAVED: {expected_label}"
        self._refresh_guide()

    def _update_status(self, lock_states) -> None:
        label = self.store.current_label
        if label is None:
            self.overlay.set_status(
                "TRAINING COMPLETE | both palms: return | Ctrl+C: quit"
            )
            return
        current_number = self.store.count(label) + 1
        progress = f"{self.store.completed_count}/{self.store.total_goal}"
        accept = ""
        if self.accept_hold.progress > 0 and not self.accept_hold.latched:
            accept = f" | SAVE {round(self.accept_hold.progress * 100)}%"
        feedback = f" | {self.recognition_feedback}" if self.recognition_feedback else ""
        self.overlay.set_status(
            f"TRAIN {self.store.group.upper()} | Draw: {label} | "
            f"sample {current_number}/{self.store.samples_per_character} | "
            f"total {progress} | 👍 save{accept}{feedback}"
        )
