"""Tests for labeled personal sample collection without camera access."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from airdesk.personal_samples import (
    DIGIT_LABELS,
    LABEL_GROUPS,
    PersonalSampleStore,
    label_key,
    make_writing_guide,
    rasterize_guide_strokes,
)


class PersonalSampleTests(unittest.TestCase):
    def test_digit_collection_group_is_available(self):
        self.assertEqual(DIGIT_LABELS, tuple("0123456789"))
        self.assertEqual(LABEL_GROUPS["digits"], DIGIT_LABELS)
        self.assertIn("0", LABEL_GROUPS["all"])

    def test_label_key_is_safe_for_letters_and_slash(self):
        self.assertEqual(label_key("A"), "U+0041")
        self.assertEqual(label_key("a"), "U+0061")
        self.assertEqual(label_key("/"), "U+002F")

    def test_rasterization_preserves_position_inside_guide(self):
        guide = (100, 50, 400, 500, 450, ".")
        image = rasterize_guide_strokes(
            [[(290, 420), (300, 430), (310, 430)]],
            guide,
        )
        self.assertIsNotNone(image)
        rows, columns = np.where(image > 0)
        self.assertGreater(float(rows.mean()), 256 * 0.65)
        self.assertAlmostEqual(float(columns.mean()), 256 / 2, delta=20)

    def test_save_writes_images_metadata_and_advances(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            store = PersonalSampleStore(root, "lowercase", ("a", "b"), 1)
            guide = make_writing_guide((1200, 800), "a")
            x, y, width, height, _baseline, _label = guide
            strokes = [[
                (x + width * 0.25, y + height * 0.75),
                (x + width * 0.50, y + height * 0.20),
                (x + width * 0.75, y + height * 0.75),
            ]]

            self.assertTrue(store.save(strokes, guide))
            self.assertEqual(store.current_label, "b")
            directory = store.directory_for("a")
            self.assertEqual(len(tuple(directory.glob("*_model.png"))), 1)
            self.assertEqual(len(tuple(directory.glob("*_guide.png"))), 1)
            metadata_files = tuple(directory.glob("*.json"))
            self.assertEqual(len(metadata_files), 1)
            metadata = json.loads(metadata_files[0].read_text(encoding="utf-8"))
            self.assertEqual(metadata["label"], "a")
            self.assertTrue(metadata["strokes"])

            resumed = PersonalSampleStore(root, "lowercase", ("a", "b"), 1)
            self.assertEqual(resumed.pending, ["b"])


if __name__ == "__main__":
    unittest.main()
