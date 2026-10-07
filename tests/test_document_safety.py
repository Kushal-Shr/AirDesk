import json
import tempfile
import unittest
from pathlib import Path

from airdesk.document_safety import DocumentSafetyManager, InsertionRecoveryJournal


class DocumentSafetyTests(unittest.TestCase):
    def test_recovery_journal_keeps_only_the_latest_entries(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "recovery.jsonl"
            journal = InsertionRecoveryJournal(path, maximum_entries=2)
            journal.record("one")
            journal.record("two")
            journal.record("three")

            entries = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual([entry["text"] for entry in entries], ["two", "three"])
            self.assertTrue(all(entry["timestamp"] for entry in entries))

    def test_insertions_are_recovered_without_triggering_a_save(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = DocumentSafetyManager(
                InsertionRecoveryJournal(Path(directory) / "recovery.jsonl"),
            )
            saves = []
            manager.after_text_insert("first")
            manager.after_text_insert("second")

            entries = [
                json.loads(line)
                for line in manager.recovery_path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual([entry["text"] for entry in entries], ["first", "second"])
            self.assertEqual(saves, [])

    def test_manual_save_runs_only_when_explicitly_requested(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = DocumentSafetyManager(
                InsertionRecoveryJournal(Path(directory) / "recovery.jsonl"),
            )
            saves = []
            action = lambda: saves.append("save") or True
            manager.after_text_insert("text")

            self.assertEqual(saves, [])
            self.assertTrue(manager.save_now(action))
            self.assertEqual(saves, ["save"])


if __name__ == "__main__":
    unittest.main()
