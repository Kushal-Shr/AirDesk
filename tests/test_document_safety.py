import json
import tempfile
import unittest
from pathlib import Path

from airdesk.document_safety import DocumentSafetyManager, InsertionRecoveryJournal


class FakeTimer:
    created = []

    def __init__(self, delay, callback):
        self.delay = delay
        self.callback = callback
        self.cancelled = False
        self.daemon = False
        self.__class__.created.append(self)

    def start(self):
        pass

    def cancel(self):
        self.cancelled = True

    def fire(self):
        if not self.cancelled:
            self.callback()


class DocumentSafetyTests(unittest.TestCase):
    def setUp(self):
        FakeTimer.created = []

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

    def test_repeated_insertions_debounce_to_one_save(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = DocumentSafetyManager(
                InsertionRecoveryJournal(Path(directory) / "recovery.jsonl"),
                save_delay=1.0,
                timer_factory=FakeTimer,
            )
            saves = []
            action = lambda: saves.append("save") or True

            manager.after_text_insert("first", action)
            manager.after_text_insert("second", action)

            self.assertEqual(len(FakeTimer.created), 2)
            self.assertTrue(FakeTimer.created[0].cancelled)
            FakeTimer.created[0].fire()
            FakeTimer.created[1].fire()
            self.assertEqual(saves, ["save"])

    def test_manual_save_cancels_pending_auto_save(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = DocumentSafetyManager(
                InsertionRecoveryJournal(Path(directory) / "recovery.jsonl"),
                timer_factory=FakeTimer,
            )
            saves = []
            action = lambda: saves.append("save") or True
            manager.after_text_insert("text", action)

            self.assertTrue(manager.save_now(action))
            FakeTimer.created[0].fire()
            self.assertEqual(saves, ["save"])


if __name__ == "__main__":
    unittest.main()
