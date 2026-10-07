"""Local recovery history and explicit document-save support."""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable


DEFAULT_RECOVERY_PATH = (
    Path.home() / "Library" / "Application Support" / "AirDesk" / "recovery.jsonl"
)


class InsertionRecoveryJournal:
    """Keep a small, local, atomic history of successfully inserted text."""

    def __init__(
        self,
        path: str | Path = DEFAULT_RECOVERY_PATH,
        maximum_entries: int = 100,
    ) -> None:
        self.path = Path(path)
        self.maximum_entries = max(1, int(maximum_entries))
        self._lock = threading.Lock()

    def record(self, text: str) -> None:
        if not text:
            return
        entry = json.dumps(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "text": text,
            },
            ensure_ascii=False,
        )
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            previous = []
            if self.path.exists():
                previous = self.path.read_text(encoding="utf-8").splitlines()
            lines = (previous + [entry])[-self.maximum_entries :]
            temporary = self.path.with_name(f"{self.path.name}.tmp")
            temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
            temporary.replace(self.path)


class DocumentSafetyManager:
    """Preserve inserted text without saving whichever app happens to be focused."""

    def __init__(
        self,
        journal: InsertionRecoveryJournal | None = None,
    ) -> None:
        self.journal = journal or InsertionRecoveryJournal()

    @property
    def recovery_path(self) -> Path:
        return self.journal.path

    def after_text_insert(self, text: str) -> None:
        try:
            self.journal.record(text)
        except OSError as error:
            print(f"AirDesk recovery log warning: {error}")

    def save_now(self, save_action: Callable[[], bool]) -> bool:
        return save_action()

    def close(self) -> None:
        pass
