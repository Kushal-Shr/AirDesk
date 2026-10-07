"""Local recovery history and non-blocking document auto-save support."""

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
    """Debounce document saves while preserving every inserted text fragment."""

    def __init__(
        self,
        journal: InsertionRecoveryJournal | None = None,
        save_delay: float = 1.0,
        timer_factory=threading.Timer,
    ) -> None:
        self.journal = journal or InsertionRecoveryJournal()
        self.save_delay = max(0.0, float(save_delay))
        self._timer_factory = timer_factory
        self._timer = None
        self._generation = 0
        self._lock = threading.Lock()

    @property
    def recovery_path(self) -> Path:
        return self.journal.path

    def after_text_insert(
        self,
        text: str,
        save_action: Callable[[], bool],
    ) -> None:
        try:
            self.journal.record(text)
        except OSError as error:
            print(f"AirDesk recovery log warning: {error}")
        self._schedule_save(save_action)

    def _schedule_save(self, save_action: Callable[[], bool]) -> None:
        with self._lock:
            self._generation += 1
            generation = self._generation
            if self._timer is not None:
                self._timer.cancel()
            timer = self._timer_factory(
                self.save_delay,
                lambda: self._run_scheduled_save(generation, save_action),
            )
            if hasattr(timer, "daemon"):
                timer.daemon = True
            self._timer = timer
            timer.start()

    def _run_scheduled_save(
        self,
        generation: int,
        save_action: Callable[[], bool],
    ) -> None:
        with self._lock:
            if generation != self._generation:
                return
            self._timer = None
        if save_action():
            print("AirDesk document: AUTO-SAVED")

    def save_now(self, save_action: Callable[[], bool]) -> bool:
        with self._lock:
            self._generation += 1
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None
        return save_action()

    def close(self) -> None:
        with self._lock:
            self._generation += 1
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None
