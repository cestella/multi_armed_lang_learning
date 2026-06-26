"""In-memory storage backend — for testing and ephemeral sessions."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone

from language_learning.models.events import Event
from language_learning.storage.base import StorageBackend


class InMemoryStorage(StorageBackend):
    """Volatile storage that lives only in RAM.

    Useful for unit tests and scenarios where no disk persistence is needed.
    Arms must be provided at construction time since there is no filesystem.
    """

    def __init__(self, arms: list[dict] | None = None) -> None:
        self._events: dict[str, list[dict]] = defaultdict(list)
        self._snapshots: dict[str, dict[str, dict]] = defaultdict(dict)
        self._transcripts: dict[str, dict[str, str]] = defaultdict(dict)
        self._history: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
        self._arms: list[dict] = arms or []

    def append_event(self, language: str, event: Event) -> None:
        self._events[language].append(event.model_dump())

    def read_events(self, language: str, after_event_id: str = "") -> list[dict]:
        events = list(self._events[language])
        if not after_event_id:
            return events
        found = False
        filtered = []
        for e in events:
            if found:
                filtered.append(e)
            elif e.get("event_id") == after_event_id:
                found = True
        return filtered

    def read_snapshot(self, language: str, snapshot_name: str) -> dict | None:
        return self._snapshots[language].get(snapshot_name)

    def write_snapshot(self, language: str, snapshot_name: str, data: dict) -> None:
        self._snapshots[language][snapshot_name] = dict(data)

    def append_transcript(self, language: str, date: str, markdown: str) -> None:
        existing = self._transcripts[language].get(date, "")
        self._transcripts[language][date] = existing + markdown

    def read_transcript(self, language: str, date: str) -> str:
        return self._transcripts[language].get(date, "")

    def append_history(self, language: str, history_name: str, record: dict) -> None:
        self._history[language][history_name].append(dict(record))

    def read_history(self, language: str, history_name: str) -> list[dict]:
        return list(self._history[language][history_name])

    def read_arms(self) -> list[dict]:
        return list(self._arms)

    def compact_logs(self, language: str) -> dict:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        events = self._events[language]
        summary = {
            "schema_version": 1,
            "language": language,
            "compacted_at": datetime.now(timezone.utc).isoformat(),
            "total_events": len(events),
            "bandit_snapshot": self._snapshots[language].get("bandit_state"),
            "learner_snapshot": self._snapshots[language].get("learner_profile"),
            "recent_turns": [
                {
                    "turn_id": e.get("turn_id"),
                    "text": e.get("payload", {}).get("text", "")[:100],
                }
                for e in events
                if e.get("type") == "user_submitted"
            ][-20:],
        }
        key = f"{ts}_summary"
        self._snapshots[language][key] = summary
        return {"status": "ok", "path": f"memory://{language}/{key}"}
