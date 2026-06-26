"""Abstract storage backend interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from language_learning.models.events import Event


class StorageBackend(ABC):
    """Protocol for all persistence operations.

    Implementations must be safe for concurrent access within a single process
    and must write snapshots atomically (write-temp → rename).
    """

    @abstractmethod
    def append_event(self, language: str, event: Event) -> None:
        """Append a domain event to the event log for the given language."""

    @abstractmethod
    def read_events(self, language: str, after_event_id: str = "") -> list[dict]:
        """Return all events for language, optionally starting after after_event_id."""

    @abstractmethod
    def read_snapshot(self, language: str, snapshot_name: str) -> dict | None:
        """Return the snapshot dict, or None if it does not exist."""

    @abstractmethod
    def write_snapshot(self, language: str, snapshot_name: str, data: dict) -> None:
        """Atomically write a snapshot."""

    @abstractmethod
    def append_transcript(self, language: str, date: str, markdown: str) -> None:
        """Append markdown text to today's session transcript."""

    @abstractmethod
    def append_history(self, language: str, history_name: str, record: dict) -> None:
        """Append a timestamped record to an append-only history file."""

    @abstractmethod
    def read_history(self, language: str, history_name: str) -> list[dict]:
        """Return all records from a history file."""

    @abstractmethod
    def read_arms(self) -> list[dict]:
        """Return the list of arm definitions."""

    @abstractmethod
    def compact_logs(self, language: str) -> dict:
        """Compact event logs into a summary and return {"status": "ok", "path": ...}."""
