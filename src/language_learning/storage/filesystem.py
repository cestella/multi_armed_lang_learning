"""Filesystem-backed storage implementation."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

import yaml

from language_learning.core.storage import append_jsonl, read_json, read_jsonl, write_json
from language_learning.models.events import Event
from language_learning.storage.base import StorageBackend


class FilesystemStorage(StorageBackend):
    """Stores all state as files under data_dir.

    Layout:
        data_dir/
          arms/arms.yaml
          state/<lang>/<snapshot_name>.json
          logs/<lang>/events.jsonl
          logs/<lang>/sessions/YYYY-MM-DD.md
          logs/<lang>/<history_name>.jsonl
          logs/<lang>/compacted/<ts>_summary.json
    """

    def __init__(self, data_dir: str | Path) -> None:
        self.data_dir = Path(data_dir)

    def append_event(self, language: str, event: Event) -> None:
        path = self.data_dir / "logs" / language / "events.jsonl"
        append_jsonl(path, event.model_dump())

    def read_events(self, language: str, after_event_id: str = "") -> list[dict]:
        path = self.data_dir / "logs" / language / "events.jsonl"
        events = read_jsonl(path)
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
        path = self.data_dir / "state" / language / f"{snapshot_name}.json"
        return read_json(path)

    def write_snapshot(self, language: str, snapshot_name: str, data: dict) -> None:
        path = self.data_dir / "state" / language / f"{snapshot_name}.json"
        write_json(path, data)

    def append_transcript(self, language: str, date: str, markdown: str) -> None:
        path = self.data_dir / "logs" / language / "sessions" / f"{date}.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a") as f:
            f.write(markdown)
            f.flush()
            os.fsync(f.fileno())

    def append_history(self, language: str, history_name: str, record: dict) -> None:
        path = self.data_dir / "state" / language / f"{history_name}.jsonl"
        append_jsonl(path, record)

    def read_history(self, language: str, history_name: str) -> list[dict]:
        path = self.data_dir / "state" / language / f"{history_name}.jsonl"
        return read_jsonl(path)

    def read_arms(self) -> list[dict]:
        arms_path = self.data_dir / "arms" / "arms.yaml"
        if not arms_path.exists():
            # Fall back to package-bundled arms
            arms_path = Path(__file__).parent.parent.parent.parent / "arms" / "arms.yaml"
        with open(arms_path) as f:
            data = yaml.safe_load(f)
        return data["arms"]

    def compact_logs(self, language: str) -> dict:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        events = read_jsonl(self.data_dir / "logs" / language / "events.jsonl")
        bandit_data = read_json(self.data_dir / "state" / language / "bandit_state.json")
        profile_data = read_json(self.data_dir / "state" / language / "learner_profile.json")

        summary = {
            "schema_version": 1,
            "language": language,
            "compacted_at": datetime.now(timezone.utc).isoformat(),
            "total_events": len(events),
            "bandit_snapshot": bandit_data,
            "learner_snapshot": profile_data,
            "recent_turns": [],
        }

        for e in [e for e in events if e.get("type") == "user_submitted"][-20:]:
            summary["recent_turns"].append({
                "turn_id": e.get("turn_id"),
                "text": e.get("payload", {}).get("text", "")[:100],
            })

        output_path = (
            self.data_dir / "logs" / language / "compacted" / f"{ts}_summary.json"
        )
        write_json(output_path, summary)
        return {"status": "ok", "path": str(output_path)}
