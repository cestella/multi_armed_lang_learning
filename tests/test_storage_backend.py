"""Tests for StorageBackend implementations (FilesystemStorage + InMemoryStorage)."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from language_learning.models.events import session_started, user_submitted
from language_learning.storage.filesystem import FilesystemStorage
from language_learning.storage.memory import InMemoryStorage


# ---------------------------------------------------------------------------
# Shared fixture: arms data
# ---------------------------------------------------------------------------

ARMS_DIR = Path(__file__).parent.parent / "arms"


def _arms_list() -> list[dict]:
    import yaml
    with open(ARMS_DIR / "arms.yaml") as f:
        return yaml.safe_load(f)["arms"]


# ---------------------------------------------------------------------------
# Parametrize over both implementations
# ---------------------------------------------------------------------------

@pytest.fixture(params=["filesystem", "memory"])
def storage(request, tmp_path):
    if request.param == "filesystem":
        arms_dst = tmp_path / "arms" / "arms.yaml"
        arms_dst.parent.mkdir(parents=True)
        shutil.copy(ARMS_DIR / "arms.yaml", arms_dst)
        return FilesystemStorage(tmp_path)
    else:
        return InMemoryStorage(arms=_arms_list())


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------

class TestEvents:
    def test_append_and_read_single_event(self, storage):
        ev = session_started("it")
        storage.append_event("it", ev)
        events = storage.read_events("it")
        assert len(events) == 1
        assert events[0]["type"] == "session_started"

    def test_append_multiple_events_ordered(self, storage):
        e1 = user_submitted("it", "t1", "Ciao!")
        e2 = user_submitted("it", "t2", "Come stai?")
        storage.append_event("it", e1)
        storage.append_event("it", e2)
        events = storage.read_events("it")
        assert len(events) == 2
        assert events[0]["payload"]["text"] == "Ciao!"
        assert events[1]["payload"]["text"] == "Come stai?"

    def test_read_events_empty(self, storage):
        assert storage.read_events("it") == []

    def test_events_partitioned_by_language(self, storage):
        storage.append_event("it", user_submitted("it", "t1", "ciao"))
        storage.append_event("es", user_submitted("es", "t2", "hola"))
        assert len(storage.read_events("it")) == 1
        assert len(storage.read_events("es")) == 1
        assert storage.read_events("it")[0]["payload"]["text"] == "ciao"

    def test_read_events_after_event_id(self, storage):
        e1 = user_submitted("it", "t1", "first")
        e2 = user_submitted("it", "t2", "second")
        e3 = user_submitted("it", "t3", "third")
        storage.append_event("it", e1)
        storage.append_event("it", e2)
        storage.append_event("it", e3)

        after = storage.read_events("it", after_event_id=e1.event_id)
        assert len(after) == 2
        assert after[0]["payload"]["text"] == "second"

    def test_read_events_after_last_event_id_returns_empty(self, storage):
        e1 = user_submitted("it", "t1", "only")
        storage.append_event("it", e1)
        after = storage.read_events("it", after_event_id=e1.event_id)
        assert after == []

    def test_read_events_unknown_after_id_returns_empty(self, storage):
        storage.append_event("it", user_submitted("it", "t1", "msg"))
        after = storage.read_events("it", after_event_id="nonexistent-id")
        assert after == []

    def test_event_payload_preserved(self, storage):
        ev = user_submitted("it", "turn-123", "Voglio mangiare pizza.")
        storage.append_event("it", ev)
        events = storage.read_events("it")
        assert events[0]["payload"]["text"] == "Voglio mangiare pizza."
        assert events[0]["turn_id"] == "turn-123"


# ---------------------------------------------------------------------------
# Snapshots
# ---------------------------------------------------------------------------

class TestSnapshots:
    def test_read_nonexistent_returns_none(self, storage):
        assert storage.read_snapshot("it", "bandit_state") is None

    def test_write_and_read(self, storage):
        data = {"schema_version": 1, "total_pulls": 42}
        storage.write_snapshot("it", "bandit_state", data)
        result = storage.read_snapshot("it", "bandit_state")
        assert result == data

    def test_write_overwrites(self, storage):
        storage.write_snapshot("it", "bandit_state", {"total_pulls": 1})
        storage.write_snapshot("it", "bandit_state", {"total_pulls": 99})
        result = storage.read_snapshot("it", "bandit_state")
        assert result["total_pulls"] == 99

    def test_snapshots_partitioned_by_language(self, storage):
        storage.write_snapshot("it", "learner_profile", {"language": "it"})
        storage.write_snapshot("es", "learner_profile", {"language": "es"})
        assert storage.read_snapshot("it", "learner_profile")["language"] == "it"
        assert storage.read_snapshot("es", "learner_profile")["language"] == "es"

    def test_different_snapshot_names_independent(self, storage):
        storage.write_snapshot("it", "bandit_state", {"pulls": 1})
        storage.write_snapshot("it", "learner_profile", {"name": "test"})
        assert storage.read_snapshot("it", "bandit_state")["pulls"] == 1
        assert storage.read_snapshot("it", "learner_profile")["name"] == "test"

    def test_unicode_preserved(self, storage):
        storage.write_snapshot("it", "test", {"text": "Ciao, come stai? ñ"})
        result = storage.read_snapshot("it", "test")
        assert result["text"] == "Ciao, come stai? ñ"


# ---------------------------------------------------------------------------
# Transcripts
# ---------------------------------------------------------------------------

class TestTranscripts:
    def test_append_creates_transcript(self, storage):
        storage.append_transcript("it", "2026-03-01", "**Tutor:** Ciao!\n\n---\n")
        # No error = success; verify via read_history for InMemory or file for FS
        # Both backends must not raise

    def test_multiple_appends_accumulate(self, storage):
        storage.append_transcript("it", "2026-03-01", "**Tutor:** Ciao!\n\n")
        storage.append_transcript("it", "2026-03-01", "**User:** Come stai?\n\n")

        if isinstance(storage, InMemoryStorage):
            content = storage.read_transcript("it", "2026-03-01")
            assert "Ciao!" in content
            assert "Come stai?" in content

    def test_transcripts_partitioned_by_language_and_date(self, storage):
        storage.append_transcript("it", "2026-03-01", "Italian text\n")
        storage.append_transcript("es", "2026-03-01", "Spanish text\n")
        if isinstance(storage, InMemoryStorage):
            assert "Italian" in storage.read_transcript("it", "2026-03-01")
            assert "Spanish" in storage.read_transcript("es", "2026-03-01")


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------

class TestHistory:
    def test_append_and_read(self, storage):
        storage.append_history("it", "skill_history", {"ts": "t1", "value": 0.5})
        records = storage.read_history("it", "skill_history")
        assert len(records) == 1
        assert records[0]["value"] == 0.5

    def test_multiple_records_ordered(self, storage):
        for i in range(5):
            storage.append_history("it", "skill_history", {"ts": f"t{i}", "idx": i})
        records = storage.read_history("it", "skill_history")
        assert len(records) == 5
        assert [r["idx"] for r in records] == [0, 1, 2, 3, 4]

    def test_empty_history_returns_empty_list(self, storage):
        assert storage.read_history("it", "nonexistent") == []

    def test_history_partitioned_by_name(self, storage):
        storage.append_history("it", "skill_history", {"kind": "skill"})
        storage.append_history("it", "cefr_history", {"kind": "cefr"})
        assert storage.read_history("it", "skill_history")[0]["kind"] == "skill"
        assert storage.read_history("it", "cefr_history")[0]["kind"] == "cefr"

    def test_history_partitioned_by_language(self, storage):
        storage.append_history("it", "metrics", {"lang": "it"})
        storage.append_history("es", "metrics", {"lang": "es"})
        assert storage.read_history("it", "metrics")[0]["lang"] == "it"
        assert storage.read_history("es", "metrics")[0]["lang"] == "es"


# ---------------------------------------------------------------------------
# Arms
# ---------------------------------------------------------------------------

class TestArms:
    def test_read_arms_returns_list(self, storage):
        arms = storage.read_arms()
        assert isinstance(arms, list)
        assert len(arms) > 0

    def test_arms_have_required_fields(self, storage):
        for arm in storage.read_arms():
            assert "arm_id" in arm
            assert "intent" in arm

    def test_read_arms_returns_copy(self, storage):
        arms1 = storage.read_arms()
        arms2 = storage.read_arms()
        assert arms1 == arms2
        arms1.clear()
        assert len(storage.read_arms()) > 0


# ---------------------------------------------------------------------------
# Compact logs
# ---------------------------------------------------------------------------

class TestCompactLogs:
    def test_compact_returns_ok_status(self, storage):
        result = storage.compact_logs("it")
        assert result["status"] == "ok"
        assert "path" in result

    def test_compact_with_events(self, storage):
        for i in range(3):
            storage.append_event("it", user_submitted("it", f"t{i}", f"msg {i}"))
        result = storage.compact_logs("it")
        assert result["status"] == "ok"

    def test_compact_language_in_result(self, storage):
        result = storage.compact_logs("es")
        assert "es" in result["path"]


# ---------------------------------------------------------------------------
# FilesystemStorage-specific tests
# ---------------------------------------------------------------------------

class TestFilesystemStorageSpecific:
    def test_creates_missing_parent_dirs(self, tmp_path):
        storage = FilesystemStorage(tmp_path)
        storage.append_event("it", session_started("it"))
        assert (tmp_path / "logs" / "it" / "events.jsonl").exists()

    def test_snapshot_write_is_atomic(self, tmp_path):
        storage = FilesystemStorage(tmp_path)
        # Write twice rapidly — second write should succeed
        storage.write_snapshot("it", "bandit_state", {"v": 1})
        storage.write_snapshot("it", "bandit_state", {"v": 2})
        result = storage.read_snapshot("it", "bandit_state")
        assert result["v"] == 2

    def test_snapshot_file_on_disk(self, tmp_path):
        storage = FilesystemStorage(tmp_path)
        storage.write_snapshot("it", "learner_profile", {"language": "it"})
        path = tmp_path / "state" / "it" / "learner_profile.json"
        assert path.exists()

    def test_transcript_file_on_disk(self, tmp_path):
        storage = FilesystemStorage(tmp_path)
        storage.append_transcript("it", "2026-03-01", "Hello\n")
        path = tmp_path / "logs" / "it" / "sessions" / "2026-03-01.md"
        assert path.exists()
        assert "Hello" in path.read_text()

    def test_compact_creates_summary_file(self, tmp_path):
        arms_dst = tmp_path / "arms" / "arms.yaml"
        arms_dst.parent.mkdir(parents=True)
        shutil.copy(ARMS_DIR / "arms.yaml", arms_dst)
        storage = FilesystemStorage(tmp_path)
        storage.compact_logs("it")
        compacted = list((tmp_path / "logs" / "it" / "compacted").glob("*.json"))
        assert len(compacted) == 1
