"""Tests for storage primitives."""

import json

from language_learning.core.storage import (
    append_jsonl,
    atomic_write,
    read_json,
    read_jsonl,
    write_json,
)


class TestAtomicWrite:
    def test_writes_content(self, tmp_path):
        path = tmp_path / "test.txt"
        atomic_write(path, "hello world")
        assert path.read_text() == "hello world"

    def test_creates_parent_dirs(self, tmp_path):
        path = tmp_path / "a" / "b" / "test.txt"
        atomic_write(path, "nested")
        assert path.read_text() == "nested"

    def test_overwrites_existing(self, tmp_path):
        path = tmp_path / "test.txt"
        atomic_write(path, "first")
        atomic_write(path, "second")
        assert path.read_text() == "second"


class TestJsonl:
    def test_append_and_read(self, tmp_path):
        path = tmp_path / "events.jsonl"
        append_jsonl(path, {"type": "a", "value": 1})
        append_jsonl(path, {"type": "b", "value": 2})
        records = read_jsonl(path)
        assert len(records) == 2
        assert records[0]["type"] == "a"
        assert records[1]["value"] == 2

    def test_read_missing_file(self, tmp_path):
        path = tmp_path / "nonexistent.jsonl"
        assert read_jsonl(path) == []

    def test_creates_parent_dirs(self, tmp_path):
        path = tmp_path / "deep" / "dir" / "events.jsonl"
        append_jsonl(path, {"k": "v"})
        assert read_jsonl(path) == [{"k": "v"}]


class TestJson:
    def test_read_missing_file(self, tmp_path):
        path = tmp_path / "nonexistent.json"
        assert read_json(path) is None

    def test_write_and_read(self, tmp_path):
        path = tmp_path / "state.json"
        data = {"schema_version": 1, "arms": {"a": {"n": 5}}}
        write_json(path, data)
        result = read_json(path)
        assert result == data

    def test_write_creates_parent_dirs(self, tmp_path):
        path = tmp_path / "state" / "it" / "bandit.json"
        write_json(path, {"test": True})
        assert read_json(path) == {"test": True}

    def test_unicode_preserved(self, tmp_path):
        path = tmp_path / "test.json"
        data = {"text": "Ciao, come stai?"}
        write_json(path, data)
        result = read_json(path)
        assert result["text"] == "Ciao, come stai?"
