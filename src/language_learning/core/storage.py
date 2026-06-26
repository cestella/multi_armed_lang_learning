"""Storage primitives: atomic writes, JSONL append/read, JSON read/write."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path


def atomic_write(path: str | Path, data: str) -> None:
    """Write data atomically: write to temp file, fsync, rename."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.rename(tmp_path, path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def append_jsonl(path: str | Path, record: dict) -> None:
    """Append one JSON line to a JSONL file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def read_jsonl(path: str | Path) -> list[dict]:
    """Read all lines from a JSONL file. Returns empty list if file doesn't exist."""
    path = Path(path)
    if not path.exists():
        return []
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def read_json(path: str | Path) -> dict | None:
    """Read a JSON file. Returns None if file doesn't exist."""
    path = Path(path)
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def write_json(path: str | Path, data: dict) -> None:
    """Write a dict as JSON atomically."""
    atomic_write(path, json.dumps(data, indent=2, ensure_ascii=False) + "\n")
