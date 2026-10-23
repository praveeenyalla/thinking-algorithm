"""Persistent JSONL storage helpers for managed memory."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)


def load_records(path: Path) -> list[dict[str, object]]:
    """Load JSONL records from disk."""
    if not path.exists():
        return []

    records: list[dict[str, object]] = []
    invalid_lines_found = False
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            invalid_lines_found = True
            logger.warning("Skipping malformed JSONL record in %s at line %s.", path, line_number)
            continue
        if not isinstance(payload, dict):
            invalid_lines_found = True
            logger.warning("Skipping non-object JSONL record in %s at line %s.", path, line_number)
            continue
        records.append(payload)

    if invalid_lines_found:
        save_records(path, records)
    return records


def save_records(path: Path, records: list[dict[str, object]]) -> None:
    """Write JSONL records to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        path.write_text("", encoding="utf-8")
        return

    payload = "\n".join(json.dumps(record, ensure_ascii=True) for record in records) + "\n"
    path.write_text(payload, encoding="utf-8")


def upsert_record(path: Path, record: dict[str, object], key: str = "id") -> bool:
    """Insert or replace a record by key."""
    records = load_records(path)
    record_key = record.get(key)
    replaced = False
    new_records: list[dict[str, object]] = []

    for existing in records:
        if existing.get(key) == record_key:
            new_records.append(record)
            replaced = True
        else:
            new_records.append(existing)

    if not replaced:
        new_records.append(record)

    save_records(path, new_records)
    return not replaced


def touch_records(path: Path, record_ids: set[str], timestamp: float) -> int:
    """Increment usage metadata for matching records."""
    if not record_ids or not path.exists():
        return 0

    records = load_records(path)
    touched = 0
    for record in records:
        if str(record.get("id", "")) not in record_ids:
            continue
        record["usage_count"] = int(record.get("usage_count", 0)) + 1
        record["last_accessed"] = timestamp
        touched += 1

    if touched:
        save_records(path, records)
    return touched


def delete_records(path: Path, predicate: Callable[[dict[str, object]], bool]) -> int:
    """Delete records that match the predicate."""
    records = load_records(path)
    kept: list[dict[str, object]] = []
    deleted = 0

    for record in records:
        if predicate(record):
            deleted += 1
            continue
        kept.append(record)

    if deleted:
        save_records(path, kept)
    return deleted
