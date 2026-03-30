"""Persistence helpers for lightweight run feedback."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path


@dataclass(frozen=True)
class FeedbackRecord:
    """A stored summary of one pipeline run."""

    problem: str
    strategy: str
    final_score: int
    improved: bool
    final_answer: str
    recorded_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


def append_feedback_record(record: FeedbackRecord, store_path: Path) -> None:
    """Append a feedback record to the local JSONL store."""
    store_path.parent.mkdir(parents=True, exist_ok=True)
    with store_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(asdict(record), ensure_ascii=True) + "\n")


def load_feedback_records(store_path: Path) -> list[FeedbackRecord]:
    """Load feedback records from the local JSONL store."""
    if not store_path.exists():
        return []

    records: list[FeedbackRecord] = []
    for line in store_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        records.append(FeedbackRecord(**json.loads(line)))
    return records
