"""Cleanup routines for persistent episodic and case memory."""

from __future__ import annotations

import time

from config import Settings, get_settings
from memory.persistent_store import delete_records


def clean_memory(settings: Settings | None = None) -> dict[str, int]:
    """Prune low-quality and stale memory records from disk."""
    active_settings = settings or get_settings()
    if not active_settings.enable_memory_cleanup:
        return {"episodic_deleted": 0, "case_deleted": 0}

    now = time.time()
    episodic_deleted = delete_records(
        active_settings.memory_store_path,
        lambda record: _should_delete_episodic(record, active_settings, now),
    )
    case_deleted = delete_records(
        active_settings.case_memory_path,
        lambda record: _should_delete_case(record, active_settings, now),
    )
    return {
        "episodic_deleted": episodic_deleted,
        "case_deleted": case_deleted,
    }


def _should_delete_episodic(record: dict[str, object], settings: Settings, now: float) -> bool:
    quality = float(record.get("quality_score", 0.0))
    if quality < settings.memory_quality_threshold:
        return True

    age_days = _age_days(now, float(record.get("timestamp", 0.0)))
    usage_count = int(record.get("usage_count", 0))
    if age_days > settings.memory_retention_days and usage_count < settings.memory_min_usage_to_keep:
        return True
    return False


def _should_delete_case(record: dict[str, object], settings: Settings, now: float) -> bool:
    score = int(record.get("score", 0))
    if score < settings.case_memory_score_threshold:
        return True

    age_days = _age_days(now, float(record.get("timestamp", 0.0)))
    usage_count = int(record.get("usage_count", 0))
    if age_days > settings.case_retention_days and usage_count < settings.case_min_usage_to_keep:
        return True
    return False


def _age_days(now: float, timestamp: float) -> float:
    return (now - timestamp) / 86400 if timestamp else float("inf")
