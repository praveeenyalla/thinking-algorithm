"""Prompt-strategy summarization over stored feedback."""

from __future__ import annotations

from collections import Counter

from learning.feedback_store import FeedbackRecord


def summarize_feedback_patterns(records: list[FeedbackRecord]) -> dict[str, object]:
    """Summarize which controller patterns have been performing well."""
    if not records:
        return {
            "record_count": 0,
            "average_score": 0.0,
            "best_strategy": None,
            "guidance": "No feedback history is available yet.",
        }

    average_score = round(sum(record.final_score for record in records) / len(records), 2)
    successful = [record for record in records if record.final_score >= 8]
    strategy_counts = Counter(record.strategy for record in successful or records)
    best_strategy = strategy_counts.most_common(1)[0][0]
    guidance = (
        f"Favor the '{best_strategy}' controller path when the query resembles previously successful runs."
    )
    return {
        "record_count": len(records),
        "average_score": average_score,
        "best_strategy": best_strategy,
        "guidance": guidance,
    }
