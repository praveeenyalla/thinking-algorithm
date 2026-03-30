"""Selection logic for the thinking layer."""

from __future__ import annotations


def select_best(options: list[str], scores: list[int]) -> str:
    """Return the highest-scored option."""
    if not options or not scores or len(options) != len(scores):
        raise ValueError("Options and scores must be non-empty and aligned.")
    return options[scores.index(max(scores))]
