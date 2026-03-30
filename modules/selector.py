"""Choose the strongest candidate answer."""

from __future__ import annotations


def select_best(options: list[str], scores: list[int]) -> str:
    """Select the option with the highest score."""
    if not options:
        raise ValueError("At least one option is required for selection.")
    if len(options) != len(scores):
        raise ValueError("Options and scores must have the same length.")

    max_index = scores.index(max(scores))
    return options[max_index]
