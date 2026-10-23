"""Query complexity classification for adaptive response behavior."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class QueryClassification:
    """Adaptive response settings derived from the user's prompt."""

    response_style: str
    reasoning_strategy: str
    wants_examples: bool
    response_mode: str


def classify_query(problem: str, response_mode: str = "chat") -> QueryClassification:
    """Classify a user question into response and reasoning modes."""
    lowered = problem.strip().lower()
    token_count = len(lowered.split())

    wants_examples = any(marker in lowered for marker in ("example", "examples", "for instance", "sample"))
    simple_starters = ("what is", "what's", "who is", "define", "meaning of")
    structured_markers = (
        "explain",
        "how",
        "why",
        "compare",
        "difference",
        "vs",
        "versus",
        "step by step",
        "steps",
        "build",
        "create",
        "implement",
    )
    detailed_markers = (
        "detailed",
        "in detail",
        "deep",
        "deep research",
        "technical",
        "architecture",
        "system design",
        "design",
        "research",
        "thorough",
        "comprehensive",
    )

    if response_mode == "deep_research":
        return QueryClassification(
            response_style="detailed",
            reasoning_strategy="deliberate",
            wants_examples=True,
            response_mode=response_mode,
        )

    is_simple = token_count <= 8 and lowered.startswith(simple_starters) and not any(
        marker in lowered for marker in detailed_markers + structured_markers
    )
    if is_simple:
        return QueryClassification(
            response_style="simple",
            reasoning_strategy="direct",
            wants_examples=wants_examples,
            response_mode=response_mode,
        )

    is_detailed = token_count > 18 or any(marker in lowered for marker in detailed_markers)
    if is_detailed:
        return QueryClassification(
            response_style="detailed",
            reasoning_strategy="deliberate",
            wants_examples=True,
            response_mode=response_mode,
        )

    return QueryClassification(
        response_style="structured" if any(marker in lowered for marker in structured_markers) or token_count > 8 else "simple",
        reasoning_strategy="deliberate" if any(marker in lowered for marker in structured_markers) or token_count > 8 else "direct",
        wants_examples=wants_examples or "explain" in lowered,
        response_mode=response_mode,
    )
