"""Evaluator agent wrapper that reuses the v6 scoring system."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agents import AgentDraft, Critique
from config import Settings, get_settings
from thinking.evaluator import EvaluationResult, evaluate_options


@dataclass(frozen=True)
class EvaluatedDraft:
    """A candidate answer paired with its critique and evaluation."""

    draft: AgentDraft
    evaluation: EvaluationResult
    critique_summary: str
    rank: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "draft": self.draft.to_dict(),
            "evaluation": self.evaluation.to_dict(),
            "critique_summary": self.critique_summary,
            "rank": self.rank,
        }


def evaluator_agent(
    problem: str,
    drafts: list[AgentDraft],
    critiques: list[Critique],
    settings: Settings | None = None,
) -> list[EvaluatedDraft]:
    """Score candidate answers using the existing evaluation module."""
    if not drafts:
        raise ValueError("At least one draft is required for evaluation.")

    active_settings = settings or get_settings()
    evaluations = evaluate_options(problem, [draft.answer for draft in drafts], settings=active_settings)
    critique_map = {critique.target_agent: critique for critique in critiques}

    combined = [
        (
            draft,
            evaluation,
            critique_map.get(draft.agent),
        )
        for draft, evaluation in zip(drafts, evaluations)
    ]
    combined.sort(key=lambda item: (item[1].total_score, int(item[1].threshold_passed)), reverse=True)

    ranked: list[EvaluatedDraft] = []
    for rank, (draft, evaluation, critique) in enumerate(combined, start=1):
        ranked.append(
            EvaluatedDraft(
                draft=draft,
                evaluation=evaluation,
                critique_summary=critique.summary if critique else "",
                rank=rank,
            )
        )
    return ranked
