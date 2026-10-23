"""Synthesizer agent for combining the strongest drafts."""

from __future__ import annotations

from agents import AgentDraft, Critique
from agents.evaluator_agent import EvaluatedDraft
from agents.research_agent import ResearchBrief
from brain.output_formatter import format_output
from config import Settings, get_settings
from models import llm_client


def synthesizer_agent(
    problem: str,
    brief: ResearchBrief,
    ranked_drafts: list[EvaluatedDraft],
    critiques: list[Critique],
    settings: Settings | None = None,
) -> AgentDraft:
    """Combine the strongest agent outputs into one controlled synthesis."""
    if not ranked_drafts:
        raise ValueError("At least one ranked draft is required for synthesis.")

    active_settings = settings or get_settings()
    ranked_section = "\n\n".join(
        (
            f"Candidate {draft.rank}\n"
            f"Role: {draft.draft.role}\n"
            f"Internal score: {draft.evaluation.total_score}\n"
            f"Critique summary: {draft.critique_summary or 'none'}\n"
            f"Answer:\n{draft.draft.answer}"
        )
        for draft in ranked_drafts[:3]
    )
    critique_section = "\n".join(
        f"- {critique.target_agent}: {', '.join(critique.missing_points)}"
        for critique in critiques
    ) or "- No critique gaps were provided."
    content = llm_client.llm_call_for_purpose(
        active_settings,
        "refiner",
        system_prompt=(
            "You combine strong draft answers into one final candidate for a conversational AI system. "
            "Keep the answer natural, specific, and stronger than any individual draft."
        ),
        user_prompt=f"""
Create one synthesized answer to the user's question.

Question:
{problem}

Research summary:
{brief.summary}

Top candidate drafts:
{ranked_section}

Critique gaps to address:
{critique_section}

Rules:
- Merge the strongest ideas while fixing the main gaps.
- Do not mention scores, critiques, ranking, agents, or pipeline details.
- Answer in natural language as if speaking directly to the user.

Return only the synthesized answer.
""",
    )
    return AgentDraft(
        agent="synthesizer_agent",
        role="Synthesizer Agent",
        answer=format_output(content, response_style="detailed"),
        notes="Merged the strongest grounded reasoning and critique fixes.",
        context_sources=[item.source for item in brief.retrieved_context[:3]],
    )
