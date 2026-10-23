"""Creative agent for alternative framings and experiments."""

from __future__ import annotations

from agents import AgentDraft
from agents.research_agent import ResearchBrief
from brain.output_formatter import format_output
from config import Settings, get_settings
from models import llm_client


def creative_agent(
    problem: str,
    brief: ResearchBrief,
    settings: Settings | None = None,
) -> AgentDraft:
    """Produce a bounded creative candidate that still respects the research brief."""
    active_settings = settings or get_settings()
    points = "\n".join(f"- {point}" for point in brief.key_points) or "- No strong points were retrieved."
    risks = "\n".join(f"- {risk}" for risk in brief.risks) or "- No explicit risks were identified."
    content = llm_client.llm_call_for_purpose(
        active_settings,
        "generator",
        system_prompt=(
            "You write a creative but responsible candidate answer for a conversational AI system. "
            "Expand the option space without drifting into fantasy or generic filler."
        ),
        user_prompt=f"""
Write one creative but practical answer to the user's question.

Question:
{problem}

Research summary:
{brief.summary}

Key points:
{points}

Risks and uncertainties:
{risks}

Rules:
- Offer at least one useful angle, example, or alternative framing the user may not have considered.
- Stay grounded in the available context and realistic constraints.
- Do not mention internal tools, agents, scores, or pipeline details.
- Keep the answer readable and useful.

Return only the answer text.
""",
    )
    return AgentDraft(
        agent="creative_agent",
        role="Creative Agent",
        answer=format_output(content, response_style="structured"),
        notes="Alternative framing that stays grounded and practical.",
        context_sources=[item.source for item in brief.retrieved_context[:3]],
    )
