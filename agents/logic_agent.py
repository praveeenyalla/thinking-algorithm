"""Logic agent for structured reasoning."""

from __future__ import annotations

from agents import AgentDraft
from agents.research_agent import ResearchBrief
from brain.output_formatter import format_output
from config import Settings, get_settings
from models import llm_client


def logic_agent(
    problem: str,
    brief: ResearchBrief,
    settings: Settings | None = None,
) -> AgentDraft:
    """Produce a structured reasoning candidate grounded in the research brief."""
    active_settings = settings or get_settings()
    points = "\n".join(f"- {point}" for point in brief.key_points) or "- No strong points were retrieved."
    risks = "\n".join(f"- {risk}" for risk in brief.risks) or "- No explicit risks were identified."
    content = llm_client.llm_call_for_purpose(
        active_settings,
        "generator",
        system_prompt=(
            "You write a rigorous candidate answer for a conversational AI system. "
            "Focus on reasoning, assumptions, tradeoffs, and a clear conclusion."
        ),
        user_prompt=f"""
Write one logically strong answer to the user's question.

Question:
{problem}

Research summary:
{brief.summary}

Key points:
{points}

Risks and uncertainties:
{risks}

Rules:
- Break the issue into the main parts before answering.
- Keep the wording natural and readable.
- Do not mention internal tools, agents, scores, or pipeline details.
- Use bullets only if they genuinely help.

Return only the answer text.
""",
    )
    return AgentDraft(
        agent="logic_agent",
        role="Logic Agent",
        answer=format_output(content, response_style="structured"),
        notes="Structured decomposition of assumptions, tradeoffs, and conclusions.",
        context_sources=[item.source for item in brief.retrieved_context[:3]],
    )
