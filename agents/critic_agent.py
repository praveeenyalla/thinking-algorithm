"""Critic agent for structured challenge of candidate answers."""

from __future__ import annotations

import json
from typing import Any

from agents import AgentDraft, Critique
from agents.research_agent import ResearchBrief
from config import Settings, get_settings
from models import llm_client


def critic_agent(
    problem: str,
    drafts: list[AgentDraft],
    brief: ResearchBrief,
    settings: Settings | None = None,
) -> list[Critique]:
    """Critique each candidate answer without taking over controller decisions."""
    if not drafts:
        return []

    active_settings = settings or get_settings()
    return [_llm_critic(problem.strip(), draft, brief, active_settings) for draft in drafts]


def _llm_critic(problem: str, draft: AgentDraft, brief: ResearchBrief, settings: Settings) -> Critique:
    joined_points = "\n".join(f"- {point}" for point in brief.key_points) or "- No strong points were retrieved."
    joined_risks = "\n".join(f"- {risk}" for risk in brief.risks) or "- No explicit risks were identified."
    content = llm_client.llm_call_for_purpose(
        settings,
        "judge",
        system_prompt=(
            "You critique draft answers for a conversational AI system. "
            "Be direct, specific, and return JSON only."
        ),
        user_prompt=f"""
Critique the draft answer below.

Question:
{problem}

Research summary:
{brief.summary}

Key points:
{joined_points}

Risks and uncertainties:
{joined_risks}

Draft answer:
{draft.answer}

Return JSON only with this shape:
{{
  "summary": "one-sentence critique summary",
  "strengths": ["strength 1", "strength 2"],
  "missing_points": ["gap 1", "gap 2", "gap 3"]
}}
""",
    )
    payload = _extract_json(content)
    strengths = [str(item).strip() for item in payload.get("strengths", []) if str(item).strip()][:3]
    missing_points = [str(item).strip() for item in payload.get("missing_points", []) if str(item).strip()][:3]
    if not missing_points:
        raise RuntimeError("Critic output did not include missing points.")
    return Critique(
        target_agent=draft.agent,
        summary=str(payload["summary"]).strip(),
        missing_points=missing_points,
        strengths=strengths,
    )


def _extract_json(content: str) -> dict[str, Any]:
    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.removeprefix("```json").removeprefix("```").strip()
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3].strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise RuntimeError(f"Could not extract JSON from critic output: {content!r}")
    try:
        return json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Critic output was not valid JSON: {content!r}") from exc
