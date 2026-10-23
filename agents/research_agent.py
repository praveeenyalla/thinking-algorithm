"""Research agent for grounded context gathering and synthesis."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

from agents import AgentDraft
from brain.output_formatter import format_output
from config import Settings, get_settings
from memory.retriever import RetrievedContext, retrieve_context
from models import llm_client


@dataclass(frozen=True)
class ResearchBrief:
    """Condensed research grounding for downstream agents."""

    problem: str
    summary: str
    key_points: list[str]
    risks: list[str]
    retrieved_context: list[RetrievedContext]

    def to_dict(self) -> dict[str, object]:
        return {
            "problem": self.problem,
            "summary": self.summary,
            "key_points": self.key_points,
            "risks": self.risks,
            "retrieved_context": [item.to_dict() for item in self.retrieved_context],
        }


def build_research_brief(
    problem: str,
    seed_context: list[RetrievedContext] | None = None,
    settings: Settings | None = None,
) -> ResearchBrief:
    """Retrieve and condense context for the deliberate multi-agent loop."""
    cleaned_problem = problem.strip()
    if not cleaned_problem:
        raise ValueError("Problem must not be empty.")

    active_settings = settings or get_settings()
    retrieved = retrieve_context(
        f"{cleaned_problem}\nresearch context evidence tradeoffs constraints decisions",
        active_settings.knowledge_base_dir,
        top_k=max(active_settings.controller_memory_top_k + 1, active_settings.generator_retrieval_top_k),
        settings=active_settings,
    )
    contexts = _merge_contexts(seed_context or [], retrieved)
    return _llm_build_research_brief(cleaned_problem, contexts, active_settings)


def research_agent(
    problem: str,
    brief: ResearchBrief,
    settings: Settings | None = None,
) -> AgentDraft:
    """Produce a grounded answer candidate from the research brief."""
    active_settings = settings or get_settings()
    joined_points = "\n".join(f"- {point}" for point in brief.key_points) or "- No strong points were retrieved."
    joined_risks = "\n".join(f"- {risk}" for risk in brief.risks) or "- No explicit risks were identified."
    content = llm_client.llm_call_for_purpose(
        active_settings,
        "generator",
        system_prompt=(
            "You produce a grounded candidate answer for a conversational AI system. "
            "Stay factual, acknowledge uncertainty when needed, and answer the user's actual question."
        ),
        user_prompt=f"""
Write one grounded candidate answer to the user's question.

Question:
{problem}

Research summary:
{brief.summary}

Key points:
{joined_points}

Risks and uncertainties:
{joined_risks}

Rules:
- Do not mention internal tools, agents, scores, or pipeline details.
- Keep the explanation natural and specific to the question.
- Use examples only if they improve understanding.

Return only the answer text.
""",
    )
    return AgentDraft(
        agent="research_agent",
        role="Research Agent",
        answer=format_output(content, response_style="detailed"),
        notes=brief.summary,
        context_sources=[item.source for item in brief.retrieved_context[:3]],
    )


def _llm_build_research_brief(
    problem: str,
    contexts: list[RetrievedContext],
    settings: Settings,
) -> ResearchBrief:
    joined_context = "\n\n".join(
        f"Source: {item.source}\nRelevance: {item.relevance:.3f}\nContent: {item.content}" for item in contexts
    ) or "No external context was retrieved."
    content = llm_client.llm_call_for_purpose(
        settings,
        "generator",
        system_prompt=(
            "You condense retrieved context into a research brief for downstream reasoning. "
            "Do not invent facts and return JSON only."
        ),
        user_prompt=f"""
Build a compact research brief for the user's question.

Question:
{problem}

Retrieved context:
{joined_context}

Return JSON only with this shape:
{{
  "summary": "short summary",
  "key_points": ["point 1", "point 2", "point 3"],
  "risks": ["risk 1", "risk 2", "risk 3"]
}}
""",
        temperature=0.2,
        top_p=1.0,
        max_tokens=500,
    )
    payload = _extract_json(content)
    return ResearchBrief(
        problem=problem,
        summary=str(payload["summary"]).strip(),
        key_points=[str(item).strip() for item in payload.get("key_points", []) if str(item).strip()][:3],
        risks=[str(item).strip() for item in payload.get("risks", []) if str(item).strip()][:3],
        retrieved_context=contexts,
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
        raise RuntimeError(f"Could not extract JSON from research brief output: {content!r}")
    try:
        return json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Research brief output was not valid JSON: {content!r}") from exc


def _merge_contexts(
    primary: list[RetrievedContext],
    secondary: list[RetrievedContext],
) -> list[RetrievedContext]:
    combined = primary + secondary
    deduped: list[RetrievedContext] = []
    seen: set[tuple[str, str]] = set()
    for item in combined:
        key = (item.source, item.content)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    deduped.sort(key=lambda item: (-item.relevance, item.source))
    return deduped
