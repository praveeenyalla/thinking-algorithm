"""Adaptive final answer presentation for end-user responses."""

from __future__ import annotations

from brain.output_formatter import format_output
from brain.query_classifier import QueryClassification, classify_query
from config import Settings, get_settings
from memory.retriever import RetrievedContext
from models import llm_client


def present_answer(
    problem: str,
    base_answer: str,
    contexts: list[RetrievedContext],
    *,
    detail_hint: str = "auto",
    response_mode: str = "chat",
    settings: Settings | None = None,
    classification: QueryClassification | None = None,
) -> str:
    """Render the final user-facing answer with adaptive style."""
    active_settings = settings or get_settings()
    resolved_classification = _resolve_classification(problem, response_mode, detail_hint, classification)
    joined_context = "\n\n".join(
        f"Source: {item.source}\nRelevance: {item.relevance:.3f}\nContent: {item.content}"
        for item in contexts
    ) or "No external context was retrieved."
    draft_section = base_answer.strip() or "No draft answer is available. Answer from the question and the available context."

    content = llm_client.llm_call_for_purpose(
        active_settings,
        "refiner",
        system_prompt=(
            "You are an advanced conversational AI assistant. "
            "Your answer is the only text the user sees. "
            "Be natural, specific, and helpful. "
            "Never mention internal scoring, evaluation, reasoning chains, agents, controller logic, or pipeline details."
        ),
        user_prompt=f"""
Answer the user's question in the requested style.

User question:
{problem}

Response style:
{_style_guidance(resolved_classification)}

Available context:
{joined_context}

Draft material to use if helpful:
{draft_section}

Rules:
- Answer exactly what the user asked.
- Do not sound templated or generic.
- Do not mention internal memory, retrieval, scores, evaluation, agents, or pipeline details.
- Use examples only if they help the user understand the topic.
- For simple questions, keep the answer short and avoid headings.
- For structured questions, keep the structure light and readable.
- For detailed or deep research questions, provide a fuller explanation with clear sections only when that improves readability.

Return only the final answer.
""",
    )
    return format_output(content, response_style=resolved_classification.response_style)


def _resolve_classification(
    problem: str,
    response_mode: str,
    detail_hint: str,
    classification: QueryClassification | None,
) -> QueryClassification:
    resolved = classification or classify_query(problem, response_mode=response_mode)
    if detail_hint == "simple":
        return QueryClassification(
            response_style="simple",
            reasoning_strategy=resolved.reasoning_strategy,
            wants_examples=resolved.wants_examples,
            response_mode=resolved.response_mode,
        )
    if detail_hint in {"detailed", "research"}:
        return QueryClassification(
            response_style="detailed",
            reasoning_strategy=resolved.reasoning_strategy,
            wants_examples=True,
            response_mode=resolved.response_mode,
        )
    return resolved


def _style_guidance(classification: QueryClassification) -> str:
    styles = {
        "simple": "Give a short, direct answer in plain language with no headings.",
        "structured": "Give a clear explanation with light structure, short paragraphs, and bullets only when helpful.",
        "detailed": (
            "Give a detailed explanation. "
            "Use sections only when they genuinely improve readability, especially for technical or research-heavy requests."
        ),
    }
    example_rule = "Include useful examples." if classification.wants_examples else "Use examples only when clearly helpful."
    return f"{styles[classification.response_style]} {example_rule}"
