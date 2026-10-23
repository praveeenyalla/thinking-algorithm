"""LLM-backed candidate generation for controller fallbacks."""

from __future__ import annotations

from dataclasses import dataclass

from brain.output_formatter import format_output
from config import Settings, get_settings
from memory.retriever import RetrievedContext, retrieve_context
from models import llm_client


@dataclass(frozen=True)
class PerspectiveSpec:
    """Generation instructions for one candidate angle."""

    name: str
    instruction: str


PERSPECTIVES = (
    PerspectiveSpec(
        name="grounded_explanation",
        instruction="Answer directly, explain the core concept clearly, and stay grounded in the available context.",
    ),
    PerspectiveSpec(
        name="analytical_reasoning",
        instruction="Break the question into the main parts, assumptions, and tradeoffs before answering.",
    ),
    PerspectiveSpec(
        name="practical_guidance",
        instruction="Keep the answer useful in practice with concrete examples or actionable guidance when relevant.",
    ),
)


def generate_options(
    problem: str,
    settings: Settings | None = None,
    seed_context: list[RetrievedContext] | None = None,
    response_style: str = "structured",
) -> list[str]:
    """Generate multiple model-backed candidate answers."""
    cleaned_problem = problem.strip()
    if not cleaned_problem:
        raise ValueError("Problem must not be empty.")

    active_settings = settings or get_settings()
    retrieval_query = (
        f"{cleaned_problem}\n"
        "explanation examples tradeoffs practical answer correctness clarity usefulness"
    )
    contexts = retrieve_context(
        retrieval_query,
        active_settings.knowledge_base_dir,
        top_k=active_settings.generator_retrieval_top_k,
        settings=active_settings,
    )
    contexts = _merge_contexts(seed_context or [], contexts)

    return [
        _generate_option(cleaned_problem, perspective, contexts, response_style, active_settings)
        for perspective in PERSPECTIVES
    ]


def _generate_option(
    problem: str,
    perspective: PerspectiveSpec,
    contexts: list[RetrievedContext],
    response_style: str,
    settings: Settings,
) -> str:
    joined_context = "\n\n".join(
        f"Source: {item.source}\nRelevance: {item.relevance:.3f}\nContent: {item.content}"
        for item in contexts
    ) or "No external context was retrieved."
    content = llm_client.llm_call_for_purpose(
        settings,
        "generator",
        system_prompt=(
            "You write distinct candidate answers for a conversational AI system. "
            "The answer must be specific to the user's question, natural to read, and free of internal system language."
        ),
        user_prompt=f"""
Create one candidate answer for the user's question.

Question:
{problem}

Angle:
{perspective.instruction}

Desired response style:
{_style_guidance(response_style)}

Available context:
{joined_context}

Rules:
- Do not mention agents, scores, evaluation, retrieval, memory, or pipeline internals.
- Do not use labels such as "Summary", "Analysis", or "Recommended next step".
- Answer the actual question instead of describing how you would answer it.
- Keep the wording natural and question-specific.

Return only the answer text.
""",
    )
    return format_output(content, response_style=response_style)


def _style_guidance(response_style: str) -> str:
    styles = {
        "simple": "Short, direct explanation with no headings.",
        "structured": "Clear explanation with light structure or bullets only when helpful.",
        "detailed": "Detailed explanation with clean sections only if they improve readability.",
    }
    return styles.get(response_style, styles["structured"])


def _merge_contexts(
    seed_context: list[RetrievedContext],
    retrieved_context: list[RetrievedContext],
) -> list[RetrievedContext]:
    combined = seed_context + retrieved_context
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
