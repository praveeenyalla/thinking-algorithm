"""Generate structured candidate answers for a problem."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from config import Settings, get_settings
from modules.retriever import RetrievedContext, retrieve_context
from openai import OpenAI


@dataclass(frozen=True)
class PerspectiveSpec:
    """Generation instructions for a candidate perspective."""

    title: str
    intent: str
    focus: str


PERSPECTIVES = (
    PerspectiveSpec(
        title="Scientific Perspective",
        intent="Explain underlying mechanisms, causal factors, and evidence-based implications.",
        focus="Mechanisms, research-style explanation, and domain dynamics.",
    ),
    PerspectiveSpec(
        title="Logical Perspective",
        intent="Break the problem into assumptions, reasoning steps, tradeoffs, and conclusions.",
        focus="Structured reasoning, assumptions, and tradeoff analysis.",
    ),
    PerspectiveSpec(
        title="Practical Perspective",
        intent="Propose a real-world plan with actions, risks, and implementation steps.",
        focus="Actionable steps, constraints, and direct practical next moves.",
    ),
)


def generate_options(problem: str, settings: Settings | None = None) -> list[str]:
    """Generate multiple structured candidate answers."""
    cleaned_problem = problem.strip()
    if not cleaned_problem:
        raise ValueError("Problem must not be empty.")

    active_settings = settings or get_settings()
    retrieval_query = (
        f"{cleaned_problem}\n"
        "generation structure correctness clarity usefulness actionable reasoning judge high-scoring answer"
    )
    contexts = retrieve_context(
        retrieval_query,
        active_settings.knowledge_base_dir,
        top_k=active_settings.generator_retrieval_top_k,
    )

    options: list[str] = []
    for perspective in PERSPECTIVES:
        if active_settings.enable_llm_generator and active_settings.generator_api_key:
            try:
                options.append(_llm_generate_option(cleaned_problem, perspective, contexts, active_settings))
                continue
            except RuntimeError:
                pass

        options.append(_heuristic_generate_option(cleaned_problem, perspective, contexts))

    return options


def _llm_generate_option(
    problem: str,
    perspective: PerspectiveSpec,
    contexts: list[RetrievedContext],
    settings: Settings,
) -> str:
    client = OpenAI(
        base_url=settings.generator_base_url,
        api_key=settings.generator_api_key,
    )
    prompt = _build_generation_prompt(problem, perspective, contexts)

    try:
        completion = client.chat.completions.create(
            model=settings.generator_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You generate candidate answers for a reasoning engine. "
                        "Write concrete, high-scoring answers optimized for correctness, clarity, and usefulness."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=settings.generator_temperature,
            top_p=settings.generator_top_p,
            max_tokens=settings.generator_max_tokens,
        )
    except Exception as exc:
        raise RuntimeError(f"LLM generator call failed: {exc}") from exc

    try:
        content = completion.choices[0].message.content
    except (AttributeError, IndexError, TypeError) as exc:
        raise RuntimeError(f"Unexpected generator response: {completion}") from exc

    if isinstance(content, list):
        content = "".join(
            getattr(part, "text", "")
            for part in content
            if getattr(part, "type", None) == "text"
        )
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("Generator returned empty content.")

    return _normalize_generated_output(content.strip())


def _heuristic_generate_option(
    problem: str,
    perspective: PerspectiveSpec,
    contexts: list[RetrievedContext],
) -> str:
    guidance = _summarize_context(contexts)
    if perspective.title == "Scientific Perspective":
        return (
            f"{perspective.title}:\n"
            f"Summary: {problem} should be analyzed through underlying mechanisms, system drivers, and observable effects.\n"
            f"Analysis: Explain the main forces shaping the issue, show how they interact, and distinguish short-term signals from long-term structural change.\n"
            f"Implications: The strongest answer should connect causes to outcomes and explain why the problem matters in measurable terms.\n"
            f"Recommended next step: Validate the main mechanisms with grounded evidence and clarify the most important variables first.\n"
            f"Judge alignment: {guidance}"
        )
    if perspective.title == "Logical Perspective":
        return (
            f"{perspective.title}:\n"
            f"Summary: {problem} can be addressed by separating assumptions, reasoning steps, and competing conclusions.\n"
            f"Analysis: Clarify what must be true, compare alternatives, test for internal consistency, and remove weak or unsupported claims.\n"
            f"Implications: Choose the path that remains coherent under scrutiny while still being understandable to the user.\n"
            f"Recommended next step: Write down the core assumptions and test which one most changes the final conclusion.\n"
            f"Judge alignment: {guidance}"
        )
    return (
        f"{perspective.title}:\n"
        f"Summary: {problem} needs a practical response that turns analysis into execution.\n"
        f"Analysis: Define immediate steps, assign priorities, and sequence the work from fastest wins to higher-effort improvements.\n"
        f"Implications: Call out the main operational risks, constraints, and expected outcomes before implementation.\n"
        f"Recommended next step: Start with one measurable action, assign ownership, and review the result before expanding scope.\n"
        f"Judge alignment: {guidance}"
    )


def _build_generation_prompt(
    problem: str,
    perspective: PerspectiveSpec,
    contexts: list[RetrievedContext],
) -> str:
    joined_context = "\n\n".join(
        f"Source: {item.source}\nRelevance: {item.relevance:.3f}\nContent: {item.content}" for item in contexts
    )
    return f"""
Create one candidate answer for the problem below.

Problem:
{problem}

Perspective:
{perspective.title}

Perspective intent:
{perspective.intent}

Focus:
{perspective.focus}

Retrieved guidance:
{joined_context}

Requirements:
- Optimize for correctness, clarity, and usefulness.
- Do not return a placeholder or title-only response.
- Make the answer self-contained and concrete.
- Use plain text section headers, not markdown bold.
- Use exactly this structure:
{perspective.title}:
Summary: ...
Analysis: ...
Implications: ...
Recommended next step: ...
- Keep it under 220 words.
"""


def _summarize_context(contexts: list[RetrievedContext]) -> str:
    sources = ", ".join(item.source for item in contexts)
    return f"Use rubric-grounded reasoning drawn from {sources}."


def _normalize_generated_output(text: str) -> str:
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = text.replace("\r\n", "\n")
    return text.strip()
