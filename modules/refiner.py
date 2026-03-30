"""Single-pass answer refinement for Thinking Algorithm v3."""

from __future__ import annotations

from config import Settings, get_settings
from modules.evaluator import EvaluationResult
from modules.retriever import RetrievedContext, retrieve_context
from openai import OpenAI


def format_feedback(evaluation: EvaluationResult) -> str:
    """Turn structured evaluation into compact feedback text for refinement."""
    criteria_lines = "\n".join(
        f"- {criterion.name}: {criterion.score}/10 -> {criterion.rationale}"
        for criterion in evaluation.criteria
    )
    return (
        f"Overall score: {evaluation.total_score}/10\n"
        f"Threshold passed: {evaluation.threshold_passed}\n"
        f"Summary: {evaluation.summary_reason}\n"
        f"Criteria:\n{criteria_lines}"
    )


def refine_answer(
    problem: str,
    answer: str,
    feedback: str,
    settings: Settings | None = None,
) -> str:
    """Improve a selected answer using one refinement pass."""
    active_settings = settings or get_settings()
    refinement_query = (
        f"{problem}\n{answer}\n{feedback}\n"
        "refine improve answer correctness clarity usefulness single-pass iteration"
    )
    contexts = retrieve_context(
        refinement_query,
        active_settings.knowledge_base_dir,
        top_k=active_settings.refiner_retrieval_top_k,
    )

    if active_settings.enable_llm_refiner and active_settings.refiner_api_key:
        try:
            return _llm_refine_answer(problem, answer, feedback, contexts, active_settings)
        except RuntimeError:
            pass

    return _heuristic_refine_answer(problem, answer, feedback, contexts)


def _llm_refine_answer(
    problem: str,
    answer: str,
    feedback: str,
    contexts: list[RetrievedContext],
    settings: Settings,
) -> str:
    client = OpenAI(
        base_url=settings.refiner_base_url,
        api_key=settings.refiner_api_key,
    )
    prompt = _build_refinement_prompt(problem, answer, feedback, contexts)

    try:
        response = client.chat.completions.create(
            model=settings.refiner_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You improve answers for a reasoning engine. "
                        "Return one stronger structured answer only, with no extra explanation."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=settings.refiner_temperature,
            top_p=settings.refiner_top_p,
            max_tokens=settings.refiner_max_tokens,
        )
    except Exception as exc:
        raise RuntimeError(f"LLM refiner call failed: {exc}") from exc

    try:
        content = response.choices[0].message.content
    except (AttributeError, IndexError, TypeError) as exc:
        raise RuntimeError(f"Unexpected refiner response: {response}") from exc

    if isinstance(content, list):
        content = "".join(
            getattr(part, "text", "")
            for part in content
            if getattr(part, "type", None) == "text"
        )
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("Refiner returned empty content.")
    return content.strip()


def _heuristic_refine_answer(
    problem: str,
    answer: str,
    feedback: str,
    contexts: list[RetrievedContext],
) -> str:
    first_line = answer.splitlines()[0].strip() if answer.splitlines() else "Refined Perspective:"
    perspective = first_line if first_line.endswith(":") else f"{first_line}:"
    guidance = ", ".join(item.source for item in contexts)
    return (
        f"{perspective}\n"
        f"Summary: {problem} should be answered with a clearer explanation of what is happening, why it matters, and what practical decision follows.\n"
        f"Analysis: Improve the original answer by making the reasoning explicit, tightening weak claims, and connecting the response more directly to the user's real decision.\n"
        f"Implications: Highlight the main tradeoffs, likely constraints, and expected outcome so the answer is easier to trust and act on.\n"
        f"Recommended next step: Start with one measurable action, define who owns it, and review the result against the strongest risk before expanding scope.\n"
        f"Execution checkpoints: name the first milestone, define the review window, and state what evidence would count as progress.\n"
        f"Success signal: the answer is stronger when it is more specific, more actionable, and better aligned with the evaluation feedback.\n"
        f"Refinement feedback used: {feedback}\n"
        f"Refinement guidance: grounded by {guidance}."
    )


def _build_refinement_prompt(
    problem: str,
    answer: str,
    feedback: str,
    contexts: list[RetrievedContext],
) -> str:
    joined_context = "\n\n".join(
        f"Source: {item.source}\nRelevance: {item.relevance:.3f}\nContent: {item.content}" for item in contexts
    )
    return f"""
Improve the answer below using the evaluation feedback.

Problem:
{problem}

Original answer:
{answer}

Evaluation feedback:
{feedback}

Retrieved refinement guidance:
{joined_context}

Rules:
- Perform exactly one refinement pass.
- Do not explain what you changed.
- Return only the improved answer.
- Keep the same perspective title.
- Use plain text.
- Use this structure:
Perspective:
Summary: ...
Analysis: ...
Implications: ...
Recommended next step: ...
Execution checkpoints: ...
Success signal: ...
"""
