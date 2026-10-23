"""Single-pass answer refinement for the thinking layer."""

from __future__ import annotations

from brain.output_formatter import format_output
from config import Settings, get_settings
from memory.retriever import RetrievedContext, retrieve_context
from models import llm_client
from thinking.evaluator import EvaluationResult


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
    response_style: str = "structured",
) -> str:
    """Improve a selected answer using one refinement pass."""
    active_settings = settings or get_settings()
    refinement_query = (
        f"{problem}\n{answer}\n{feedback}\n"
        "improve answer clarity depth examples usefulness"
    )
    contexts = retrieve_context(
        refinement_query,
        active_settings.knowledge_base_dir,
        top_k=active_settings.refiner_retrieval_top_k,
        settings=active_settings,
    )
    refined = _llm_refine_answer(problem, answer, feedback, contexts, response_style, active_settings)
    return format_output(refined, response_style=response_style)


def _llm_refine_answer(
    problem: str,
    answer: str,
    feedback: str,
    contexts: list[RetrievedContext],
    response_style: str,
    settings: Settings,
) -> str:
    prompt = _build_refinement_prompt(problem, answer, feedback, contexts, response_style)
    return llm_client.llm_call_for_purpose(
        settings,
        "refiner",
        system_prompt=(
            "You improve answers for a conversational AI system. "
            "Preserve the substance, remove generic phrasing, and produce a stronger answer with natural wording."
        ),
        user_prompt=prompt,
    )


def _build_refinement_prompt(
    problem: str,
    answer: str,
    feedback: str,
    contexts: list[RetrievedContext],
    response_style: str,
) -> str:
    joined_context = "\n\n".join(
        f"Source: {item.source}\nRelevance: {item.relevance:.3f}\nContent: {item.content}" for item in contexts
    ) or "No additional context was retrieved."
    return f"""
Improve the draft answer below.

User question:
{problem}

Current draft:
{answer}

Internal review feedback:
{feedback}

Relevant context:
{joined_context}

Target style:
{_style_guidance(response_style)}

Rules:
- Return only the improved answer.
- Do not mention the review process, scores, or internal system behavior.
- Remove generic filler and make the answer specific to the question.
- Use examples only if they help the user understand the answer.

Write the final refined answer.
"""


def _style_guidance(response_style: str) -> str:
    styles = {
        "simple": "Short, direct explanation with no headings.",
        "structured": "Clear explanation with small readable sections or bullets only when useful.",
        "detailed": "Detailed explanation with clean structure and relevant examples.",
    }
    return styles.get(response_style, styles["structured"])
