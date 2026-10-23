"""RAG-grounded evaluator for the thinking layer."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from config import Settings, get_settings
from memory.retriever import RetrievedContext, retrieve_context
from models import llm_client


@dataclass(frozen=True)
class CriterionEvaluation:
    """A score and rationale for a single evaluation criterion."""

    name: str
    score: int
    rationale: str

    def to_dict(self) -> dict[str, str | int]:
        return {
            "name": self.name,
            "score": self.score,
            "rationale": self.rationale,
        }


@dataclass(frozen=True)
class EvaluationResult:
    """Structured evaluation for one candidate answer."""

    answer: str
    total_score: int
    threshold_passed: bool
    judge_mode: str
    summary_reason: str
    criteria: list[CriterionEvaluation]
    retrieved_context: list[RetrievedContext]

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "total_score": self.total_score,
            "threshold_passed": self.threshold_passed,
            "judge_mode": self.judge_mode,
            "summary_reason": self.summary_reason,
            "criteria": [criterion.to_dict() for criterion in self.criteria],
            "retrieved_context": [item.to_dict() for item in self.retrieved_context],
        }


def evaluate_options(problem: str, options: list[str], settings: Settings | None = None) -> list[EvaluationResult]:
    """Evaluate all candidate answers with RAG-grounded judging."""
    if not options:
        raise ValueError("At least one option is required for evaluation.")

    active_settings = settings or get_settings()
    results: list[EvaluationResult] = []

    for option in options:
        query = f"{problem}\n{option}"
        contexts = retrieve_context(
            query,
            active_settings.knowledge_base_dir,
            top_k=active_settings.retrieval_top_k,
            settings=active_settings,
        )
        results.append(_judge_option(problem, option, contexts, active_settings))

    return results


def _judge_option(
    problem: str,
    answer: str,
    contexts: list[RetrievedContext],
    settings: Settings,
) -> EvaluationResult:
    if settings.enable_llm_judge and settings.judge_api_key:
        try:
            return _llm_judge(problem, answer, contexts, settings)
        except RuntimeError:
            pass

    return _heuristic_judge(problem, answer, contexts, settings)


def _heuristic_judge(
    problem: str,
    answer: str,
    contexts: list[RetrievedContext],
    settings: Settings,
) -> EvaluationResult:
    problem_tokens = _tokenize(problem)
    answer_tokens = _tokenize(answer)
    lowered_answer = answer.lower()

    problem_overlap = len(problem_tokens.intersection(answer_tokens)) / max(len(problem_tokens), 1)
    asks_for_steps = any(marker in problem.lower() for marker in ("how", "step", "build", "implement"))
    asks_for_comparison = any(marker in problem.lower() for marker in ("compare", "difference", "vs", "versus"))
    has_list_structure = bool(re.search(r"(^|\n)(- |\d+\. )", answer))
    has_example = "example" in lowered_answer or "for example" in lowered_answer
    has_internal_artifacts = any(
        marker in lowered_answer for marker in ("score:", "evaluation", "pipeline", "agent", "controller")
    )

    correctness = _clamp(4 + round(problem_overlap * 4) + (1 if contexts else 0) - int(has_internal_artifacts))
    clarity = _clamp(
        4
        + (1 if 20 <= len(answer.split()) <= 350 else 0)
        + (1 if len(answer.splitlines()) <= 14 else 0)
        + (1 if "." in answer else 0)
        - int(has_internal_artifacts)
    )
    usefulness = _clamp(
        4
        + (1 if problem_overlap >= 0.55 else 0)
        + (1 if has_example else 0)
        + (1 if has_list_structure and (asks_for_steps or asks_for_comparison) else 0)
        + (1 if contexts else 0)
        - int(has_internal_artifacts)
    )

    criteria = [
        CriterionEvaluation(
            name="correctness",
            score=correctness,
            rationale="Grounded against the problem statement and retrieved evaluation context.",
        ),
        CriterionEvaluation(
            name="clarity",
            score=clarity,
            rationale="Assessed for readable phrasing and directness.",
        ),
        CriterionEvaluation(
            name="usefulness",
            score=usefulness,
            rationale="Assessed for practical decision value and actionability.",
        ),
    ]

    weighted_total = round(
        sum(criterion.score * settings.criterion_weights[criterion.name] for criterion in criteria)
    )
    threshold_passed = weighted_total >= settings.evaluation_threshold
    sources = ", ".join(item.source for item in contexts)
    summary_reason = (
        f"Score derived from fine-grained rubric judging with RAG grounding. "
        f"Retrieved sources: {sources}. "
        f"Threshold {'passed' if threshold_passed else 'failed'} at {settings.evaluation_threshold}/10."
    )
    return EvaluationResult(
        answer=answer,
        total_score=weighted_total,
        threshold_passed=threshold_passed,
        judge_mode="heuristic_rag",
        summary_reason=summary_reason,
        criteria=criteria,
        retrieved_context=contexts,
    )


def _llm_judge(
    problem: str,
    answer: str,
    contexts: list[RetrievedContext],
    settings: Settings,
) -> EvaluationResult:
    prompt = _build_judge_prompt(problem, answer, contexts, settings.evaluation_threshold)
    content = llm_client.llm_call_for_purpose(
        settings,
        "judge",
        system_prompt="You are an LLM judge. Evaluate candidate answers using retrieved context and return JSON only.",
        user_prompt=prompt,
    )

    parsed = _extract_json(content)
    criteria = [
        CriterionEvaluation(
            name=name,
            score=_clamp(int(details["score"])),
            rationale=str(details["rationale"]).strip(),
        )
        for name, details in parsed["criteria"].items()
    ]
    return EvaluationResult(
        answer=answer,
        total_score=_clamp(int(parsed["total_score"])),
        threshold_passed=bool(parsed["threshold_passed"]),
        judge_mode="llm_rag",
        summary_reason=str(parsed["summary_reason"]).strip(),
        criteria=criteria,
        retrieved_context=contexts,
    )


def _build_judge_prompt(
    problem: str,
    answer: str,
    contexts: list[RetrievedContext],
    threshold: int,
) -> str:
    joined_context = "\n\n".join(
        f"Source: {item.source}\nRelevance: {item.relevance:.3f}\nContent: {item.content}" for item in contexts
    )
    return f"""
Evaluate this candidate answer using the retrieved context.

Criteria:
- correctness
- clarity
- usefulness

Problem:
{problem}

Candidate answer:
{answer}

Retrieved context:
{joined_context}

Threshold:
{threshold}

Return JSON only using this shape:
{{
  "criteria": {{
    "correctness": {{"score": 1, "rationale": "short reason"}},
    "clarity": {{"score": 1, "rationale": "short reason"}},
    "usefulness": {{"score": 1, "rationale": "short reason"}}
  }},
  "total_score": 1,
  "threshold_passed": false,
  "summary_reason": "short reason"
}}
"""


def _extract_json(content: str) -> dict[str, Any]:
    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.removeprefix("```json").removeprefix("```").strip()
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3].strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise RuntimeError(f"Could not extract JSON from judge output: {content!r}")
    try:
        return json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Judge output was not valid JSON: {content!r}") from exc


def _tokenize(text: str) -> set[str]:
    return {token for token in "".join(char.lower() if char.isalnum() else " " for char in text).split() if token}


def _clamp(value: int) -> int:
    return max(1, min(10, value))
