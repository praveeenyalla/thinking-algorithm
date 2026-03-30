"""RAG-grounded evaluator for candidate answers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from config import Settings, get_settings
from modules.retriever import RetrievedContext, retrieve_context
from openai import OpenAI


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
        contexts = retrieve_context(query, active_settings.knowledge_base_dir, top_k=active_settings.retrieval_top_k)
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
    profile = _perspective_profile(answer)
    problem_tokens = _tokenize(problem)
    answer_tokens = _tokenize(answer)
    lowered_answer = answer.lower()

    problem_overlap = len(problem_tokens.intersection(answer_tokens)) / max(len(problem_tokens), 1)
    relevance_bonus = 1 if contexts else 0

    refinement_correctness_bonus = 1 if "refinement feedback used:" in lowered_answer else 0
    refinement_clarity_bonus = 2 if "execution checkpoints:" in lowered_answer else 0
    refinement_usefulness_bonus = 2 if "success signal:" in lowered_answer else 0

    correctness = _clamp(
        5 + round(problem_overlap * 2) + profile["correctness"] + relevance_bonus + refinement_correctness_bonus
    )
    clarity = _clamp(5 + profile["clarity"] + (1 if len(answer) <= 80 else 0) + refinement_clarity_bonus)
    usefulness = _clamp(
        4 + profile["usefulness"] + (1 if problem_overlap >= 0.8 else 0) + refinement_usefulness_bonus
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
    client = OpenAI(
        base_url=settings.judge_base_url,
        api_key=settings.judge_api_key,
    )

    try:
        response = client.chat.completions.create(
            model=settings.judge_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an LLM judge. Evaluate candidate answers using retrieved context and return JSON only."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            top_p=1,
            max_tokens=800,
        )
    except Exception as exc:
        raise RuntimeError(f"LLM judge call failed: {exc}") from exc

    try:
        content = response.choices[0].message.content
    except (AttributeError, IndexError, TypeError) as exc:
        raise RuntimeError(f"Unexpected judge response: {response}") from exc

    if isinstance(content, list):
        content = "".join(
            getattr(part, "text", "")
            for part in content
            if getattr(part, "type", None) == "text"
        )
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError(f"Judge response did not contain text content: {response}")

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


def _perspective_profile(answer: str) -> dict[str, int]:
    lowered = answer.lower()
    first_line = lowered.splitlines()[0] if lowered.splitlines() else lowered
    if "scientific explanation" in first_line or "scientific perspective" in first_line:
        return {"correctness": 1, "clarity": 0, "usefulness": 0}
    if "logical reasoning" in first_line or "logical perspective" in first_line:
        return {"correctness": 0, "clarity": 1, "usefulness": 1}
    if "practical solution" in first_line or "practical perspective" in first_line:
        return {"correctness": 0, "clarity": 0, "usefulness": 5}
    return {"correctness": 0, "clarity": 0, "usefulness": 0}


def _tokenize(text: str) -> set[str]:
    return {token for token in "".join(char.lower() if char.isalnum() else " " for char in text).split() if token}


def _clamp(value: int) -> int:
    return max(1, min(10, value))
