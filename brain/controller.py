"""Control layer that orchestrates the AI brain system."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from agents import AgentDraft
from agents.creative_agent import creative_agent
from agents.critic_agent import critic_agent
from agents.evaluator_agent import evaluator_agent
from agents.logic_agent import logic_agent
from agents.research_agent import build_research_brief, research_agent
from agents.synthesizer_agent import synthesizer_agent
from brain.query_classifier import QueryClassification, classify_query
from config import Settings, get_settings
from learning.feedback_store import FeedbackRecord, append_feedback_record
from memory.case_memory import RetrievedCase, retrieve_case, store_case
from memory.cleanup import clean_memory
from memory.retriever import RetrievedContext, clear_retrieval_cache, remember_text, retrieve_context
from thinking.evaluator import EvaluationResult, evaluate_options
from thinking.generator import generate_options
from thinking.presenter import present_answer
from thinking.refiner import format_feedback, refine_answer
from thinking.selector import select_best
from utils.logger import get_logger

logger = get_logger(__name__)
RESPONSE_MODES = {"chat", "deep_research"}


@dataclass(frozen=True)
class ControlDecision:
    """The controller's decision about how to run the pipeline."""

    strategy: str
    reason: str
    classification: QueryClassification
    memory_context: list[RetrievedContext]

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "reason": self.reason,
            "response_style": self.classification.response_style,
            "response_mode": self.classification.response_mode,
            "wants_examples": self.classification.wants_examples,
            "memory_context": [item.to_dict() for item in self.memory_context],
        }


def decide_control_strategy(
    problem: str,
    settings: Settings | None = None,
    response_mode: str = "chat",
) -> ControlDecision:
    """Choose how much reasoning depth to use for a problem."""
    active_settings = settings or get_settings()
    cleaned_problem = problem.strip()
    memory_context = retrieve_context(
        cleaned_problem,
        active_settings.knowledge_base_dir,
        top_k=active_settings.controller_memory_top_k,
        settings=active_settings,
    )
    classification = classify_query(cleaned_problem, response_mode=response_mode)
    reason = (
        "The question is simple enough for a direct conversational answer."
        if classification.reasoning_strategy == "direct"
        else "The question benefits from the deliberate multi-agent reasoning path."
    )
    return ControlDecision(
        strategy=classification.reasoning_strategy,
        reason=reason,
        classification=classification,
        memory_context=memory_context,
    )


def thinking_pipeline(
    problem: str,
    settings: Settings | None = None,
    response_mode: str = "chat",
) -> dict[str, Any]:
    """Run the controller-led AI brain pipeline."""
    cleaned_problem = problem.strip()
    if not cleaned_problem:
        raise ValueError("Problem must not be empty.")

    active_settings = settings or get_settings()
    normalized_mode = response_mode if response_mode in RESPONSE_MODES else "chat"
    logger.info("Starting controller pipeline for problem: %s", cleaned_problem)

    cleanup_result = clean_memory(active_settings)
    if cleanup_result["episodic_deleted"] or cleanup_result["case_deleted"]:
        clear_retrieval_cache()

    decision = decide_control_strategy(cleaned_problem, active_settings, response_mode=normalized_mode)
    matched_case = retrieve_case(cleaned_problem, settings=active_settings)
    context = _merge_contexts(decision.memory_context, _case_contexts(matched_case))

    draft_answer = ""
    draft_evaluation: EvaluationResult | None = None
    if decision.strategy == "deliberate":
        if active_settings.enable_multi_agent_deliberate:
            draft_answer, draft_evaluation = _run_multi_agent_deliberate_pipeline(
                cleaned_problem,
                context,
                matched_case.answer if matched_case else None,
                active_settings,
            )
        else:
            draft_answer, draft_evaluation = _run_standard_generation_cycle(
                cleaned_problem,
                context,
                matched_case.answer if matched_case else None,
                active_settings,
                decision.classification.response_style,
            )
        draft_answer = refine_answer(
            cleaned_problem,
            draft_answer,
            format_feedback(draft_evaluation),
            settings=active_settings,
            response_style=decision.classification.response_style,
        )

    final_answer = present_answer(
        cleaned_problem,
        draft_answer,
        context,
        detail_hint=decision.classification.response_style,
        response_mode=normalized_mode,
        settings=active_settings,
        classification=decision.classification,
    )
    final_evaluation = evaluate_options(cleaned_problem, [final_answer], settings=active_settings)[0]
    final_score = final_evaluation.total_score

    if active_settings.persist_feedback:
        append_feedback_record(
            FeedbackRecord(
                problem=cleaned_problem,
                strategy=decision.strategy,
                final_score=final_score,
                improved=decision.strategy == "deliberate",
                final_answer=final_answer,
            ),
            active_settings.feedback_store_path,
        )

    if active_settings.persist_memory:
        remember_text(
            final_answer,
            source=f"answer_memory::{cleaned_problem}",
            quality_score=final_score,
            settings=active_settings,
            metadata={
                "problem": cleaned_problem,
                "strategy": decision.strategy,
                "response_style": decision.classification.response_style,
            },
        )
        store_case(
            cleaned_problem,
            final_answer,
            final_score,
            settings=active_settings,
        )

    logger.info("Controller pipeline finished with strategy=%s score=%s", decision.strategy, final_score)
    return {"final_answer": final_answer}


def _run_standard_generation_cycle(
    problem: str,
    seed_context: list[RetrievedContext],
    matched_case_answer: str | None,
    settings: Settings,
    response_style: str,
) -> tuple[str, EvaluationResult]:
    options = generate_options(problem, settings=settings, seed_context=seed_context, response_style=response_style)
    if matched_case_answer and matched_case_answer not in options:
        options.append(matched_case_answer)
    evaluations, scores, best_option, best_evaluation = _evaluate_candidates(problem, options, settings)
    return best_option, best_evaluation


def _run_multi_agent_deliberate_pipeline(
    problem: str,
    seed_context: list[RetrievedContext],
    matched_case_answer: str | None,
    settings: Settings,
) -> tuple[str, EvaluationResult]:
    brief = build_research_brief(problem, seed_context=seed_context, settings=settings)

    with ThreadPoolExecutor(max_workers=3) as executor:
        research_future = executor.submit(research_agent, problem, brief, settings)
        logic_future = executor.submit(logic_agent, problem, brief, settings)
        creative_future = executor.submit(creative_agent, problem, brief, settings)
        drafts = [
            research_future.result(),
            logic_future.result(),
            creative_future.result(),
        ]

    if matched_case_answer and matched_case_answer not in {draft.answer for draft in drafts}:
        drafts.append(
            AgentDraft(
                agent="case_memory_agent",
                role="Case Memory",
                answer=matched_case_answer,
                notes="Similar past answer reused as supporting evidence.",
                context_sources=["case_memory"],
            )
        )

    critiques = critic_agent(problem, drafts, brief, settings=settings)
    ranked_drafts = evaluator_agent(problem, drafts, critiques, settings=settings)
    synthesized_draft = synthesizer_agent(problem, brief, ranked_drafts, critiques, settings=settings)

    candidate_answers = [item.draft.answer for item in ranked_drafts]
    if synthesized_draft.answer not in candidate_answers:
        candidate_answers.append(synthesized_draft.answer)

    return _select_best_answer(problem, candidate_answers, settings)


def _select_best_answer(
    problem: str,
    options: list[str],
    settings: Settings,
) -> tuple[str, EvaluationResult]:
    evaluations, scores, best_option, best_evaluation = _evaluate_candidates(problem, options, settings)
    return best_option, best_evaluation


def _evaluate_candidates(
    problem: str,
    options: list[str],
    settings: Settings,
) -> tuple[list[EvaluationResult], list[int], str, EvaluationResult]:
    evaluations = evaluate_options(problem, options, settings=settings)
    scores = [evaluation.total_score for evaluation in evaluations]
    best_option = select_best(options, scores)
    best_index = options.index(best_option)
    return evaluations, scores, best_option, evaluations[best_index]


def _case_contexts(matched_case: RetrievedCase | None) -> list[RetrievedContext]:
    if matched_case is None:
        return []
    return [
        RetrievedContext(
            source="case_memory",
            content=(
                f"Similar past question: {matched_case.problem}\n"
                f"Reusable answer draft: {matched_case.answer}"
            ),
            relevance=matched_case.weighted_score,
            memory_type="case_memory",
            quality_score=float(matched_case.score),
            metadata={"problem": matched_case.problem},
        )
    ]


def _merge_contexts(*context_groups: list[RetrievedContext]) -> list[RetrievedContext]:
    combined: list[RetrievedContext] = []
    for group in context_groups:
        combined.extend(group)

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
