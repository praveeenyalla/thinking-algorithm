"""Control layer that orchestrates the AI brain system."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from config import Settings, get_settings
from learning.feedback_store import FeedbackRecord, append_feedback_record
from memory.case_memory import retrieve_case, store_case
from memory.cleanup import clean_memory
from memory.retriever import RetrievedContext, clear_retrieval_cache, remember_text, retrieve_context
from thinking.evaluator import evaluate_options
from thinking.generator import generate_options
from thinking.refiner import format_feedback, refine_answer
from thinking.selector import select_best
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class ControlDecision:
    """The controller's decision about how to run the pipeline."""

    strategy: str
    reason: str
    memory_context: list[RetrievedContext]

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "reason": self.reason,
            "memory_context": [item.to_dict() for item in self.memory_context],
        }


def decide_control_strategy(problem: str, settings: Settings | None = None) -> ControlDecision:
    """Choose how much reasoning depth to use for a problem."""
    active_settings = settings or get_settings()
    cleaned_problem = problem.strip()
    memory_context = retrieve_context(
        cleaned_problem,
        active_settings.knowledge_base_dir,
        top_k=active_settings.controller_memory_top_k,
        settings=active_settings,
    )
    token_count = len(cleaned_problem.split())
    lowered_problem = cleaned_problem.lower()
    complex_markers = {
        "architecture",
        "compare",
        "design",
        "improve",
        "plan",
        "research",
        "strategy",
        "tradeoff",
    }
    has_complex_marker = any(marker in lowered_problem for marker in complex_markers)

    if token_count <= active_settings.quick_query_token_limit and not has_complex_marker:
        return ControlDecision(
            strategy="direct",
            reason="Short query with low structural complexity, so the controller skips refinement.",
            memory_context=memory_context,
        )

    return ControlDecision(
        strategy="deliberate",
        reason="Query is substantial enough to justify full generate-evaluate-refine reasoning.",
        memory_context=memory_context,
    )


def thinking_pipeline(problem: str, settings: Settings | None = None) -> dict[str, Any]:
    """Run the controller-led AI brain pipeline."""
    active_settings = settings or get_settings()
    logger.info("Starting controller pipeline for problem: %s", problem)

    cleanup_result = clean_memory(active_settings)
    if cleanup_result["episodic_deleted"] or cleanup_result["case_deleted"]:
        clear_retrieval_cache()
    decision = decide_control_strategy(problem, active_settings)
    matched_case = retrieve_case(problem, settings=active_settings)
    case_reused = False

    if matched_case and decision.strategy == "direct":
        options = [matched_case.answer]
        evaluations = evaluate_options(problem, options, settings=active_settings)
        scores = [evaluation.total_score for evaluation in evaluations]
        best_option = matched_case.answer
        best_evaluation = evaluations[0]
        case_reused = True
    else:
        options = generate_options(problem, settings=active_settings, seed_context=decision.memory_context)
        if matched_case and matched_case.answer not in options:
            options.append(matched_case.answer)
        evaluations = evaluate_options(problem, options, settings=active_settings)
        scores = [evaluation.total_score for evaluation in evaluations]
        best_option = select_best(options, scores)
        best_index = options.index(best_option)
        best_evaluation = evaluations[best_index]

    result: dict[str, Any] = {
        "problem": problem,
        "controller": decision.to_dict(),
        "memory_cleanup": cleanup_result,
        "case_memory": matched_case.to_dict() if matched_case else None,
        "case_reused": case_reused,
        "options": options,
        "scores": scores,
        "evaluations": [evaluation.to_dict() for evaluation in evaluations],
        "best_option": best_option,
        "best_evaluation": best_evaluation.to_dict(),
        "best_feedback": format_feedback(best_evaluation),
    }

    if decision.strategy == "deliberate" and not case_reused:
        refined_answer = refine_answer(problem, best_option, result["best_feedback"], settings=active_settings)
        refined_evaluation = evaluate_options(problem, [refined_answer], settings=active_settings)[0]
        improved = refined_evaluation.total_score > best_evaluation.total_score
        final_answer = refined_answer if improved else best_option
        final_evaluation = refined_evaluation if improved else best_evaluation
        result.update(
            {
                "refined_answer": refined_answer,
                "refined_evaluation": refined_evaluation.to_dict(),
                "improved": improved,
                "final_answer": final_answer,
                "final_score": final_evaluation.total_score,
                "final_evaluation": final_evaluation.to_dict(),
            }
        )
    else:
        result.update(
            {
                "refined_answer": best_option,
                "refined_evaluation": best_evaluation.to_dict(),
                "improved": False,
                "final_answer": best_option,
                "final_score": best_evaluation.total_score,
                "final_evaluation": best_evaluation.to_dict(),
            }
        )

    if active_settings.persist_feedback:
        append_feedback_record(
            FeedbackRecord(
                problem=problem,
                strategy=decision.strategy,
                final_score=result["final_score"],
                improved=result["improved"],
                final_answer=result["final_answer"],
            ),
            active_settings.feedback_store_path,
        )

    stored_memory_chunks = 0
    stored_case = False
    if active_settings.persist_memory:
        stored_memory_chunks = remember_text(
            result["final_answer"],
            source=f"answer_memory::{problem}",
            quality_score=result["final_score"],
            settings=active_settings,
            metadata={
                "problem": problem,
                "strategy": decision.strategy,
                "score": str(result["final_score"]),
            },
        )
        stored_case = store_case(
            problem,
            result["final_answer"],
            result["final_score"],
            settings=active_settings,
        )
    result["stored_memory_chunks"] = stored_memory_chunks
    result["stored_case"] = stored_case

    logger.info("Controller pipeline finished with strategy=%s score=%s", decision.strategy, result["final_score"])
    return result
