from main import thinking_pipeline
from brain.controller import decide_control_strategy
from config import Settings, get_settings
from memory.case_memory import store_case
from memory.retriever import retrieve_context
from thinking.evaluator import evaluate_options
from thinking.generator import generate_options
from thinking.selector import select_best


def test_generator_creates_three_perspectives() -> None:
    settings = Settings(enable_llm_generator=False, enable_llm_judge=False)
    options = generate_options("Impact of AI on jobs", settings=settings)

    assert len(options) == 3
    assert options[0].startswith("Scientific Perspective:")
    assert options[1].startswith("Logical Perspective:")
    assert options[2].startswith("Practical Perspective:")
    assert all("Summary:" in option for option in options)
    assert all("Recommended next step:" in option for option in options)


def test_retriever_returns_rag_grounding_documents(tmp_path) -> None:
    base_settings = get_settings()
    settings = Settings(
        knowledge_base_dir=base_settings.knowledge_base_dir,
        memory_store_path=tmp_path / "vector_memory.jsonl",
        case_memory_path=tmp_path / "case_memory.jsonl",
    )
    contexts = retrieve_context(
        "Use RAG to evaluate answers with rubric threshold and robustness checks",
        settings.knowledge_base_dir,
        top_k=3,
        settings=settings,
    )

    assert len(contexts) == 3
    assert any(context.source.startswith("rag_evaluation_flow.md#chunk-") for context in contexts)
    assert all(context.content for context in contexts)


def test_pipeline_selects_the_highest_scored_option(tmp_path) -> None:
    base_settings = get_settings()
    settings = Settings(
        knowledge_base_dir=base_settings.knowledge_base_dir,
        memory_store_path=tmp_path / "vector_memory.jsonl",
        case_memory_path=tmp_path / "case_memory.jsonl",
        enable_llm_generator=False,
        enable_llm_judge=False,
        enable_llm_refiner=False,
        persist_memory=False,
    )
    result = thinking_pipeline("Impact of AI on jobs", settings=settings)

    assert result["controller"]["strategy"] == "deliberate"
    assert result["controller"]["memory_context"]
    assert result["memory_cleanup"] == {"episodic_deleted": 0, "case_deleted": 0}
    assert result["case_reused"] is False
    assert len(result["options"]) == 3
    assert len(result["scores"]) == 3
    assert len(result["evaluations"]) == 3
    assert result["best_option"] == select_best(result["options"], result["scores"])
    evaluations = evaluate_options("Impact of AI on jobs", result["options"], settings=settings)
    assert result["scores"] == [evaluation.total_score for evaluation in evaluations]
    assert result["best_option"].startswith("Practical Perspective:")
    assert any(evaluation["threshold_passed"] for evaluation in result["evaluations"])
    assert all(len(evaluation["criteria"]) == 3 for evaluation in result["evaluations"])
    assert all(evaluation["retrieved_context"] for evaluation in result["evaluations"])
    assert "Execution checkpoints:" in result["refined_answer"]
    assert "Success signal:" in result["refined_answer"]
    assert result["refined_evaluation"]["total_score"] >= result["best_evaluation"]["total_score"]
    assert result["final_score"] == max(
        result["best_evaluation"]["total_score"],
        result["refined_evaluation"]["total_score"],
    )
    assert result["final_answer"] in {result["best_option"], result["refined_answer"]}


def test_controller_can_choose_direct_path_for_short_queries(tmp_path) -> None:
    base_settings = get_settings()
    settings = Settings(
        knowledge_base_dir=base_settings.knowledge_base_dir,
        memory_store_path=tmp_path / "vector_memory.jsonl",
        case_memory_path=tmp_path / "case_memory.jsonl",
        enable_llm_generator=False,
        enable_llm_judge=False,
        enable_llm_refiner=False,
        persist_memory=False,
    )
    decision = decide_control_strategy("AI risks?", settings=settings)

    assert decision.strategy == "direct"
    assert decision.memory_context


def test_pipeline_can_reuse_high_quality_case_for_similar_short_query(tmp_path) -> None:
    base_settings = get_settings()
    settings = Settings(
        knowledge_base_dir=base_settings.knowledge_base_dir,
        memory_store_path=tmp_path / "vector_memory.jsonl",
        case_memory_path=tmp_path / "case_memory.jsonl",
        enable_llm_generator=False,
        enable_llm_judge=False,
        enable_llm_refiner=False,
        persist_memory=False,
    )
    store_case(
        "Impact of AI on jobs",
        "Practical Perspective:\nSummary: Reuse strong workforce transition plans.\nAnalysis: Match the old solution to the new query.\nImplications: Faster answer reuse reduces wasted work.\nRecommended next step: Apply the past plan and update it only if the context changes.",
        9,
        settings=settings,
    )

    result = thinking_pipeline("AI job impact", settings=settings)

    assert result["controller"]["strategy"] == "direct"
    assert result["case_reused"] is True
    assert result["case_memory"] is not None
    assert len(result["options"]) == 1
    assert result["final_answer"].startswith("Practical Perspective:")
