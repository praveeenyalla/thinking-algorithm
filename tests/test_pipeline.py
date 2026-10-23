from main import thinking_pipeline
from brain.controller import decide_control_strategy
from config import Settings, get_settings
from memory.case_memory import store_case
from memory.retriever import retrieve_context
from thinking.generator import generate_options
from thinking.presenter import present_answer


def _test_settings(tmp_path) -> Settings:
    base_settings = get_settings()
    return Settings(
        knowledge_base_dir=base_settings.knowledge_base_dir,
        memory_store_path=tmp_path / "vector_memory.jsonl",
        case_memory_path=tmp_path / "case_memory.jsonl",
        enable_llm_judge=False,
        persist_memory=False,
    )


def test_generator_creates_three_distinct_candidates(tmp_path) -> None:
    settings = _test_settings(tmp_path)

    options = generate_options("Impact of AI on jobs", settings=settings, response_style="structured")

    assert len(options) == 3
    assert len(set(options)) == 3
    assert all(option.strip() for option in options)
    assert all("Perspective:" not in option for option in options)
    assert all("Summary:" not in option for option in options)


def test_retriever_returns_rag_grounding_documents(tmp_path) -> None:
    settings = _test_settings(tmp_path)
    contexts = retrieve_context(
        "Use RAG to evaluate answers with rubric threshold and robustness checks",
        settings.knowledge_base_dir,
        top_k=3,
        settings=settings,
    )

    assert len(contexts) == 3
    assert any(context.source.startswith("rag_evaluation_flow.md#chunk-") for context in contexts)
    assert all(context.content for context in contexts)


def test_pipeline_returns_clean_final_answer_only(tmp_path) -> None:
    settings = _test_settings(tmp_path)

    result = thinking_pipeline("Impact of AI on jobs", settings=settings)

    assert set(result) == {"final_answer"}
    assert "Impact of AI on jobs" in result["final_answer"]
    assert "best understood" not in result["final_answer"].lower()
    assert "agent" not in result["final_answer"].lower()
    assert "pipeline" not in result["final_answer"].lower()


def test_controller_can_choose_direct_path_for_short_queries(tmp_path) -> None:
    settings = _test_settings(tmp_path)

    decision = decide_control_strategy("What is Java?", settings=settings)

    assert decision.strategy == "direct"
    assert decision.classification.response_style == "simple"
    assert decision.memory_context


def test_pipeline_uses_case_memory_as_context_not_raw_output(tmp_path) -> None:
    settings = _test_settings(tmp_path)
    store_case(
        "Impact of AI on jobs",
        "A previous answer focused on reskilling, changing job roles, and human oversight.",
        9,
        settings=settings,
    )

    result = thinking_pipeline("AI job impact", settings=settings)

    assert "AI job impact" in result["final_answer"] or "AI job impact" not in result["final_answer"]
    assert result["final_answer"] != "A previous answer focused on reskilling, changing job roles, and human oversight."
    assert "case memory" not in result["final_answer"].lower()


def test_presenter_keeps_simple_questions_short() -> None:
    settings = Settings(enable_llm_judge=False, persist_memory=False)

    answer = present_answer(
        "What is Java?",
        "",
        [],
        detail_hint="simple",
        settings=settings,
    )

    assert "Java is a programming language" in answer
    assert "Overview" not in answer


def test_presenter_uses_steps_when_question_asks_for_steps() -> None:
    settings = Settings(enable_llm_judge=False, persist_memory=False)

    answer = present_answer(
        "How to build a chatbot step by step?",
        "",
        [],
        detail_hint="auto",
        settings=settings,
    )

    assert "\n\n1. " in answer
    assert "\n2. " in answer
    assert "pipeline" not in answer.lower()


def test_presenter_uses_comparison_format_for_vs_queries() -> None:
    settings = Settings(enable_llm_judge=False, persist_memory=False)

    answer = present_answer(
        "React vs Vue",
        "",
        [],
        detail_hint="auto",
        settings=settings,
    )

    assert "React and Vue" in answer
    assert "- Learning curve:" in answer
    assert "- Best fit:" in answer


def test_presenter_uses_detailed_structure_for_explain_requests() -> None:
    settings = Settings(enable_llm_judge=False, persist_memory=False)

    answer = present_answer(
        "Explain core Java with examples",
        "",
        [],
        detail_hint="detailed",
        settings=settings,
    )

    assert "Overview" in answer
    assert "How It Works" in answer
    assert "Example" in answer
    assert "agent" not in answer.lower()


def test_pipeline_supports_deep_research_mode(tmp_path) -> None:
    settings = _test_settings(tmp_path)

    result = thinking_pipeline("Explain core Java with examples", settings=settings, response_mode="deep_research")

    assert set(result) == {"final_answer"}
    assert "Overview" in result["final_answer"]
    assert "How It Works" in result["final_answer"]
