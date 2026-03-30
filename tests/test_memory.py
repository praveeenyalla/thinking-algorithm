import time

from config import Settings, get_settings
from memory.case_memory import retrieve_case, store_case
from memory.cleanup import clean_memory
from memory.chunker import chunk_text
from memory.persistent_store import load_records, save_records
from memory.retriever import remember_text, retrieve_context


def test_chunker_creates_overlapping_chunks() -> None:
    text = " ".join(f"word{i}" for i in range(1, 31))
    chunks = chunk_text(text, chunk_size=10, chunk_overlap=2)

    assert len(chunks) == 4
    assert chunks[0].split()[-2:] == chunks[1].split()[:2]


def test_retriever_can_return_saved_memory_chunks(tmp_path) -> None:
    base_settings = get_settings()
    settings = Settings(
        knowledge_base_dir=base_settings.knowledge_base_dir,
        memory_store_path=tmp_path / "vector_memory.jsonl",
        case_memory_path=tmp_path / "case_memory.jsonl",
        enable_llm_generator=False,
        enable_llm_judge=False,
        enable_llm_refiner=False,
        persist_memory=True,
        persist_feedback=False,
    )

    stored_count = remember_text(
        "AI hiring safety checklist should include bias review, human escalation, and audit logs.",
        source="memory-test",
        quality_score=9,
        settings=settings,
        metadata={"kind": "episodic_test"},
    )
    contexts = retrieve_context(
        "What checklist should I use for AI hiring bias review?",
        settings.knowledge_base_dir,
        top_k=5,
        settings=settings,
    )

    assert stored_count >= 1
    assert any(context.memory_type == "episodic_memory" for context in contexts)
    assert any("bias review" in context.content.lower() for context in contexts)
    stored_records = load_records(settings.memory_store_path)
    assert any(int(record.get("usage_count", 0)) >= 1 for record in stored_records)


def test_low_quality_memory_is_not_stored(tmp_path) -> None:
    base_settings = get_settings()
    settings = Settings(
        knowledge_base_dir=base_settings.knowledge_base_dir,
        memory_store_path=tmp_path / "vector_memory.jsonl",
        case_memory_path=tmp_path / "case_memory.jsonl",
        persist_memory=True,
    )

    stored_count = remember_text(
        "Weak generic answer with no real utility.",
        source="low-quality-test",
        quality_score=5,
        settings=settings,
    )

    assert stored_count == 0
    assert not settings.memory_store_path.exists()


def test_case_memory_retrieves_similar_high_quality_solution(tmp_path) -> None:
    base_settings = get_settings()
    settings = Settings(
        knowledge_base_dir=base_settings.knowledge_base_dir,
        memory_store_path=tmp_path / "vector_memory.jsonl",
        case_memory_path=tmp_path / "case_memory.jsonl",
        persist_memory=True,
    )

    stored = store_case(
        "Impact of AI on jobs",
        "Practical Perspective:\nSummary: Use reskilling, role redesign, and human oversight.",
        9,
        settings=settings,
    )
    matched_case = retrieve_case("AI job impact", settings=settings)

    assert stored is True
    assert matched_case is not None
    assert matched_case.score == 9
    assert matched_case.similarity >= settings.case_reuse_similarity_threshold
    case_records = load_records(settings.case_memory_path)
    assert any(int(record.get("usage_count", 0)) >= 1 for record in case_records)


def test_cleanup_removes_old_unused_memory_and_cases(tmp_path) -> None:
    base_settings = get_settings()
    settings = Settings(
        knowledge_base_dir=base_settings.knowledge_base_dir,
        memory_store_path=tmp_path / "vector_memory.jsonl",
        case_memory_path=tmp_path / "case_memory.jsonl",
        memory_retention_days=30,
        case_retention_days=30,
        memory_min_usage_to_keep=1,
        case_min_usage_to_keep=1,
        persist_memory=True,
    )

    remember_text(
        "Persistent memory that should expire because it is old and unused.",
        source="cleanup-test",
        quality_score=9,
        settings=settings,
    )
    store_case(
        "Old AI governance checklist",
        "Practical Perspective:\nSummary: Old retained case.",
        9,
        settings=settings,
    )

    old_timestamp = time.time() - (90 * 86400)
    episodic_records = load_records(settings.memory_store_path)
    for record in episodic_records:
        record["timestamp"] = old_timestamp
        record["usage_count"] = 0
    save_records(settings.memory_store_path, episodic_records)

    case_records = load_records(settings.case_memory_path)
    for record in case_records:
        record["timestamp"] = old_timestamp
        record["usage_count"] = 0
    save_records(settings.case_memory_path, case_records)

    cleanup_result = clean_memory(settings)

    assert cleanup_result["episodic_deleted"] >= 1
    assert cleanup_result["case_deleted"] >= 1
    assert load_records(settings.memory_store_path) == []
    assert load_records(settings.case_memory_path) == []
