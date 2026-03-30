"""Memory retrieval and write-back for knowledge and episodic memory."""

from __future__ import annotations

from dataclasses import replace
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
import time

from config import Settings, get_settings
from memory.persistent_store import touch_records
from memory.vector_store import PersistentVectorStore, _stable_document_id, append_memory_text


@dataclass(frozen=True)
class RetrievedContext:
    """A retrieved memory item used to ground the pipeline."""

    source: str
    content: str
    relevance: float
    memory_type: str
    quality_score: float
    metadata: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "source": self.source,
            "content": self.content,
            "relevance": round(self.relevance, 3),
            "memory_type": self.memory_type,
            "quality_score": round(self.quality_score, 2),
            "metadata": self.metadata,
        }


def retrieve_context(
    query: str,
    knowledge_base_dir: Path,
    top_k: int = 3,
    settings: Settings | None = None,
) -> list[RetrievedContext]:
    """Retrieve the most relevant knowledge and memory chunks for a query."""
    active_settings = settings or get_settings()
    store = _load_store(
        str(knowledge_base_dir),
        str(active_settings.memory_store_path),
        active_settings.memory_chunk_size,
        active_settings.memory_chunk_overlap,
        active_settings.enable_remote_embeddings,
        active_settings.embedding_model,
        active_settings.embedding_base_url,
        active_settings.embedding_api_key,
    )
    expanded_query = (
        f"{query} correctness clarity usefulness threshold rubric rag robustness "
        "judge scorer evaluator generation refinement controller memory"
    )
    results = [
        RetrievedContext(
            source=document.source,
            content=document.content,
            relevance=score,
            memory_type=document.memory_type,
            quality_score=document.quality_score,
            metadata=document.metadata,
        )
        for document, score in store.search(expanded_query, settings=active_settings, top_k=top_k)
    ]
    _touch_retrieved_memory(results, active_settings)
    return results


def remember_text(
    text: str,
    *,
    source: str,
    quality_score: float,
    settings: Settings | None = None,
    metadata: dict[str, str] | None = None,
) -> int:
    """Persist text into episodic memory and invalidate the retrieval cache."""
    active_settings = settings or get_settings()
    stored_count = append_memory_text(
        text,
        source=source,
        memory_store_path=active_settings.memory_store_path,
        settings=active_settings,
        quality_score=quality_score,
        metadata=metadata,
    )
    _load_store.cache_clear()
    return stored_count


def clear_retrieval_cache() -> None:
    """Clear the cached retrieval store after lifecycle changes."""
    _load_store.cache_clear()


@lru_cache(maxsize=8)
def _load_store(
    knowledge_base_dir: str,
    memory_store_path: str,
    chunk_size: int,
    chunk_overlap: int,
    enable_remote_embeddings: bool,
    embedding_model: str,
    embedding_base_url: str,
    embedding_api_key: str | None,
) -> PersistentVectorStore:
    base_settings = get_settings()
    store_settings = replace(
        base_settings,
        knowledge_base_dir=Path(knowledge_base_dir),
        memory_store_path=Path(memory_store_path),
        memory_chunk_size=chunk_size,
        memory_chunk_overlap=chunk_overlap,
        enable_remote_embeddings=enable_remote_embeddings,
        embedding_model=embedding_model,
        embedding_base_url=embedding_base_url,
        embedding_api_key=embedding_api_key,
    )
    return PersistentVectorStore.from_sources(
        knowledge_base_dir=Path(knowledge_base_dir),
        memory_store_path=Path(memory_store_path),
        settings=store_settings,
    )


def _touch_retrieved_memory(results: list[RetrievedContext], settings: Settings) -> None:
    record_ids: set[str] = set()
    for item in results:
        if item.memory_type != "episodic_memory":
            continue
        source = item.source.split("#chunk-", maxsplit=1)[0]
        chunk_number = 0
        if "#chunk-" in item.source:
            chunk_number = int(item.source.rsplit("#chunk-", maxsplit=1)[1]) - 1
        record_id = _stable_document_id(source, item.content, chunk_number)
        record_ids.add(record_id)

    if record_ids:
        touch_records(settings.memory_store_path, record_ids, time.time())
