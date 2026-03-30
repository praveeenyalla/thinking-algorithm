"""Persistent vector store for knowledge and episodic memory chunks."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from pathlib import Path
import math
import time
from typing import Any

from config import Settings
from memory.chunker import chunk_text
from memory.embedder import EmbeddingVector, cosine_similarity, embedding_from_dict, get_embedding
from memory.persistent_store import load_records, upsert_record


@dataclass(frozen=True)
class VectorDocument:
    """A document chunk stored in the vector store."""

    id: str
    source: str
    content: str
    embedding: EmbeddingVector
    memory_type: str
    quality_score: float
    timestamp: float
    usage_count: int
    last_accessed: float
    metadata: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "source": self.source,
            "content": self.content,
            "embedding": self.embedding.to_dict(),
            "memory_type": self.memory_type,
            "quality_score": self.quality_score,
            "timestamp": self.timestamp,
            "usage_count": self.usage_count,
            "last_accessed": self.last_accessed,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "VectorDocument":
        return cls(
            id=str(payload["id"]),
            source=str(payload["source"]),
            content=str(payload["content"]),
            embedding=embedding_from_dict(payload["embedding"]),
            memory_type=str(payload["memory_type"]),
            quality_score=float(payload.get("quality_score", _default_quality_score(str(payload["memory_type"])))),
            timestamp=float(payload.get("timestamp", 0.0)),
            usage_count=int(payload.get("usage_count", 0)),
            last_accessed=float(payload.get("last_accessed", payload.get("timestamp", 0.0))),
            metadata={str(key): str(value) for key, value in payload.get("metadata", {}).items()},
        )


class PersistentVectorStore:
    """Local store that combines knowledge-base chunks and saved memory chunks."""

    def __init__(self, documents: list[VectorDocument]) -> None:
        self._documents = documents

    @classmethod
    def from_sources(
        cls,
        knowledge_base_dir: Path,
        memory_store_path: Path,
        settings: Settings,
    ) -> "PersistentVectorStore":
        """Load chunked knowledge documents and remembered episodic chunks."""
        documents = _load_knowledge_documents(knowledge_base_dir, settings)
        documents.extend(_load_memory_documents(memory_store_path))
        return cls(documents)

    def search(self, query: str, settings: Settings, top_k: int) -> list[tuple[VectorDocument, float]]:
        """Return the top matching documents for a query."""
        query_embedding = get_embedding(query, settings=settings)
        scored: list[tuple[VectorDocument, float]] = []

        for document in self._documents:
            similarity = cosine_similarity(query_embedding, document.embedding)
            density_bonus = _density_bonus(query_embedding, document.embedding)
            memory_bonus = 0.15 if document.memory_type == "episodic_memory" else 0.0
            quality_weight = max(0.1, min(document.quality_score, 10.0) / 10.0)
            score = (similarity + density_bonus + memory_bonus) * quality_weight
            scored.append((document, score))

        scored.sort(key=lambda item: (-item[1], item[0].source))
        return scored[:top_k]


def append_memory_text(
    text: str,
    *,
    source: str,
    memory_store_path: Path,
    settings: Settings,
    quality_score: float,
    metadata: dict[str, str] | None = None,
) -> int:
    """Chunk, embed, and persist a text into episodic memory."""
    if quality_score < settings.memory_quality_threshold:
        return 0

    chunks = chunk_text(
        text,
        chunk_size=settings.memory_chunk_size,
        chunk_overlap=settings.memory_chunk_overlap,
    )
    if not chunks:
        return 0

    stored_count = 0
    for index, chunk in enumerate(chunks):
        now = time.time()
        document = VectorDocument(
            id=_stable_document_id(source, chunk, index),
            source=f"{source}#chunk-{index + 1}",
            content=chunk,
            embedding=get_embedding(chunk, settings=settings),
            memory_type="episodic_memory",
            quality_score=quality_score,
            timestamp=now,
            usage_count=0,
            last_accessed=now,
            metadata=metadata or {},
        )
        inserted = upsert_record(memory_store_path, document.to_dict())
        if inserted:
            stored_count += 1

    return stored_count


def _load_knowledge_documents(knowledge_base_dir: Path, settings: Settings) -> list[VectorDocument]:
    if not knowledge_base_dir.exists():
        raise FileNotFoundError(f"Knowledge base directory does not exist: {knowledge_base_dir}")

    documents: list[VectorDocument] = []
    for path in sorted(knowledge_base_dir.glob("*.md")):
        content = path.read_text(encoding="utf-8").strip()
        chunks = chunk_text(
            content,
            chunk_size=settings.memory_chunk_size,
            chunk_overlap=settings.memory_chunk_overlap,
        )
        for index, chunk in enumerate(chunks or [content]):
            documents.append(
                VectorDocument(
                    id=_stable_document_id(path.name, chunk, index),
                    source=f"{path.name}#chunk-{index + 1}",
                    content=chunk,
                    embedding=get_embedding(chunk, settings=settings),
                    memory_type="knowledge_base",
                    quality_score=9.0,
                    timestamp=path.stat().st_mtime,
                    usage_count=0,
                    last_accessed=path.stat().st_mtime,
                    metadata={"document": path.name},
                )
            )

    if not documents:
        raise FileNotFoundError(f"No knowledge base documents found in: {knowledge_base_dir}")

    return documents


def _load_memory_documents(memory_store_path: Path) -> list[VectorDocument]:
    if not memory_store_path.exists():
        return []

    documents: list[VectorDocument] = []
    for payload in load_records(memory_store_path):
        documents.append(VectorDocument.from_dict(payload))
    return documents


def _density_bonus(query_embedding: EmbeddingVector, document_embedding: EmbeddingVector) -> float:
    if query_embedding.kind != "sparse" or document_embedding.kind != "sparse":
        return 0.0
    query_values = query_embedding.values
    document_values = document_embedding.values
    if not isinstance(query_values, dict) or not isinstance(document_values, dict):
        return 0.0
    overlap = len(set(query_values).intersection(document_values))
    return overlap / max(math.sqrt(len(document_values)), 1.0)


def _stable_document_id(source: str, chunk: str, index: int) -> str:
    digest = hashlib.sha1(f"{source}:{index}:{chunk}".encode("utf-8")).hexdigest()
    return digest[:16]


def _default_quality_score(memory_type: str) -> float:
    if memory_type == "knowledge_base":
        return 9.0
    return 8.0
