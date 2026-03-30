"""Local RAG retriever for evaluator grounding."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


@dataclass(frozen=True)
class RetrievedContext:
    """A retrieved knowledge snippet used to ground evaluation."""

    source: str
    content: str
    relevance: float

    def to_dict(self) -> dict[str, str | float]:
        return {
            "source": self.source,
            "content": self.content,
            "relevance": round(self.relevance, 3),
        }


def retrieve_context(query: str, knowledge_base_dir: Path, top_k: int = 3) -> list[RetrievedContext]:
    """Retrieve the most relevant knowledge snippets for a query."""
    documents = _load_documents(str(knowledge_base_dir))
    query_tokens = _tokenize(
        f"{query} correctness clarity usefulness threshold rubric rag robustness judge scorer evaluator"
    )

    ranked: list[RetrievedContext] = []
    for source, content in documents:
        content_tokens = _tokenize(content)
        overlap = len(query_tokens.intersection(content_tokens))
        normalized_overlap = overlap / max(len(query_tokens), 1)
        density_bonus = overlap / max(math.sqrt(len(content_tokens)), 1.0)
        relevance = normalized_overlap + density_bonus
        ranked.append(RetrievedContext(source=source, content=content, relevance=relevance))

    ranked.sort(key=lambda item: (-item.relevance, item.source))
    return ranked[:top_k]


@lru_cache(maxsize=4)
def _load_documents(knowledge_base_dir: str) -> tuple[tuple[str, str], ...]:
    base_path = Path(knowledge_base_dir)
    if not base_path.exists():
        raise FileNotFoundError(f"Knowledge base directory does not exist: {base_path}")

    documents: list[tuple[str, str]] = []
    for path in sorted(base_path.glob("*.md")):
        documents.append((path.name, path.read_text(encoding="utf-8").strip()))

    if not documents:
        raise FileNotFoundError(f"No knowledge base documents found in: {base_path}")
    return tuple(documents)


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))
