"""Embedding utilities for the memory layer."""

from __future__ import annotations

from dataclasses import dataclass
from collections import Counter
import math
import re
from typing import Any

from openai import OpenAI

from config import Settings, get_settings

TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class EmbeddingVector:
    """Embedding container for either sparse local or dense remote vectors."""

    kind: str
    values: dict[str, float] | list[float]
    norm: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "values": self.values,
            "norm": self.norm,
        }


def tokenize(text: str) -> list[str]:
    """Tokenize text for local retrieval."""
    return TOKEN_PATTERN.findall(text.lower())


def get_embedding(text: str, settings: Settings | None = None) -> EmbeddingVector:
    """Return an embedding for text using remote embeddings or local fallback."""
    active_settings = settings or get_settings()
    if active_settings.enable_remote_embeddings and active_settings.embedding_api_key:
        try:
            return _remote_embedding(text, active_settings)
        except RuntimeError:
            pass
    return _local_sparse_embedding(text)


def embedding_from_dict(payload: dict[str, Any]) -> EmbeddingVector:
    """Reconstruct an embedding vector from serialized data."""
    return EmbeddingVector(
        kind=str(payload["kind"]),
        values=payload["values"],
        norm=float(payload["norm"]),
    )


def _local_sparse_embedding(text: str) -> EmbeddingVector:
    """Create a normalized sparse embedding from text."""
    counts = Counter(tokenize(text))
    if not counts:
        return EmbeddingVector(kind="sparse", values={}, norm=0.0)

    total = sum(counts.values())
    weights = {token: count / total for token, count in counts.items()}
    norm = math.sqrt(sum(weight * weight for weight in weights.values()))
    return EmbeddingVector(kind="sparse", values=weights, norm=norm)

def _remote_embedding(text: str, settings: Settings) -> EmbeddingVector:
    """Create a dense embedding through an OpenAI-compatible embeddings endpoint."""
    client = OpenAI(
        base_url=settings.embedding_base_url,
        api_key=settings.embedding_api_key,
    )
    try:
        response = client.embeddings.create(
            model=settings.embedding_model,
            input=text,
        )
    except Exception as exc:
        raise RuntimeError(f"Embedding call failed: {exc}") from exc

    try:
        values = [float(value) for value in response.data[0].embedding]
    except (AttributeError, IndexError, TypeError) as exc:
        raise RuntimeError(f"Unexpected embedding response: {response}") from exc

    norm = math.sqrt(sum(value * value for value in values))
    return EmbeddingVector(kind="dense", values=values, norm=norm)


def cosine_similarity(left: EmbeddingVector, right: EmbeddingVector) -> float:
    """Compute cosine similarity for either sparse or dense embeddings."""
    if left.kind != right.kind:
        return 0.0
    if left.norm == 0 or right.norm == 0:
        return 0.0

    if left.kind == "dense":
        left_values = left.values
        right_values = right.values
        if not isinstance(left_values, list) or not isinstance(right_values, list):
            return 0.0
        dot_product = sum(left_value * right_value for left_value, right_value in zip(left_values, right_values))
        return dot_product / (left.norm * right.norm)

    left_weights = left.values
    right_weights = right.values
    if not isinstance(left_weights, dict) or not isinstance(right_weights, dict):
        return 0.0
    shared_tokens = set(left_weights).intersection(right_weights)
    dot_product = sum(left_weights[token] * right_weights[token] for token in shared_tokens)
    return dot_product / (left.norm * right.norm)
