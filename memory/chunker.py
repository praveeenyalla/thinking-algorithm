"""Chunking utilities for episodic and knowledge memory."""

from __future__ import annotations


def chunk_text(text: str, chunk_size: int = 120, chunk_overlap: int = 30) -> list[str]:
    """Split text into overlapping word chunks."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must not be negative.")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size.")

    words = text.split()
    if not words:
        return []

    chunks: list[str] = []
    step = chunk_size - chunk_overlap
    for index in range(0, len(words), step):
        chunk_words = words[index : index + chunk_size]
        if not chunk_words:
            break
        chunks.append(" ".join(chunk_words))
        if index + chunk_size >= len(words):
            break

    return chunks
