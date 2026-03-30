"""Case-based memory for reusing strong past solutions."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import time

from config import Settings, get_settings
from memory.embedder import EmbeddingVector, cosine_similarity, embedding_from_dict, get_embedding, tokenize
from memory.persistent_store import load_records, touch_records, upsert_record


@dataclass(frozen=True)
class CaseRecord:
    """A solved case retained for future reuse."""

    id: str
    problem: str
    answer: str
    score: int
    problem_embedding: EmbeddingVector
    timestamp: float
    usage_count: int
    last_accessed: float

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "problem": self.problem,
            "answer": self.answer,
            "score": self.score,
            "problem_embedding": self.problem_embedding.to_dict(),
            "timestamp": self.timestamp,
            "usage_count": self.usage_count,
            "last_accessed": self.last_accessed,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "CaseRecord":
        return cls(
            id=str(payload["id"]),
            problem=str(payload["problem"]),
            answer=str(payload["answer"]),
            score=int(payload["score"]),
            problem_embedding=embedding_from_dict(payload["problem_embedding"]),
            timestamp=float(payload.get("timestamp", 0.0)),
            usage_count=int(payload.get("usage_count", 0)),
            last_accessed=float(payload.get("last_accessed", payload.get("timestamp", 0.0))),
        )


@dataclass(frozen=True)
class RetrievedCase:
    """A matched past case and its retrieval scores."""

    problem: str
    answer: str
    score: int
    similarity: float
    weighted_score: float
    timestamp: float
    usage_count: int

    def to_dict(self) -> dict[str, object]:
        return {
            "problem": self.problem,
            "answer": self.answer,
            "score": self.score,
            "similarity": round(self.similarity, 3),
            "weighted_score": round(self.weighted_score, 3),
            "timestamp": self.timestamp,
            "usage_count": self.usage_count,
        }


def store_case(problem: str, answer: str, score: int, settings: Settings | None = None) -> bool:
    """Store a solved case when its quality score is high enough."""
    active_settings = settings or get_settings()
    if score < active_settings.case_memory_score_threshold:
        return False

    case_record = CaseRecord(
        id=_case_id(problem, answer),
        problem=problem,
        answer=answer,
        score=score,
        problem_embedding=get_embedding(problem, settings=active_settings),
        timestamp=time.time(),
        usage_count=0,
        last_accessed=time.time(),
    )
    return upsert_record(active_settings.case_memory_path, case_record.to_dict())


def retrieve_case(problem: str, settings: Settings | None = None) -> RetrievedCase | None:
    """Retrieve the best matching high-quality past case."""
    active_settings = settings or get_settings()
    cases = _load_cases(active_settings.case_memory_path)
    if not cases:
        return None

    query_embedding = get_embedding(problem, settings=active_settings)
    best_match: RetrievedCase | None = None

    for case in cases:
        embedding_similarity = cosine_similarity(query_embedding, case.problem_embedding)
        overlap_similarity = _token_overlap_similarity(problem, case.problem)
        similarity = max(embedding_similarity, overlap_similarity)
        weighted_score = similarity * (case.score / 10.0)
        candidate = RetrievedCase(
            problem=case.problem,
            answer=case.answer,
            score=case.score,
            similarity=similarity,
            weighted_score=weighted_score,
            timestamp=case.timestamp,
            usage_count=case.usage_count,
        )
        if best_match is None or candidate.weighted_score > best_match.weighted_score:
            best_match = candidate

    if best_match is None or best_match.similarity < active_settings.case_reuse_similarity_threshold:
        return None
    matching_ids = {
        case.id
        for case in cases
        if case.problem == best_match.problem and case.answer == best_match.answer and case.score == best_match.score
    }
    touch_records(active_settings.case_memory_path, matching_ids, time.time())
    return best_match


def _load_cases(case_memory_path: Path) -> list[CaseRecord]:
    if not case_memory_path.exists():
        return []

    return [CaseRecord.from_dict(payload) for payload in load_records(case_memory_path)]


def _case_id(problem: str, answer: str) -> str:
    digest = hashlib.sha1(f"{problem}:{answer}".encode("utf-8")).hexdigest()
    return digest[:16]


def _token_overlap_similarity(left: str, right: str) -> float:
    left_tokens = {_normalize_token(token) for token in tokenize(left)}
    right_tokens = {_normalize_token(token) for token in tokenize(right)}
    left_tokens.discard("")
    right_tokens.discard("")
    if not left_tokens or not right_tokens:
        return 0.0
    overlap = len(left_tokens.intersection(right_tokens))
    return overlap / max(min(len(left_tokens), len(right_tokens)), 1)


def _normalize_token(token: str) -> str:
    if token.endswith("s") and len(token) > 3:
        return token[:-1]
    return token
