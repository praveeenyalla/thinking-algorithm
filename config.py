"""Configuration values for the AI brain system."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


def _load_env_file() -> None:
    """Load simple KEY=VALUE pairs from a local .env file if present."""
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", maxsplit=1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


_load_env_file()


def _env_flag(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    """Runtime settings for the pipeline."""

    log_level: str = os.getenv("LOG_LEVEL", "WARNING")
    knowledge_base_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "knowledge_base")
    ui_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "ui")
    retrieval_top_k: int = 3
    generator_retrieval_top_k: int = 3
    controller_memory_top_k: int = 3
    memory_chunk_size: int = int(os.getenv("MEMORY_CHUNK_SIZE", "120"))
    memory_chunk_overlap: int = int(os.getenv("MEMORY_CHUNK_OVERLAP", "30"))
    memory_store_path: Path = field(
        default_factory=lambda: PROJECT_ROOT / "memory" / "data" / "vector_memory.jsonl"
    )
    case_memory_path: Path = field(
        default_factory=lambda: PROJECT_ROOT / "memory" / "data" / "case_memory.jsonl"
    )
    memory_quality_threshold: int = int(os.getenv("MEMORY_QUALITY_THRESHOLD", "7"))
    case_memory_score_threshold: int = int(os.getenv("CASE_MEMORY_SCORE_THRESHOLD", "8"))
    case_reuse_similarity_threshold: float = float(os.getenv("CASE_REUSE_SIMILARITY_THRESHOLD", "0.45"))
    enable_memory_cleanup: bool = field(default_factory=lambda: not _env_flag("DISABLE_MEMORY_CLEANUP"))
    memory_retention_days: int = int(os.getenv("MEMORY_RETENTION_DAYS", "30"))
    case_retention_days: int = int(os.getenv("CASE_RETENTION_DAYS", "45"))
    memory_min_usage_to_keep: int = int(os.getenv("MEMORY_MIN_USAGE_TO_KEEP", "1"))
    case_min_usage_to_keep: int = int(os.getenv("CASE_MIN_USAGE_TO_KEEP", "1"))
    quick_query_token_limit: int = int(os.getenv("QUICK_QUERY_TOKEN_LIMIT", "4"))
    enable_multi_agent_deliberate: bool = field(default_factory=lambda: not _env_flag("DISABLE_MULTI_AGENT_DELIBERATE"))
    evaluation_threshold: int = 8
    criterion_weights: dict[str, float] = field(
        default_factory=lambda: {
            "correctness": 0.4,
            "clarity": 0.3,
            "usefulness": 0.3,
        }
    )
    enable_llm_generator: bool = field(default_factory=lambda: _env_flag("ENABLE_LLM_GENERATOR"))
    generator_model: str = field(default_factory=lambda: os.getenv("GENERATOR_MODEL", "openai/gpt-oss-120b"))
    generator_api_key: str | None = field(
        default_factory=lambda: (
            os.getenv("GENERATOR_API_KEY")
            or os.getenv("API_KEY")
            or os.getenv("NVIDIA_API_KEY")
            or os.getenv("OPENAI_API_KEY")
        )
    )
    generator_base_url: str = field(
        default_factory=lambda: (
            os.getenv("GENERATOR_BASE_URL")
            or os.getenv("OPENAI_BASE_URL")
            or "https://integrate.api.nvidia.com/v1"
        )
    )
    generator_temperature: float = float(os.getenv("GENERATOR_TEMPERATURE", "0.7"))
    generator_top_p: float = float(os.getenv("GENERATOR_TOP_P", "1.0"))
    generator_max_tokens: int = int(os.getenv("GENERATOR_MAX_TOKENS", "700"))
    enable_llm_judge: bool = field(default_factory=lambda: _env_flag("ENABLE_LLM_JUDGE"))
    judge_model: str = field(default_factory=lambda: os.getenv("JUDGE_MODEL", "openai/gpt-oss-120b"))
    judge_api_key: str | None = field(
        default_factory=lambda: (
            os.getenv("JUDGE_API_KEY")
            or os.getenv("API_KEY")
            or os.getenv("NVIDIA_API_KEY")
            or os.getenv("OPENAI_API_KEY")
        )
    )
    judge_base_url: str = field(
        default_factory=lambda: (
            os.getenv("JUDGE_BASE_URL")
            or os.getenv("OPENAI_BASE_URL")
            or "https://integrate.api.nvidia.com/v1"
        )
    )
    enable_llm_refiner: bool = field(default_factory=lambda: _env_flag("ENABLE_LLM_REFINER"))
    refiner_model: str = field(default_factory=lambda: os.getenv("REFINER_MODEL", "openai/gpt-oss-120b"))
    refiner_api_key: str | None = field(
        default_factory=lambda: (
            os.getenv("REFINER_API_KEY")
            or os.getenv("API_KEY")
            or os.getenv("NVIDIA_API_KEY")
            or os.getenv("OPENAI_API_KEY")
        )
    )
    refiner_base_url: str = field(
        default_factory=lambda: (
            os.getenv("REFINER_BASE_URL")
            or os.getenv("OPENAI_BASE_URL")
            or "https://integrate.api.nvidia.com/v1"
        )
    )
    refiner_retrieval_top_k: int = 3
    refiner_temperature: float = float(os.getenv("REFINER_TEMPERATURE", "0.6"))
    refiner_top_p: float = float(os.getenv("REFINER_TOP_P", "1.0"))
    refiner_max_tokens: int = int(os.getenv("REFINER_MAX_TOKENS", "900"))
    enable_remote_embeddings: bool = field(default_factory=lambda: _env_flag("ENABLE_REMOTE_EMBEDDINGS"))
    embedding_model: str = field(default_factory=lambda: os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"))
    embedding_api_key: str | None = field(
        default_factory=lambda: (
            os.getenv("EMBEDDING_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or os.getenv("API_KEY")
        )
    )
    embedding_base_url: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_BASE_URL") or "https://api.openai.com/v1"
    )
    persist_memory: bool = field(default_factory=lambda: not _env_flag("DISABLE_MEMORY_WRITE"))
    persist_feedback: bool = field(default_factory=lambda: _env_flag("PERSIST_FEEDBACK"))
    feedback_store_path: Path = field(
        default_factory=lambda: PROJECT_ROOT / "learning" / "data" / "feedback_store.jsonl"
    )


def get_settings() -> Settings:
    """Return the default project settings."""
    return Settings()
