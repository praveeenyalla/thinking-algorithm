from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any
from urllib import error as urllib_error
from urllib import request as urllib_request

from app.models import BrainContext, MemorySnapshot


CONSTRAINT_MARKERS = {"budget", "cost", "deadline", "latency", "privacy", "risk", "safety", "security", "time"}


@dataclass(slots=True)
class OpenAICompatibleClient:
    api_key: str
    model: str
    base_url: str = "https://api.openai.com/v1"

    @classmethod
    def from_env(cls) -> "OpenAICompatibleClient | None":
        enabled = os.getenv("COGNITIVE_AI_ENABLE_LLM", "").strip().lower() in {"1", "true", "yes", "on"}
        api_key = os.getenv("OPENAI_API_KEY")
        if not enabled or not api_key:
            return None
        return cls(
            api_key=api_key,
            model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
            base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        )

    def analyze(self, problem: str, memory: MemorySnapshot) -> dict[str, Any]:
        prompt = (
            "Return JSON with keys intent, constraints, sub_tasks. "
            "Keep constraints and sub_tasks as short arrays of strings. "
            f"Problem: {problem}\n"
            f"Recent problems: {memory.recent_problems}"
        )
        payload = json.dumps(
            {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "You are a reasoning engine for a distributed cognitive AI system."},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.2,
            }
        ).encode("utf-8")
        request = urllib_request.Request(
            url=f"{self.base_url.rstrip('/')}/chat/completions",
            data=payload,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib_request.urlopen(request, timeout=20) as response:
                raw = json.loads(response.read().decode("utf-8"))
        except (urllib_error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"LLM reasoning failed: {exc}") from exc

        try:
            content = raw["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Unexpected LLM response shape: {raw}") from exc
        return _extract_json(content)


class BrainReasoner:
    def __init__(self, llm_client: OpenAICompatibleClient | None = None) -> None:
        self._llm_client = llm_client if llm_client is not None else OpenAICompatibleClient.from_env()

    @property
    def llm_enabled(self) -> bool:
        return self._llm_client is not None

    @property
    def mode(self) -> str:
        return "openai_compatible" if self.llm_enabled else "heuristic"

    def analyze(self, problem: str, memory: MemorySnapshot) -> BrainContext:
        normalized = re.sub(r"\s+", " ", problem.strip()).lower()
        heuristic = self._heuristic_analysis(normalized, memory)

        if self._llm_client is not None:
            try:
                enriched = self._llm_client.analyze(problem, memory)
                return BrainContext(
                    normalized_problem=normalized,
                    intent=_coerce_text(enriched.get("intent"), heuristic["intent"]),
                    constraints=_coerce_list(enriched.get("constraints"), heuristic["constraints"]),
                    sub_tasks=_coerce_list(enriched.get("sub_tasks"), heuristic["sub_tasks"]),
                    cognitive_trace=heuristic["cognitive_trace"] + ["LLM augmentation applied to Brain analysis."],
                    llm_mode="openai_compatible",
                )
            except RuntimeError as exc:
                heuristic["cognitive_trace"].append(str(exc))

        return BrainContext(
            normalized_problem=normalized,
            intent=heuristic["intent"],
            constraints=heuristic["constraints"],
            sub_tasks=heuristic["sub_tasks"],
            cognitive_trace=heuristic["cognitive_trace"],
            llm_mode="heuristic",
        )

    def _heuristic_analysis(self, normalized: str, memory: MemorySnapshot) -> dict[str, Any]:
        constraints = _extract_constraints(normalized)
        intent = _infer_intent(normalized)
        sub_tasks = [
            "Clarify the request context",
            "Distribute research, logic, and validation work",
            "Merge agent outputs into a final recommendation",
            "Record feedback for the next run",
        ]
        cognitive_trace = [
            "Normalized the user problem for shared agent context.",
            f"Inferred intent: {intent}",
            f"Detected constraints: {', '.join(constraints) if constraints else 'none explicit'}.",
        ]
        if memory.recent_problems:
            cognitive_trace.append(f"Loaded {len(memory.recent_problems)} recent problems from memory.")
        else:
            cognitive_trace.append("No prior memory available yet.")
        return {
            "intent": intent,
            "constraints": constraints or ["No explicit constraints detected"],
            "sub_tasks": sub_tasks,
            "cognitive_trace": cognitive_trace,
        }


def _infer_intent(normalized: str) -> str:
    if any(token in normalized for token in ("build", "create", "develop", "design")):
        return "Design and implement a working system."
    if any(token in normalized for token in ("analyze", "review", "explain", "understand")):
        return "Analyze the request and produce a structured plan."
    if any(token in normalized for token in ("improve", "upgrade", "optimize")):
        return "Refine an existing system with stronger architecture and execution clarity."
    return "Resolve the request using a coordinated cognitive workflow."


def _extract_constraints(normalized: str) -> list[str]:
    found = [marker for marker in CONSTRAINT_MARKERS if marker in normalized]
    phrase_matches = re.findall(r"(?:with|under|without)\s+([a-z0-9 -]{4,40})", normalized)
    candidates = found + [match.strip() for match in phrase_matches]
    ordered: list[str] = []
    for item in candidates:
        cleaned = item.strip()
        if not cleaned:
            continue
        overlap_index = next(
            (index for index, existing in enumerate(ordered) if cleaned in existing or existing in cleaned),
            None,
        )
        if overlap_index is None:
            ordered.append(cleaned)
        elif len(cleaned) > len(ordered[overlap_index]):
            ordered[overlap_index] = cleaned
    return ordered[:4]


def _coerce_text(value: Any, fallback: str) -> str:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return fallback


def _coerce_list(value: Any, fallback: list[str]) -> list[str]:
    if isinstance(value, list):
        cleaned = [str(item).strip() for item in value if str(item).strip()]
        if cleaned:
            return cleaned[:4]
    return fallback


def _extract_json(content: str) -> dict[str, Any]:
    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise RuntimeError(f"Could not extract JSON from LLM output: {content!r}")
    try:
        return json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON from LLM output: {content!r}") from exc
