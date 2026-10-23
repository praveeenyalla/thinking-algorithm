"""v7 multi-agent building blocks for deliberate reasoning."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class AgentDraft:
    """One candidate answer produced by a specialized agent."""

    agent: str
    role: str
    answer: str
    notes: str
    context_sources: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent": self.agent,
            "role": self.role,
            "answer": self.answer,
            "notes": self.notes,
            "context_sources": self.context_sources,
        }


@dataclass(frozen=True)
class Critique:
    """Structured critique of one agent draft."""

    target_agent: str
    summary: str
    missing_points: list[str]
    strengths: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_agent": self.target_agent,
            "summary": self.summary,
            "missing_points": self.missing_points,
            "strengths": self.strengths,
        }


__all__ = ["AgentDraft", "Critique"]
