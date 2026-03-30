from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class MessageEvent(BaseModel):
    topic: str
    sender: str
    recipient: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)


class BrainContext(BaseModel):
    normalized_problem: str
    intent: str
    constraints: list[str]
    sub_tasks: list[str]
    cognitive_trace: list[str]
    llm_mode: str


class AgentTask(BaseModel):
    agent_name: str
    objective: str
    context: BrainContext


class AgentFinding(BaseModel):
    agent_name: str
    summary: str
    details: list[str]
    confidence: float = Field(..., ge=0.0, le=1.0)
    recommended_action: str


class OrchestrationStatus(BaseModel):
    distributed_tasks: int
    completed_agents: list[str]
    bus_events: int
    loop_status: str


class FeedbackRecord(BaseModel):
    selected_strategy: str
    improvement_hint: str


class MemorySnapshot(BaseModel):
    total_runs: int
    recent_problems: list[str]
    latest_feedback: str | None = None


class CognitiveResponse(BaseModel):
    request_id: str
    brain: BrainContext
    heart: OrchestrationStatus
    agent_results: list[AgentFinding]
    final_answer: str
    message_log: list[MessageEvent]
    feedback: FeedbackRecord
    memory: MemorySnapshot
