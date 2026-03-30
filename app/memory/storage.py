from __future__ import annotations

from collections import deque
from threading import Lock

from app.models import AgentFinding, FeedbackRecord, MemorySnapshot


class MemoryStore:
    def __init__(self, max_runs: int = 25) -> None:
        self._runs: deque[dict] = deque(maxlen=max_runs)
        self._lock = Lock()

    def record_run(
        self,
        problem: str,
        final_answer: str,
        feedback: FeedbackRecord,
        agent_results: list[AgentFinding],
    ) -> None:
        entry = {
            "problem": problem,
            "final_answer": final_answer,
            "feedback": feedback,
            "agents": [item.agent_name for item in agent_results],
        }
        with self._lock:
            self._runs.append(entry)

    def snapshot(self) -> MemorySnapshot:
        with self._lock:
            recent_problems = [entry["problem"] for entry in list(self._runs)[-5:]][::-1]
            latest_feedback = self._runs[-1]["feedback"].improvement_hint if self._runs else None
            return MemorySnapshot(
                total_runs=len(self._runs),
                recent_problems=recent_problems,
                latest_feedback=latest_feedback,
            )
