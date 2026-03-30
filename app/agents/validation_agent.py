from app.models import AgentFinding, AgentTask, MemorySnapshot


class ValidationAgent:
    name = "ValidationAgent"

    def run(self, task: AgentTask, memory: MemorySnapshot) -> AgentFinding:
        constraints = task.context.constraints or ["No explicit constraints detected"]
        details = [
            f"Validation checkpoints should cover: {', '.join(constraints)}.",
            "Parallel agent outputs need agreement checks before synthesis.",
            f"Historical context available from memory: {len(memory.recent_problems)} recent prompts.",
        ]
        summary = "Guard the system with explicit risk checks, missing-context detection, and post-synthesis validation."
        recommended_action = "Define acceptance tests for correctness, safety, and operational constraints before expanding the agent graph."
        confidence = 0.8 if constraints and constraints[0] != "No explicit constraints detected" else 0.76
        return AgentFinding(
            agent_name=self.name,
            summary=summary,
            details=details,
            confidence=confidence,
            recommended_action=recommended_action,
        )
