from app.models import AgentFinding, AgentTask, MemorySnapshot


class LogicAgent:
    name = "LogicAgent"

    def run(self, task: AgentTask, memory: MemorySnapshot) -> AgentFinding:
        ordered_steps = ", ".join(task.context.sub_tasks[:3])
        constraints = ", ".join(task.context.constraints) if task.context.constraints else "no explicit constraints"
        details = [
            f"Execution sequence: {ordered_steps}.",
            f"Constraint set: {constraints}.",
            f"Memory-informed continuity: {memory.total_runs} total runs recorded so far.",
        ]
        summary = "Drive the solution through a staged reasoning plan with explicit tradeoff checks between each phase."
        recommended_action = "Implement the system in modular layers so each reasoning step can be tested and replaced independently."
        confidence = 0.83
        return AgentFinding(
            agent_name=self.name,
            summary=summary,
            details=details,
            confidence=confidence,
            recommended_action=recommended_action,
        )
