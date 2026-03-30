from app.models import AgentFinding, AgentTask, MemorySnapshot


class ResearchAgent:
    name = "ResearchAgent"

    def run(self, task: AgentTask, memory: MemorySnapshot) -> AgentFinding:
        keywords = [token for token in task.context.normalized_problem.split() if len(token) > 3][:4]
        if not keywords:
            keywords = task.context.normalized_problem.split()[:4]

        details = [
            f"Focus terms identified: {', '.join(keywords) or 'general context'}.",
            f"Primary intent from Brain: {task.context.intent}.",
            f"Recent memory signals available: {len(memory.recent_problems)} prior runs.",
        ]
        summary = "Collect domain context first, then narrow the problem into evidence-backed workstreams."
        recommended_action = "Create a short research brief for the highest-impact requirements before deeper implementation."
        confidence = 0.78 if task.context.constraints else 0.74
        return AgentFinding(
            agent_name=self.name,
            summary=summary,
            details=details,
            confidence=confidence,
            recommended_action=recommended_action,
        )
