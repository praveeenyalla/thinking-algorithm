from app.models import AgentFinding, BrainContext, MemorySnapshot


class BrainSynthesizer:
    def synthesize(
        self,
        problem: str,
        brain_context: BrainContext,
        agent_findings: list[AgentFinding],
        memory: MemorySnapshot,
    ) -> str:
        ranked = sorted(agent_findings, key=lambda item: item.confidence, reverse=True)
        lead = ranked[0]
        support = " ".join(f"{item.agent_name}: {item.summary}" for item in ranked)
        next_actions = " ".join(f"{item.agent_name} -> {item.recommended_action}" for item in ranked)
        memory_note = (
            f"Memory currently tracks {memory.total_runs} previous runs."
            if memory.total_runs
            else "This is the first recorded run in memory."
        )
        return (
            f"Recommended direction: {lead.summary} "
            f"Problem focus: {problem}. "
            f"Brain intent: {brain_context.intent} "
            f"Support from agents: {support} "
            f"Next actions: {next_actions} "
            f"{memory_note}"
        )
