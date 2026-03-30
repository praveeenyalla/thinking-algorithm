from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from uuid import uuid4

from app.agents import LogicAgent, ResearchAgent, ValidationAgent
from app.brain.reasoning import BrainReasoner
from app.brain.synthesis import BrainSynthesizer
from app.communication.message_queue import MessageBus
from app.memory.storage import MemoryStore
from app.models import (
    AgentFinding,
    AgentTask,
    CognitiveResponse,
    FeedbackRecord,
    MemorySnapshot,
    OrchestrationStatus,
)


class HeartController:
    def __init__(
        self,
        reasoner: BrainReasoner,
        synthesizer: BrainSynthesizer,
        memory_store: MemoryStore,
        agents: list[object],
    ) -> None:
        self._reasoner = reasoner
        self._synthesizer = synthesizer
        self._memory_store = memory_store
        self._agents = {agent.name: agent for agent in agents}

    @classmethod
    def build_default(cls) -> "HeartController":
        return cls(
            reasoner=BrainReasoner(),
            synthesizer=BrainSynthesizer(),
            memory_store=MemoryStore(),
            agents=[ResearchAgent(), LogicAgent(), ValidationAgent()],
        )

    @property
    def llm_mode(self) -> str:
        return self._reasoner.mode

    def memory_snapshot(self) -> MemorySnapshot:
        return self._memory_store.snapshot()

    def run(self, problem: str) -> CognitiveResponse:
        request_id = uuid4().hex[:12]
        bus = MessageBus()
        memory_before = self._memory_store.snapshot()

        bus.publish(
            topic="input.received",
            sender="api",
            payload={"request_id": request_id, "problem": problem},
        )

        brain_context = self._reasoner.analyze(problem, memory_before)
        bus.publish(
            topic="brain.context_ready",
            sender="brain",
            payload=brain_context.model_dump(),
        )

        tasks = self._build_tasks(brain_context)
        bus.publish(
            topic="heart.tasks_created",
            sender="heart",
            payload={"count": len(tasks), "agents": [task.agent_name for task in tasks]},
        )

        agent_results = self._dispatch(tasks, memory_before, bus)
        completed_agents = [result.agent_name for result in agent_results]
        lead_agent = max(agent_results, key=lambda item: item.confidence).agent_name
        bus.publish(
            topic="heart.tasks_completed",
            sender="heart",
            payload={"completed_agents": completed_agents},
        )

        final_answer = self._synthesizer.synthesize(problem, brain_context, agent_results, memory_before)
        bus.publish(
            topic="brain.synthesized",
            sender="brain",
            payload={"request_id": request_id, "lead_agent": lead_agent},
        )

        feedback = self._build_feedback(agent_results, brain_context.llm_mode)
        self._memory_store.record_run(problem, final_answer, feedback, agent_results)
        bus.publish(
            topic="feedback.recorded",
            sender="memory",
            payload=feedback.model_dump(),
        )

        memory_after = self._memory_store.snapshot()
        message_log = bus.history()
        heart_status = OrchestrationStatus(
            distributed_tasks=len(tasks),
            completed_agents=completed_agents,
            bus_events=len(message_log),
            loop_status="complete",
        )
        return CognitiveResponse(
            request_id=request_id,
            brain=brain_context,
            heart=heart_status,
            agent_results=agent_results,
            final_answer=final_answer,
            message_log=message_log,
            feedback=feedback,
            memory=memory_after,
        )

    def _build_tasks(self, brain_context) -> list[AgentTask]:
        return [
            AgentTask(
                agent_name="ResearchAgent",
                objective="Collect context, requirements, and relevant signals.",
                context=brain_context,
            ),
            AgentTask(
                agent_name="LogicAgent",
                objective="Construct the execution and reasoning plan.",
                context=brain_context,
            ),
            AgentTask(
                agent_name="ValidationAgent",
                objective="Stress test the proposal against risks and constraints.",
                context=brain_context,
            ),
        ]

    def _dispatch(
        self,
        tasks: list[AgentTask],
        memory: MemorySnapshot,
        bus: MessageBus,
    ) -> list[AgentFinding]:
        results: list[AgentFinding] = []
        with ThreadPoolExecutor(max_workers=len(tasks)) as executor:
            futures = {}
            for task in tasks:
                bus.publish(
                    topic="heart.task_dispatched",
                    sender="heart",
                    recipient=task.agent_name,
                    payload={"objective": task.objective},
                )
                future = executor.submit(self._run_agent, task, memory, bus)
                futures[future] = task.agent_name

            for future in as_completed(futures):
                results.append(future.result())

        return sorted(results, key=lambda item: item.agent_name)

    def _run_agent(
        self,
        task: AgentTask,
        memory: MemorySnapshot,
        bus: MessageBus,
    ) -> AgentFinding:
        bus.publish(
            topic="agent.started",
            sender=task.agent_name,
            payload={"objective": task.objective},
        )
        result = self._agents[task.agent_name].run(task, memory)
        bus.publish(
            topic="agent.completed",
            sender=task.agent_name,
            payload={
                "summary": result.summary,
                "confidence": result.confidence,
            },
        )
        return result

    def _build_feedback(self, results: list[AgentFinding], llm_mode: str) -> FeedbackRecord:
        lead = max(results, key=lambda item: item.confidence)
        if llm_mode == "heuristic":
            hint = (
                f"Lead strategy came from {lead.agent_name}. "
                "Enable `COGNITIVE_AI_ENABLE_LLM=1` with an API key when you want Brain reasoning upgraded beyond heuristics."
            )
        else:
            hint = f"Lead strategy came from {lead.agent_name}. Tighten validation criteria before adding more autonomous agents."
        return FeedbackRecord(selected_strategy=lead.agent_name, improvement_hint=hint)
