# Brain-to-System Mapping

This project can be viewed as a simplified artificial cognitive system inspired by brain structure.

## Core mapping

- Cerebral cortex -> reasoning system
- Cerebellum -> optimization and coordination
- Neurons -> thinking modules
- Synapses -> communication links
- Memory -> knowledge base and future vector storage
- Spinal cord -> execution and action pipeline

## Current project mapping

### Cerebral cortex

The cortex analogy maps to the modules that perform high-level reasoning:

- `modules/generator.py`
- `modules/evaluator.py`
- `modules/refiner.py`
- `modules/selector.py`

These modules generate candidate thoughts, judge them, improve them, and choose the strongest output.

### Cerebellum

The cerebellum analogy maps to coordination and optimization across the pipeline:

- `main.py` orchestrates the stages
- selection and refinement comparison optimize the final response
- scoring helps stabilize which answer is retained

This is not a separate module yet, but it already exists as control logic.

### Neurons

Each module can be viewed as a neuron-like processing unit:

- input arrives
- internal transformation happens
- output is passed forward

Examples:

- generator: problem -> candidate answers
- evaluator: candidate answer -> score and feedback
- refiner: answer plus feedback -> improved answer

### Synapses

Synapses are the links between modules:

- generator -> evaluator
- evaluator -> selector
- selector -> refiner
- refiner -> evaluator

Today these links are function calls and structured Python data. Later they can become message buses, tool calls, or agent-to-agent interfaces.

### Memory

Current memory-like parts:

- `knowledge_base/` stores static guidance
- `modules/retriever.py` retrieves relevant context

This is closer to semantic memory than full episodic memory. Persistent user history and vector storage are still future work.

### Spinal cord

The spinal-cord analogy maps to the path that converts reasoning into visible action:

- `api/main.py`
- `index.html`
- CLI output in `main.py`

These parts take the internal reasoning result and deliver it to the outside world.

## Interpreting the neuron images

The neuron diagrams suggest this computational pattern:

1. Inputs arrive from multiple sources.
2. A weighted transformation happens.
3. An activation or decision is produced.
4. The result is transmitted onward.

In this project:

- inputs -> user problem plus retrieved context
- weighted transformation -> generation, scoring, and refinement rules
- activation -> score threshold, selection, or improvement decision
- output -> next module input or final answer

## Interpreting the brain-system images

The cortex and memory diagrams point to a broader long-term architecture:

- reasoning engine at the center
- memory feeding the reasoner
- goal and reward systems shaping selection
- coordination systems improving performance

The current project already has:

- reasoning
- retrieval
- scoring
- refinement
- execution through API and UI

The current project does not yet have:

- persistent episodic memory
- vector database
- explicit reward model
- planner module
- multi-agent communication fabric

## Practical takeaway

This project is now a small brain-inspired cognitive loop:

input -> generate -> evaluate -> refine -> re-evaluate -> output

The next major brain-like upgrades would be:

- persistent memory
- vector storage
- planner and goal modules
- message-passing communication between reasoning units
- specialized sub-agents for distinct cortical functions
