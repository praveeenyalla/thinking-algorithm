# AI Brain System v6

A brain-inspired AI system built on top of existing models rather than model training. The current implementation combines a controller layer, persistent managed memory, a thinking layer, a lightweight learning layer, a FastAPI interface, and a browser UI.

Persistent Memory v6 now includes:

- durable JSONL-backed storage for episodic memory and case memory
- chunking for knowledge and answer memory
- embeddings with local sparse fallback and optional remote embedding API
- weighted retrieval that ranks by relevance and memory quality
- quality-gated storage so weak answers are not retained
- usage tracking and last-access timestamps for retrieved memories
- cleanup rules for old, low-quality, or unused memories
- case-based reuse of strong past solutions for similar short queries

## Image-guided design

The architecture changes follow the ideas in the images you provided:

- memory is externalized persistent state, not model weights or just the context window
- chunks become embeddings, embeddings become indexed records, and those records are saved to disk
- retrieval is selective and ranked
- persistent memory has a lifecycle: add, retrieve, update usage, and delete
- long-term memory influences future behavior across separate runs

## Core architecture

- `api/`: FastAPI endpoints and browser entrypoint
- `brain/`: controller and strategy selection
- `memory/`: chunking, embedding, retrieval, case memory, cleanup, and persistent storage
- `memory/persistent_store.py`: JSONL-backed load/save/upsert/touch/delete helpers
- `memory/cleanup.py`: forgetting and lifecycle pruning
- `thinking/`: generator, evaluator, selector, and refiner
- `learning/`: lightweight feedback storage and optimization summaries
- `models/`: shared LLM client helpers
- `ui/`: browser chat interface
- `knowledge_base/`: markdown documents used for local RAG grounding

See [brain_system_mapping.md](/D:/Thinking algorithm/knowledge_base/brain_system_mapping.md) for the brain analogy that maps cortex, memory, synapses, and output to the current code.

## Runtime flow

1. Input enters through the CLI or `POST /think`.
2. The controller runs memory cleanup before reasoning.
3. The memory layer loads persistent episodic and case records from disk.
4. The controller retrieves relevant context and checks for a reusable high-quality case.
5. The thinking layer either reuses the case directly or generates fresh candidates.
6. The evaluator scores answers on correctness, clarity, and usefulness.
7. The selector picks the strongest candidate.
8. For deliberate queries, the refiner improves the chosen answer once and the evaluator scores it again.
9. The controller returns the better of the original or refined answer.
10. High-quality final answers are written back into persistent chunk memory and persistent case memory.
11. Retrieval updates usage counts so old, unused records can be pruned later.

## Project structure

```text
ai-brain-system/
|-- api/
|-- brain/
|-- knowledge_base/
|-- learning/
|-- memory/
|   |-- case_memory.py
|   |-- chunker.py
|   |-- cleanup.py
|   |-- embedder.py
|   |-- persistent_store.py
|   |-- retriever.py
|   |-- vector_store.py
|-- models/
|-- modules/
|-- tests/
|-- thinking/
|-- ui/
|-- config.py
|-- index.html
|-- main.py
|-- pyproject.toml
```

`modules/` remains on disk for backward compatibility, but the active architecture now runs through `brain/`, `memory/`, `thinking/`, `learning/`, `models/`, and `ui/`.

## Quick start

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e .[dev]
python main.py "Impact of AI on jobs"
```

## Run the browser UI

```powershell
uvicorn api.main:app --reload
```

Then open [http://127.0.0.1:8000](http://127.0.0.1:8000).

Endpoints:

- `GET /`
- `GET /health`
- `POST /think`

## Optional live model paths

The system works offline by default. To enable the NVIDIA-hosted `openai/gpt-oss-120b` model paths:

```powershell
setx API_KEY "your_api_key_here"
setx ENABLE_LLM_GENERATOR "1"
setx ENABLE_LLM_JUDGE "1"
setx ENABLE_LLM_REFINER "1"
setx GENERATOR_BASE_URL "https://integrate.api.nvidia.com/v1"
setx JUDGE_BASE_URL "https://integrate.api.nvidia.com/v1"
setx REFINER_BASE_URL "https://integrate.api.nvidia.com/v1"
```

The shared OpenAI-compatible client lives in [llm_client.py](/D:/Thinking algorithm/models/llm_client.py).

To enable remote embeddings instead of the local sparse embedder:

```powershell
setx ENABLE_REMOTE_EMBEDDINGS "1"
setx OPENAI_API_KEY "your_openai_key_here"
setx EMBEDDING_MODEL "text-embedding-3-small"
setx EMBEDDING_BASE_URL "https://api.openai.com/v1"
```

## Current implementation boundaries

- No raw model training
- No biological brain simulation
- No multi-agent planner yet
- No dedicated external vector database yet
- No long-term autonomous self-improvement loop yet

What exists now is a practical staged system: controller, persistent memory, retrieval, reasoning, single-pass refinement, lifecycle cleanup, API, and UI.
