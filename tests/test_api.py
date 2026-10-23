import pytest
from fastapi.testclient import TestClient

from api.main import app
from brain.controller import thinking_pipeline as controller_thinking_pipeline
from config import Settings, get_settings
import api.main as api_module


client = TestClient(app)


@pytest.fixture(autouse=True)
def disable_runtime_memory_writes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("DISABLE_MEMORY_WRITE", "1")
    monkeypatch.setenv("DISABLE_MEMORY_CLEANUP", "1")
    base = get_settings()
    settings = Settings(
        knowledge_base_dir=base.knowledge_base_dir,
        memory_store_path=tmp_path / "vector_memory.jsonl",
        case_memory_path=tmp_path / "case_memory.jsonl",
        enable_llm_judge=False,
        persist_memory=False,
    )
    monkeypatch.setattr(
        api_module,
        "thinking_pipeline",
        lambda problem, response_mode="chat": controller_thinking_pipeline(
            problem,
            settings=settings,
            response_mode=response_mode,
        ),
    )


def test_health_endpoint_returns_ok() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_index_page_loads() -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert "AI Brain System v7" in response.text
    assert "Chat Mode" in response.text


def test_think_endpoint_returns_pipeline_result() -> None:
    response = client.post("/think", json={"problem": "Impact of AI on jobs"})

    assert response.status_code == 200
    payload = response.json()["result"]
    assert set(payload) == {"final_answer"}
    assert payload["final_answer"]
    assert "Impact of AI on jobs" in payload["final_answer"]
    assert "best understood" not in payload["final_answer"].lower()
    assert "agent" not in payload["final_answer"].lower()
    assert "pipeline" not in payload["final_answer"].lower()


def test_think_endpoint_supports_deep_research_mode() -> None:
    response = client.post("/think", json={"problem": "Explain core Java with examples", "mode": "deep_research"})

    assert response.status_code == 200
    payload = response.json()["result"]
    assert set(payload) == {"final_answer"}
    assert "Overview" in payload["final_answer"]
    assert "How It Works" in payload["final_answer"]


def test_think_endpoint_accepts_short_chat_inputs() -> None:
    response = client.post("/think", json={"problem": "hi"})

    assert response.status_code == 200
    payload = response.json()["result"]
    assert payload["final_answer"]


def test_think_endpoint_accepts_legacy_query_field_and_spaced_mode() -> None:
    response = client.post("/think", json={"query": "Explain core Java with examples", "mode": "deep research"})

    assert response.status_code == 200
    payload = response.json()["result"]
    assert "Overview" in payload["final_answer"]
