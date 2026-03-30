import pytest
from fastapi.testclient import TestClient

from api.main import app


client = TestClient(app)


@pytest.fixture(autouse=True)
def disable_runtime_memory_writes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DISABLE_MEMORY_WRITE", "1")
    monkeypatch.setenv("DISABLE_MEMORY_CLEANUP", "1")


def test_health_endpoint_returns_ok() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_index_page_loads() -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert "AI Brain System v6" in response.text
    assert "Chat Mode" in response.text


def test_think_endpoint_returns_pipeline_result() -> None:
    response = client.post("/think", json={"problem": "Impact of AI on jobs"})

    assert response.status_code == 200
    payload = response.json()["result"]
    assert payload["problem"] == "Impact of AI on jobs"
    assert payload["final_answer"]
    assert payload["final_score"] >= payload["best_evaluation"]["total_score"]
