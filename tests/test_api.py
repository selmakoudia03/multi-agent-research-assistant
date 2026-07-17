from fastapi.testclient import TestClient

from magent.api import app

client = TestClient(app)


def test_health_endpoint() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_chat_endpoint_returns_full_trace() -> None:
    response = client.post("/chat", json={"query": "What is RAG?"})
    assert response.status_code == 200

    body = response.json()
    assert body["answer"]
    assert body["plan"]
    assert body["verdict"] in {"approve", "revise"}
    assert body["trace"]


def test_chat_endpoint_rejects_missing_query() -> None:
    response = client.post("/chat", json={})
    assert response.status_code == 422
