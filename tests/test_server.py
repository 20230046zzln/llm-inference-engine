"""End-to-end tests of the FastAPI app in mock mode (no GPU required).

Uses Starlette's ``TestClient`` (an httpx client) which runs the lifespan
startup/shutdown, loading the backend exactly as in production.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from engine.config import EngineConfig
from engine.server import create_app


@pytest.fixture
def client():
    cfg = EngineConfig(mode="mock")
    with TestClient(create_app(cfg)) as c:
        yield c


def test_health(client: TestClient):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
    assert resp.json()["backend"] == "mock"


def test_models(client: TestClient):
    resp = client.get("/v1/models")
    assert resp.status_code == 200
    assert resp.json()["data"][0]["backend"] == "mock"


def test_chat_completion_non_stream(client: TestClient):
    resp = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "say hello"}],
            "max_tokens": 8,
            "stream": False,
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["choices"][0]["message"]["content"]
    assert body["usage"]["completion_tokens"] == 8


def test_chat_completion_stream(client: TestClient):
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "stream please"}],
            "max_tokens": 6,
            "stream": True,
        },
    ) as resp:
        assert resp.status_code == 200
        lines = list(resp.iter_lines())
    assert any("assistant" in line for line in lines)
    assert any("[DONE]" in line for line in lines)


def test_completions_endpoint(client: TestClient):
    resp = client.post(
        "/v1/completions",
        json={"prompt": "once upon a time", "max_tokens": 5, "stream": False},
    )
    assert resp.status_code == 200
    assert resp.json()["choices"][0]["text"]
