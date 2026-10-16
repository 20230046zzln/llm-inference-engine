"""Tests for the mock backend and backend factory."""
from __future__ import annotations

import pytest

from engine.backends import MockBackend, build_backend
from engine.backends.base import GenerationParams
from engine.config import EngineConfig


@pytest.mark.asyncio
async def test_mock_backend_generate():
    backend = MockBackend()
    await backend.load()
    out = await backend.generate("r1", "hello world prompt", GenerationParams(max_new_tokens=5))
    assert out.backend == "mock"
    assert out.completion_tokens == 5
    assert out.text


@pytest.mark.asyncio
async def test_mock_backend_stream():
    backend = MockBackend()
    await backend.load()
    chunks = [c async for c in backend.stream("r2", "hi", GenerationParams(max_new_tokens=4))]
    assert len(chunks) == 4


def test_factory_modes():
    assert isinstance(build_backend(EngineConfig(mode="mock")), MockBackend)

    # GPU backends construct without a GPU (lazy import).
    from engine.backends.trtllm_backend import TrtLlmBackend
    from engine.backends.vllm_backend import VllmBackend

    assert isinstance(build_backend(EngineConfig(mode="vllm")), VllmBackend)
    assert isinstance(build_backend(EngineConfig(mode="trtllm")), TrtLlmBackend)


@pytest.mark.asyncio
async def test_health():
    backend = MockBackend()
    assert not await backend.health()
    await backend.load()
    assert await backend.health()
