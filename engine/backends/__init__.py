"""Backend factory."""
from __future__ import annotations

from engine.backends.base import BaseBackend, GenerationParams, MockBackend, RequestOutput
from engine.config import EngineConfig


def build_backend(config: EngineConfig) -> BaseBackend:
    """Instantiate the backend described by ``config.mode``.

    Importing concrete GPU backends is deferred until load time, so constructing
    them never requires a GPU.
    """
    mode = config.mode
    if mode == "mock":
        return MockBackend()
    if mode == "vllm":
        from engine.backends.vllm_backend import VllmBackend

        return VllmBackend(config.model, config.vllm, config.parallel)
    if mode == "trtllm":
        from engine.backends.trtllm_backend import TrtLlmBackend

        return TrtLlmBackend(config.model, config.trtllm, config.parallel)
    if mode == "hybrid":
        from engine.backends.hybrid_backend import HybridBackend

        return HybridBackend(config)
    raise ValueError(f"Unknown backend mode: {mode}")


__all__ = [
    "BaseBackend",
    "GenerationParams",
    "RequestOutput",
    "MockBackend",
    "build_backend",
]
