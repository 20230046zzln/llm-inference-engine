"""llm-inference-engine

A high-performance LLM serving engine that fuses **vLLM Continuous Batching**
with **TensorRT-LLM** compiled/quantized kernels and NCCL tensor parallelism.

Public surface:
    >>> from engine import EngineConfig, ServingEngine
    >>> from engine.scheduler import ContinuousBatchingScheduler
"""
from __future__ import annotations

__version__ = "0.1.0"

from engine.config import EngineConfig, load_config

__all__ = [
    "EngineConfig",
    "load_config",
    "__version__",
]
