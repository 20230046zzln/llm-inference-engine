"""Offline (no server) inference against a backend.

For a quick CPU run without a GPU, force the mock backend via mode ``mock``.
"""
from __future__ import annotations

import asyncio

from engine.backends import build_backend
from engine.backends.base import GenerationParams
from engine.config import EngineConfig


async def main() -> None:
    # Toggle to "vllm" / "trtllm" / "hybrid" on a GPU host.
    cfg = EngineConfig(mode="mock")
    backend = build_backend(cfg)
    await backend.load()

    params = GenerationParams(max_new_tokens=16, temperature=0.0)
    output = await backend.generate("demo-1", "hello world from the inference engine", params)
    print(f"[{output.backend}] {output.text}")
    print(f"tokens={output.completion_tokens} ttft={output.ttft_ms:.2f}ms")

    await backend.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
