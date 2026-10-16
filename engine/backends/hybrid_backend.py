"""Fused hybrid backend.

Owns a :class:`VllmBackend` and a :class:`TrtLlmBackend` and dispatches each
request through :class:`HybridRouter`. Both engines share the same GPU pool and
NCCL process group; the scheduler/backends keep their own KV partitions sized by
the configs.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from engine.backends.base import BaseBackend, GenerationParams, RequestOutput
from engine.backends.trtllm_backend import TrtLlmBackend
from engine.backends.vllm_backend import VllmBackend
from engine.config import EngineConfig
from engine.router import BackendLoad, HybridRouter
from engine.utils.logging import get_logger

logger = get_logger("hybrid")


class HybridBackend(BaseBackend):
    name = "hybrid"

    def __init__(self, config: EngineConfig) -> None:
        super().__init__(config.model.name_or_path)
        self.config = config
        self.vllm = VllmBackend(config.model, config.vllm, config.parallel)
        self.trtllm = TrtLlmBackend(config.model, config.trtllm, config.parallel)
        self.router = HybridRouter(config.router)
        self._inflight = {"trtllm": 0, "vllm": 0}

    async def load(self) -> None:
        # TensorRT-LLM plan is loaded first; vLLM then sizes its KV cache from the
        # remaining memory. In practice these can be enabled one at a time too.
        try:
            await self.trtllm.load()
        except Exception as exc:  # pragma: no cover - environment dependent
            logger.warning("TRT-LLM backend unavailable (%s); running vLLM only", exc)
        try:
            await self.vllm.load()
        except Exception as exc:  # pragma: no cover - environment dependent
            logger.warning("vLLM backend unavailable (%s)", exc)
        self._loaded = self.trtllm._loaded or self.vllm._loaded
        logger.info("Hybrid backend ready (trt=%s, vllm=%s)", self.trtllm._loaded, self.vllm._loaded)

    # ------------------------------------------------------------------ #
    def _loads(self) -> tuple[BackendLoad, BackendLoad]:
        trt = BackendLoad(self._inflight["trtllm"], self.config.trtllm.max_batch_size)
        vllm = BackendLoad(self._inflight["vllm"], self.config.vllm.max_num_seqs)
        return trt, vllm

    async def _prompt_tokens(self, backend: BaseBackend, prompt: str) -> int:
        tok = getattr(backend, "_tokenizer", None)
        if tok is not None:
            return len(tok(prompt, add_special_tokens=False).input_ids)
        return len(prompt.split())

    @asynccontextmanager
    async def _track(self, name: str):
        self._inflight[name] += 1
        try:
            yield
        finally:
            self._inflight[name] -= 1

    async def _choose(self, prompt: str, params: GenerationParams) -> tuple[str, BaseBackend]:
        # Prefer a tokenizer that is loaded; fall back across engines.
        ref = self.trtllm if self.trtllm._tokenizer else self.vllm
        n = await self._prompt_tokens(ref, prompt)
        trt_load, vllm_load = self._loads()
        name = self.router.select(
            prompt_tokens=n, params=params, trtllm=trt_load, vllm=vllm_load
        )
        backend = self.trtllm if name == "trtllm" else self.vllm
        if not backend._loaded:  # fail over to whichever is available
            backend = self.vllm if self.vllm._loaded else self.trtllm
            name = backend.name
        return name, backend

    async def generate(self, request_id: str, prompt: str, params: GenerationParams) -> RequestOutput:
        name, backend = await self._choose(prompt, params)
        async with self._track(name):
            return await backend.generate(request_id, prompt, params)

    async def stream(self, request_id: str, prompt: str, params: GenerationParams) -> AsyncIterator[str]:
        name, backend = await self._choose(prompt, params)
        async with self._track(name):
            async for delta in backend.stream(request_id, prompt, params):
                yield delta

    async def health(self) -> bool:
        return await self.trtllm.health() or await self.vllm.health()

    async def shutdown(self) -> None:
        await self.trtllm.shutdown()
        await self.vllm.shutdown()
        self._loaded = False

    def stats(self) -> dict[str, float]:
        return {
            "inflight_trtllm": float(self._inflight["trtllm"]),
            "inflight_vllm": float(self._inflight["vllm"]),
            **{f"trtllm_{k}": v for k, v in self.trtllm.stats().items()},
            **{f"vllm_{k}": v for k, v in self.vllm.stats().items()},
        }
