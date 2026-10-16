"""TensorRT-LLM backend.

Loads a compiled ``.engine`` plan (built by :mod:`engine.builder.trt_engine_builder`)
through TensorRT-LLM's C++ ``ModelRunnerCpp``, which supports inflight batching,
paged KV cache and INT8/FP16 kernels. The synchronous runner is executed in a
worker thread so the async event loop stays responsive.

TensorRT-LLM is imported lazily; this module is safe to import on CPU/CI.
"""
from __future__ import annotations

import asyncio
import os
import time
from typing import AsyncIterator

from engine.backends.base import BaseBackend, GenerationParams, RequestOutput
from engine.config import ModelConfig, ParallelConfig, TrtLlmConfig
from engine.utils.logging import get_logger

logger = get_logger("trtllm")


class TrtLlmBackend(BaseBackend):
    name = "trtllm"

    def __init__(
        self,
        model: ModelConfig,
        trt_cfg: TrtLlmConfig,
        parallel: ParallelConfig,
        *,
        rank: int | None = None,
    ) -> None:
        super().__init__(model.name_or_path)
        self.model_cfg = model
        self.cfg = trt_cfg
        self.parallel = parallel
        self.rank = rank if rank is not None else int(os.environ.get("RANK", "0"))
        self._runner = None
        self._tokenizer = None

    # ------------------------------------------------------------------ #
    async def load(self) -> None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            self.model_cfg.tokenizer or self.model_cfg.name_or_path,
            trust_remote_code=self.model_cfg.trust_remote_code,
        )
        runner = await asyncio.to_thread(self._build_runner)
        self._runner = runner
        self._tokenizer = tokenizer
        self._loaded = True
        logger.info(
            "TensorRT-LLM engine loaded: engine_dir=%s rank=%d quant=%s",
            self.cfg.engine_dir,
            self.rank,
            self.cfg.quantization,
        )

    def _build_runner(self):
        try:
            from tensorrt_llm.runtime import ModelRunnerCpp  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "tensorrt_llm is not installed. Build inside the provided Dockerfile "
                "or follow docs/tensorrt_llm_build.md."
            ) from exc

        return ModelRunnerCpp.from_dir(
            engine_dir=self.cfg.engine_dir,
            rank=self.rank,
            debug_mode=False,
            free_gpu_memory_fraction=self.cfg.kv_cache_free_gpu_mem_fraction,
            enable_kv_cache_reuse=self.cfg.enable_kv_cache_reuse,
            use_inflight_batching=self.cfg.use_inflight_batching,
            max_batch_size=self.cfg.max_batch_size,
            max_input_len=self.cfg.max_input_len,
            max_output_len=self.cfg.max_output_len,
        )

    # ------------------------------------------------------------------ #
    def _tokenize(self, prompt: str) -> list[int]:
        assert self._tokenizer is not None
        return self._tokenizer(prompt, add_special_tokens=False).input_ids

    def _runner_kwargs(self, params: GenerationParams) -> dict:
        return dict(
            max_new_tokens=params.max_new_tokens,
            temperature=max(params.temperature, 1e-5),
            top_p=params.top_p,
            repetition_penalty=params.repetition_penalty,
            end_id=self._tokenizer.eos_token_id if self._tokenizer else None,
            pad_id=self._tokenizer.pad_token_id if self._tokenizer else None,
        )

    async def generate(self, request_id: str, prompt: str, params: GenerationParams) -> RequestOutput:
        if self._runner is None:
            raise RuntimeError("Backend is not loaded")
        start = time.perf_counter()
        input_ids = self._tokenize(prompt)
        kwargs = self._runner_kwargs(params)

        def _run():
            return self._runner.generate([input_ids], **kwargs)

        result = await asyncio.to_thread(_run)
        first_token_at = time.perf_counter()  # runner.generate returns full sequence
        output_ids = list(result[0]["output_ids"][0][0])
        text = self._tokenizer.decode(output_ids, skip_special_tokens=True)
        total = (time.perf_counter() - start) * 1000
        ttft = (first_token_at - start) * 1000
        n = max(1, len(output_ids))
        out = RequestOutput(
            request_id=request_id,
            text=text,
            token_ids=output_ids,
            finish_reason="stop",
            prompt_tokens=len(input_ids),
            backend=self.name,
            ttft_ms=ttft,
            tpot_ms=max(0.0, (total - ttft) / n),
        )
        self._record_success(params, out, self.name)
        return out

    async def stream(self, request_id: str, prompt: str, params: GenerationParams) -> AsyncIterator[str]:
        if self._runner is None:
            raise RuntimeError("Backend is not loaded")
        # ModelRunnerCpp returns the completed sequence; emulate token streaming by
        # decoding incrementally. For true token-level streaming use the Executor API.
        input_ids = self._tokenize(prompt)
        kwargs = self._runner_kwargs(params)
        result = await asyncio.to_thread(
            lambda: self._runner.generate([input_ids], streaming=True, **kwargs)
        )
        output_ids = list(result[0]["output_ids"][0][0])
        last = 0
        for i in range(1, len(output_ids) + 1):
            text = self._tokenizer.decode(output_ids[:i], skip_special_tokens=True)
            delta = text[last:]
            last = len(text)
            if delta:
                yield delta

    async def shutdown(self) -> None:
        logger.info("Shutting down TensorRT-LLM backend")
        self._runner = None
        self._loaded = False
