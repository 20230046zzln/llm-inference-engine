"""vLLM backend.

Wraps vLLM's async engine, which provides Continuous Batching + PagedAttention
natively. The engine is imported lazily so this module stays importable on
machines (and CI runners) without a GPU.
"""
from __future__ import annotations

import asyncio
import uuid
from typing import AsyncIterator

from engine.backends.base import BaseBackend, GenerationParams, RequestOutput
from engine.config import ModelConfig, ParallelConfig, VllmConfig
from engine.utils.logging import get_logger

logger = get_logger("vllm")


class VllmBackend(BaseBackend):
    name = "vllm"

    def __init__(
        self,
        model: ModelConfig,
        vllm_cfg: VllmConfig,
        parallel: ParallelConfig,
    ) -> None:
        super().__init__(model.name_or_path)
        self.model_cfg = model
        self.cfg = vllm_cfg
        self.parallel = parallel
        self._engine = None
        self._tokenizer = None

    # ------------------------------------------------------------------ #
    async def load(self) -> None:
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("transformers is required to build a tokenizer") from exc

        tp = self.parallel.tensor_parallel_size
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_cfg.tokenizer or self.model_cfg.name_or_path,
            trust_remote_code=self.model_cfg.trust_remote_code,
        )

        def _build():
            return self._build_engine(tp)

        self._engine = await asyncio.to_thread(_build)
        self._tokenizer = tokenizer
        self._loaded = True
        logger.info(
            "vLLM engine loaded: model=%s tp=%d max_num_seqs=%d",
            self.model_cfg.name_or_path,
            tp,
            self.cfg.max_num_seqs,
        )

    def _build_engine(self, tp: int):
        # vLLM >= 0.7 ships AsyncLLM; older versions expose AsyncLLMEngine.
        try:
            from vllm import AsyncLLM  # type: ignore

            return AsyncLLM(
                model=self.model_cfg.name_or_path,
                dtype=self.cfg.dtype,
                tensor_parallel_size=tp,
                gpu_memory_utilization=self.cfg.gpu_memory_utilization,
                max_num_seqs=self.cfg.max_num_seqs,
                max_num_batched_tokens=self.cfg.max_num_batched_tokens,
                enable_prefix_caching=self.cfg.enable_prefix_caching,
                enforce_eager=self.cfg.enforce_eager,
                trust_remote_code=self.model_cfg.trust_remote_code,
            )
        except ImportError:
            from vllm.engine.arg_utils import AsyncEngineArgs  # type: ignore
            from vllm.engine.async_llm_engine import AsyncLLMEngine  # type: ignore

            args = AsyncEngineArgs(
                model=self.model_cfg.name_or_path,
                dtype=self.cfg.dtype,
                tensor_parallel_size=tp,
                gpu_memory_utilization=self.cfg.gpu_memory_utilization,
                max_num_seqs=self.cfg.max_num_seqs,
                max_num_batched_tokens=self.cfg.max_num_batched_tokens,
                enable_prefix_caching=self.cfg.enable_prefix_caching,
                enforce_eager=self.cfg.enforce_eager,
                trust_remote_code=self.model_cfg.trust_remote_code,
            )
            return AsyncLLMEngine.from_engine_args(args)

    # ------------------------------------------------------------------ #
    def _sampling_params(self, params: GenerationParams):
        from vllm import SamplingParams  # type: ignore

        return SamplingParams(
            max_tokens=params.max_new_tokens,
            temperature=params.temperature,
            top_p=params.top_p,
            repetition_penalty=params.repetition_penalty,
            stop_token_ids=params.stop_token_ids or None,
            stop=params.stop_strings or None,
            seed=params.seed,
        )

    def _prompt_tokens(self, prompt: str) -> int:
        if self._tokenizer is None:
            return len(prompt.split())
        return len(self._tokenizer(prompt, add_special_tokens=False).input_ids)

    async def generate(self, request_id: str, prompt: str, params: GenerationParams) -> RequestOutput:
        if self._engine is None:
            raise RuntimeError("Backend is not loaded")
        import time

        start = time.perf_counter()
        sampling = self._sampling_params(params)
        last = None
        first_token_at: float | None = None
        async for output in self._engine.generate(prompt, sampling, request_id):
            if first_token_at is None and output.outputs and output.outputs[0].token_ids:
                first_token_at = time.perf_counter()
            last = output

        if last is None:
            return RequestOutput(request_id=request_id, backend=self.name)

        result = last.outputs[0]
        ttft = (first_token_at - start) * 1000 if first_token_at else None
        total = (time.perf_counter() - start) * 1000
        n = max(1, len(result.token_ids))
        tpot = (total - (ttft or 0)) / n
        out = RequestOutput(
            request_id=request_id,
            text=result.text,
            token_ids=list(result.token_ids),
            finish_reason=result.finish_reason or "stop",
            prompt_tokens=len(last.prompt_token_ids),
            backend=self.name,
            ttft_ms=ttft,
            tpot_ms=tpot,
        )
        self._record_success(params, out, self.name)
        return out

    async def stream(
        self, request_id: str, prompt: str, params: GenerationParams
    ) -> AsyncIterator[str]:
        if self._engine is None:
            raise RuntimeError("Backend is not loaded")
        sampling = self._sampling_params(params)
        last_len = 0
        async for output in self._engine.generate(prompt, sampling, request_id):
            text = output.outputs[0].text
            delta = text[last_len:]
            last_len = len(text)
            if delta:
                yield delta

    async def shutdown(self) -> None:
        logger.info("Shutting down vLLM backend")
        self._engine = None
        self._loaded = False
