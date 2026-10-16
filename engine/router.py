"""Hybrid backend router.

Decides, per request, whether the compiled TensorRT-LLM engine or the vLLM
Continuous Batching engine should serve it. The decision exploits the
complementary strengths observed in the benchmarks:

* TensorRT-LLM (fused kernels, INT8/FP16) wins on **compute-heavy prefill** of
  long prompts and on deterministic low-latency single streams.
* vLLM (PagedAttention, Continuous Batching) wins when many requests share the
  decode batch and scheduling flexibility matters.
"""
from __future__ import annotations

from dataclasses import dataclass

from engine.backends.base import GenerationParams
from engine.config import RouterConfig
from engine.utils.logging import get_logger

logger = get_logger("router")


@dataclass
class BackendLoad:
    inflight: int = 0
    capacity: int = 256

    @property
    def utilization(self) -> float:
        return self.inflight / self.capacity if self.capacity else 1.0


class HybridRouter:
    def __init__(self, config: RouterConfig) -> None:
        self.config = config
        self._stats = {"trtllm": 0, "vllm": 0}

    def select(
        self,
        *,
        prompt_tokens: int,
        params: GenerationParams,
        trtllm: BackendLoad,
        vllm: BackendLoad,
    ) -> str:
        strategy = self.config.strategy

        if strategy == "fixed":
            return "trtllm"

        if strategy == "load_balance":
            return "trtllm" if trtllm.utilization <= vllm.utilization else "vllm"

        # Default: latency_aware.
        # Long / compute-heavy prefills -> fused + quantized TRT kernels.
        if prompt_tokens >= self.config.trtllm_min_prefill_tokens:
            if trtllm.utilization < 1.0:
                choice = "trtllm"
            elif self.config.spill_to_vllm_on_overload:
                choice = "vllm"
            else:
                choice = "trtllm"
        else:
            # Short prompts mixed with high concurrency -> vLLM batching.
            choice = "vllm"

        # Overload spill-over regardless of prompt length.
        if choice == "trtllm" and trtllm.utilization >= 1.0:
            choice = "vllm" if self.config.spill_to_vllm_on_overload else "trtllm"

        self._stats[choice] += 1
        logger.debug(
            "route prompt_tokens=%d -> %s (trt_util=%.2f vllm_util=%.2f)",
            prompt_tokens,
            choice,
            trtllm.utilization,
            vllm.utilization,
        )
        return choice

    def routing_stats(self) -> dict[str, int]:
        return dict(self._stats)
