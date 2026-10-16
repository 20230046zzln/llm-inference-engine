"""Backend abstraction shared by the vLLM, TensorRT-LLM and fused engines."""
from __future__ import annotations

import abc
import time
from dataclasses import dataclass, field
from typing import AsyncIterator, Iterable

from engine.utils import metrics as M
from engine.utils.logging import get_logger

logger = get_logger("backends")


@dataclass
class GenerationParams:
    """Sampling parameters (backend-agnostic)."""

    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 1.0
    repetition_penalty: float = 1.0
    stop_token_ids: list[int] = field(default_factory=list)
    stop_strings: list[str] = field(default_factory=list)
    seed: int | None = None
    priority: int = 0


@dataclass
class RequestOutput:
    request_id: str
    text: str = ""
    token_ids: list[int] = field(default_factory=list)
    finish_reason: str = "stop"
    prompt_tokens: int = 0
    backend: str = "unknown"
    ttft_ms: float | None = None
    tpot_ms: float | None = None

    @property
    def completion_tokens(self) -> int:
        return len(self.token_ids)


class BaseBackend(abc.ABC):
    """Common interface every inference backend implements."""

    name: str = "base"

    def __init__(self, model_name: str = "default") -> None:
        self.model_name = model_name
        self._loaded = False
        self._num_requests = 0
        self._num_tokens = 0

    # ------------------------------------------------------------------ #
    @abc.abstractmethod
    async def load(self) -> None:
        """Allocate / load the engine (weights, KV cache, compiled plan)."""

    @abc.abstractmethod
    async def generate(self, request_id: str, prompt: str, params: GenerationParams) -> RequestOutput:
        """Run a single non-streaming request."""

    @abc.abstractmethod
    def stream(
        self, request_id: str, prompt: str, params: GenerationParams
    ) -> AsyncIterator[str]:
        """Stream decoded text deltas (async iterator)."""

    async def health(self) -> bool:
        return self._loaded

    async def shutdown(self) -> None:  # pragma: no cover - overridden
        self._loaded = False

    # ------------------------------------------------------------------ #
    def stats(self) -> dict[str, float]:
        return {
            "loaded": float(self._loaded),
            "requests": float(self._num_requests),
            "generated_tokens": float(self._num_tokens),
        }

    def _record_success(self, params: GenerationParams, output: RequestOutput, backend: str) -> None:
        self._num_requests += 1
        self._num_tokens += output.completion_tokens
        M.REQUESTS_TOTAL.labels(backend=backend, model=self.model_name, status="ok").inc()
        M.THROUGHPUT_TOKENS.labels(backend=backend).inc(output.completion_tokens)
        if output.ttft_ms is not None:
            M.TTFT_HIST.labels(backend=backend).observe(output.ttft_ms)
        if output.tpot_ms is not None:
            M.TPOT_HIST.labels(backend=backend).observe(output.tpot_ms)


class MockBackend(BaseBackend):
    """Deterministic backend used for unit tests and CPU CI.

    It tokenizes by whitespace, emits a fixed number of "tokens" (integer word
    hashes) and drives the real Continuous Batching scheduler, so the serving
    path and batching logic are exercised end-to-end without a GPU.
    """

    name = "mock"

    def __init__(self, model_name: str = "mock", *, words: tuple[str, ...] = ()) -> None:
        super().__init__(model_name)
        self._words = words or ("the", "model", "infers", "a", "token", "quickly", "done", "<eos>")
        self._eos_id = len(self._words) - 1

    def tokenize(self, text: str) -> list[int]:
        # One token per word, mapped deterministically to the mock vocabulary.
        return [hash(w) % (len(self._words) - 1) for w in text.split()] or [0]

    def detokenize(self, ids: Iterable[int]) -> str:
        return " ".join(self._words[i % len(self._words)] for i in ids)

    async def load(self) -> None:
        self._loaded = True

    async def generate(self, request_id: str, prompt: str, params: GenerationParams) -> RequestOutput:
        start = time.perf_counter()
        prompt_ids = self.tokenize(prompt)
        n = params.max_new_tokens
        ids = [(i * 7 + 1) % (len(self._words) - 1) for i in range(n)]
        ttft = (time.perf_counter() - start) * 1000
        out = RequestOutput(
            request_id=request_id,
            text=self.detokenize(ids),
            token_ids=ids,
            prompt_tokens=len(prompt_ids),
            backend=self.name,
            ttft_ms=ttft,
            tpot_ms=1.0,
        )
        self._record_success(params, out, self.name)
        return out

    async def stream(self, request_id: str, prompt: str, params: GenerationParams) -> AsyncIterator[str]:
        prompt_ids = self.tokenize(prompt)
        for i in range(params.max_new_tokens):
            tok = (i * 7 + 1) % (len(self._words) - 1)
            yield self._words[tok] + " "

    def __repr__(self) -> str:
        return f"<MockBackend loaded={self._loaded}>"
