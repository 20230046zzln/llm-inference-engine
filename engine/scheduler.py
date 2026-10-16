"""Continuous Batching scheduler (iteration-level).

This is the scheduling core ported from the design used by vLLM and reimplemented
here so the batching policy can be unit-tested without a GPU and reused by both
the vLLM and TensorRT-LLM backends.

Key properties
--------------
* **Iteration-level scheduling**: the batch is re-evaluated before *every* decode
  step. New requests are admitted and finished requests are removed immediately,
  rather than waiting for a whole static batch to finish.
* **Chunked prefill**: long prompts are split into ``prefill_chunk_size`` chunks so
  a single long prefill cannot block the decode batch.
* **Mixed prefill + decode**: one engine step can contain both prefill chunks and
  one-token decode work (when ``prefill_decode_mixed``).
* **Admission control + preemption**: sequences occupy KV blocks; when capacity is
  exhausted the newest decodable sequences are preempted and recomputed later.
"""
from __future__ import annotations

import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Iterable

# Default wall clock; benchmarks inject a simulated (virtual) clock.
_REAL_CLOCK = time.perf_counter

from engine.config import SchedulerConfig
from engine.utils.logging import get_logger
from engine.utils import metrics as M

logger = get_logger("scheduler")


class SequenceStatus(str, Enum):
    WAITING = "waiting"
    RUNNING_PREFILL = "running_prefill"
    RUNNING_DECODING = "running_decoding"
    FINISHED_STOPPED = "finished_stop"
    FINISHED_LENGTH = "finished_length"


# A callable that "executes" one iteration: it receives prefill tasks
# (seq, chunk_len) and decode seqs, and returns {seq_id: [new_token_ids]}.
# Real backends plug the engine forward pass here; tests pass a mock.
GenerateFn = Callable[
    ["list[tuple['Sequence', int]]", "list['Sequence']"],
    dict[str, list[int]],
]


class Sequence:
    """A single generation request and its mutable runtime state."""

    __slots__ = (
        "id",
        "prompt_token_ids",
        "output_token_ids",
        "status",
        "max_new_tokens",
        "priority",
        "arrival_time",
        "first_token_time",
        "finish_time",
        "num_computed_tokens",
        "stop_token_ids",
        "num_preemptions",
    )

    def __init__(
        self,
        prompt_token_ids: list[int],
        *,
        max_new_tokens: int = 512,
        priority: int = 0,
        stop_token_ids: Iterable[int] | None = None,
        seq_id: str | None = None,
        arrival_time: float | None = None,
    ) -> None:
        self.id = seq_id or uuid.uuid4().hex[:12]
        self.prompt_token_ids = list(prompt_token_ids)
        self.output_token_ids: list[int] = []
        self.status = SequenceStatus.WAITING
        self.max_new_tokens = max_new_tokens
        self.priority = priority
        self.arrival_time = arrival_time if arrival_time is not None else time.perf_counter()
        self.first_token_time: float | None = None
        self.finish_time: float | None = None
        self.num_computed_tokens = 0
        self.stop_token_ids = set(stop_token_ids or [])
        self.num_preemptions = 0

    # ------------------------------------------------------------------ #
    @property
    def prompt_len(self) -> int:
        return len(self.prompt_token_ids)

    @property
    def seq_len(self) -> int:
        return self.prompt_len + len(self.output_token_ids)

    @property
    def num_kv_tokens(self) -> int:
        """KV tokens currently resident (computed prompt + generated output)."""
        return self.num_computed_tokens

    @property
    def is_finished(self) -> bool:
        return self.status in (
            SequenceStatus.FINISHED_STOPPED,
            SequenceStatus.FINISHED_LENGTH,
        )

    @property
    def ttft(self) -> float | None:
        if self.first_token_time is None:
            return None
        return (self.first_token_time - self.arrival_time) * 1000.0

    def remaining_prefill(self) -> int:
        return self.prompt_len - self.num_computed_tokens

    def mark_first_token(self, now: float) -> None:
        if self.first_token_time is None:
            self.first_token_time = now
            ttft = self.ttft
            if ttft is not None:
                M.TTFT_HIST.labels(backend="engine").observe(ttft)


@dataclass
class SchedulerOutput:
    """The plan for one engine iteration."""

    prefill_tasks: list[tuple[Sequence, int]] = field(default_factory=list)
    decode_seqs: list[Sequence] = field(default_factory=list)
    preempted: list[Sequence] = field(default_factory=list)
    finished: list[Sequence] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not (self.prefill_tasks or self.decode_seqs)

    @property
    def num_batched_tokens(self) -> int:
        return sum(chunk for _, chunk in self.prefill_tasks) + len(self.decode_seqs)


class ContinuousBatchingScheduler:
    """Iteration-level Continuous Batching scheduler.

    Parameters
    ----------
    config:
        Scheduler configuration.
    kv_cache_token_capacity:
        Maximum number of KV tokens that can simultaneously reside across the
        batch (models the GPU KV cache size).
    """

    def __init__(
        self,
        config: SchedulerConfig | None = None,
        *,
        kv_cache_token_capacity: int = 256 * 2048,
        clock: Callable[[], float] = _REAL_CLOCK,
    ) -> None:
        self.config = config or SchedulerConfig()
        self.waiting: deque[Sequence] = deque()
        self.running: dict[str, Sequence] = {}
        self.kv_capacity = kv_cache_token_capacity
        self._steps = 0
        self._used_kv_tokens = 0
        self._clock = clock

    # ------------------------------------------------------------------ #
    # Admission
    # ------------------------------------------------------------------ #
    def add_sequence(self, seq: Sequence) -> None:
        if len(self.waiting) >= self.config.max_num_waiting:
            raise RuntimeError("Waiting queue is full; backpressure should be applied upstream.")
        seq.status = SequenceStatus.WAITING
        self.waiting.append(seq)
        M.WAITING_SEQS.set(len(self.waiting))
        logger.debug("admit seq=%s prompt_len=%d", seq.id, seq.prompt_len)

    def add_request(
        self,
        prompt_token_ids: list[int],
        *,
        max_new_tokens: int = 512,
        priority: int = 0,
        stop_token_ids: Iterable[int] | None = None,
    ) -> Sequence:
        seq = Sequence(
            prompt_token_ids,
            max_new_tokens=max_new_tokens,
            priority=priority,
            stop_token_ids=stop_token_ids,
            arrival_time=self._clock(),
        )
        self.add_sequence(seq)
        return seq

    # ------------------------------------------------------------------ #
    # Scheduling
    # ------------------------------------------------------------------ #
    @staticmethod
    def _resident(seq: Sequence) -> int:
        """KV tokens currently resident for a sequence.

        Partial prefill: ``num_computed``. Completed prefill: the full prompt
        length plus every generated output token.
        """
        return seq.num_computed_tokens + len(seq.output_token_ids)

    def schedule(self) -> SchedulerOutput:
        """Compute the next iteration's batch. Must be followed by :meth:`update`."""
        self._steps += 1
        out = SchedulerOutput()

        # Ground-truth resident KV across the running set (recomputed each step).
        used = sum(self._resident(s) for s in self.running.values())

        # 1) All currently decodable sequences want exactly 1 token this step.
        decoding = [
            s for s in self.running.values() if s.status == SequenceStatus.RUNNING_DECODING
        ]
        prefill_in_flight = [
            s for s in self.running.values() if s.status == SequenceStatus.RUNNING_PREFILL
        ]

        # A decode step occupies one token of the mixed batch budget.
        decode_tokens = len(decoding) if self.config.prefill_decode_mixed else 0
        token_budget = self.config.prefill_chunk_size
        if self.config.prefill_decode_mixed:
            token_budget = max(0, self.config.prefill_chunk_size - decode_tokens)

        # 2) Continue prefill chunks already in flight (their KV becomes resident).
        for seq in prefill_in_flight:
            if token_budget <= 0:
                break
            chunk = min(seq.remaining_prefill(), token_budget)
            out.prefill_tasks.append((seq, chunk))
            used += chunk
            token_budget -= chunk

        # 3) Order the waiting queue according to policy, then admit.
        self._order_waiting()
        while self.waiting and token_budget > 0:
            if len(self.running) >= self.config.max_num_running:
                break
            seq = self.waiting[0]
            chunk = min(seq.remaining_prefill(), token_budget)
            if chunk <= 0:
                break

            # KV capacity check; preempt decodable sequences to make room.
            if used + chunk > self.kv_capacity:
                preempted = self._preempt_for(chunk)
                out.preempted.extend(preempted)
                used = sum(self._resident(s) for s in self.running.values())
                if used + chunk > self.kv_capacity:
                    break  # could not reclaim enough; wait for decodes to finish

            self.waiting.popleft()
            seq.status = SequenceStatus.RUNNING_PREFILL
            self.running[seq.id] = seq
            used += chunk
            out.prefill_tasks.append((seq, chunk))
            token_budget -= chunk

        self._used_kv_tokens = used

        # 4) Decode work is appended (mixed batching).
        if self.config.prefill_decode_mixed:
            out.decode_seqs = decoding
        else:
            # Prefill-dominant step: only decode if no prefill work remains.
            out.decode_seqs = [] if out.prefill_tasks else decoding

        M.BATCH_SIZE.set(len(out.decode_seqs) + len(out.prefill_tasks))
        M.RUNNING_SEQS.set(len(self.running))
        M.WAITING_SEQS.set(len(self.waiting))
        logger.debug(
            "step=%d prefill=%d decode=%d waiting=%d kv_util=%.2f",
            self._steps,
            len(out.prefill_tasks),
            len(out.decode_seqs),
            len(self.waiting),
            self.kv_utilization(),
        )
        return out

    def update(
        self,
        out: SchedulerOutput,
        generated: dict[str, list[int]],
        *,
        now: float | None = None,
    ) -> list[Sequence]:
        """Apply engine results for an iteration; return newly finished sequences."""
        now = now if now is not None else self._clock()
        finished: list[Sequence] = []

        # Prefill tasks: advance computed offsets; a completed prefill yields the
        # first output token (produced in the same forward pass).
        for seq, chunk in out.prefill_tasks:
            seq.num_computed_tokens += chunk
            new_tokens = generated.get(seq.id, [])
            if seq.remaining_prefill() == 0:
                seq.status = SequenceStatus.RUNNING_DECODING
                if new_tokens:
                    self._append_tokens(seq, new_tokens, now)
                else:
                    seq.mark_first_token(now)
            else:
                # partial chunk: no output tokens, KV for the chunk is resident.
                pass

        # Decode steps: each sequence produced one (or more) token.
        for seq in out.decode_seqs:
            new_tokens = generated.get(seq.id, [])
            if new_tokens:
                self._append_tokens(seq, new_tokens, now)

        # Finalize anything that stopped. KV accounting is recomputed in schedule().
        for seq in list(self.running.values()):
            reason = self._check_finish(seq)
            if reason is not None:
                seq.status = reason
                seq.finish_time = now
                del self.running[seq.id]
                finished.append(seq)
                M.REQUESTS_TOTAL.labels(backend="engine", model="default", status="ok").inc()

        out.finished = finished
        return finished

    # ------------------------------------------------------------------ #
    # Full driving loop (used by simulation / tests / offline backends)
    # ------------------------------------------------------------------ #
    def run_iteration(self, generate: GenerateFn) -> list[Sequence]:
        out = self.schedule()
        if out.is_empty:
            return []
        generated = generate(out.prefill_tasks, out.decode_seqs)
        return self.update(out, generated)

    def run_until_idle(self, generate: GenerateFn, max_steps: int = 100_000) -> list[Sequence]:
        all_finished: list[Sequence] = []
        while self.waiting or self.running:
            finished = self.run_iteration(generate)
            all_finished.extend(finished)
            if self._steps >= max_steps:
                raise RuntimeError("Scheduler exceeded max_steps (possible deadlock).")
        return all_finished

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _append_tokens(self, seq: Sequence, tokens: list[int], now: float) -> None:
        for tok in tokens:
            seq.output_token_ids.append(tok)
            M.THROUGHPUT_TOKENS.labels(backend="engine").inc()
        seq.mark_first_token(now)

    def _check_finish(self, seq: Sequence) -> SequenceStatus | None:
        if seq.status not in (SequenceStatus.RUNNING_DECODING, SequenceStatus.RUNNING_PREFILL):
            return None
        if seq.output_token_ids and seq.output_token_ids[-1] in seq.stop_token_ids:
            return SequenceStatus.FINISHED_STOPPED
        if len(seq.output_token_ids) >= seq.max_new_tokens:
            return SequenceStatus.FINISHED_LENGTH
        return None

    def _order_waiting(self) -> None:
        if self.config.scheduling_policy in ("fcfs", "latency_aware"):
            # latency_aware prioritizes long prompts (they dominate TTFT) then age.
            if self.config.scheduling_policy == "latency_aware":
                ordered = sorted(
                    self.waiting,
                    key=lambda s: (-s.priority, -s.prompt_len, s.arrival_time),
                )
            else:
                ordered = sorted(self.waiting, key=lambda s: (-s.priority, s.arrival_time))
            self.waiting = deque(ordered)

    def _preempt_for(self, required_tokens: int) -> list[Sequence]:
        """Reclaim KV by preempting the most recently admitted decodable sequences.

        Emulates vLLM's RECOMPUTE preemption: the sequence returns to the head of
        the waiting queue and its prompt is recomputed later.
        """
        victims = sorted(
            (s for s in self.running.values() if s.status == SequenceStatus.RUNNING_DECODING),
            key=lambda s: (s.priority, -s.arrival_time),
        )
        preempted: list[Sequence] = []
        reclaimed = 0
        for seq in victims:
            if self._used_kv_tokens - reclaimed + required_tokens <= self.kv_capacity:
                break
            reclaimed += self._resident(seq)
            seq.num_preemptions += 1
            seq.status = SequenceStatus.WAITING
            seq.num_computed_tokens = 0
            seq.output_token_ids.clear()
            seq.first_token_time = None
            preempted.append(seq)
            self.waiting.appendleft(seq)

        for seq in preempted:
            self.running.pop(seq.id, None)
        self._used_kv_tokens -= reclaimed
        if preempted:
            logger.info("preempted %d sequence(s) to free %d KV tokens", len(preempted), reclaimed)
        return preempted

    # ------------------------------------------------------------------ #
    @property
    def num_steps(self) -> int:
        return self._steps

    def kv_utilization(self) -> float:
        return self._used_kv_tokens / self.kv_capacity if self.kv_capacity else 0.0

    def drain_finished_metrics(self) -> dict[str, float]:
        return {
            "running": len(self.running),
            "waiting": len(self.waiting),
            "kv_util": self.kv_utilization(),
            "steps": self._steps,
        }
