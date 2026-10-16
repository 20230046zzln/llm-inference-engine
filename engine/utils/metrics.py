"""Prometheus metrics for the serving engine.

Importing this module without prometheus-client installed degrades gracefully to
no-op counters so the package stays usable on minimal installs.
"""
from __future__ import annotations

try:
    from prometheus_client import Counter, Gauge, Histogram

    HAS_PROM = True
except Exception:  # pragma: no cover - optional dependency
    HAS_PROM = False

    class _Noop:
        def labels(self, **_kw):
            return self

        def inc(self, *_a, **_kw):
            pass

        def observe(self, *_a, **_kw):
            pass

        def set(self, *_a, **_kw):
            pass

    Counter = Gauge = Histogram = _Noop  # type: ignore


REQUESTS_TOTAL = Counter(
    "llm_requests_total",
    "Total inference requests received",
    ["backend", "model", "status"],
)

TTFT_HIST = Histogram(
    "llm_time_to_first_token_ms",
    "Time to first token (prefill latency) in ms",
    ["backend"],
    buckets=(10, 25, 50, 100, 200, 350, 500, 750, 1000, 2000, 5000),
)

TPOT_HIST = Histogram(
    "llm_time_per_output_token_ms",
    "Time per output token (decode latency) in ms",
    ["backend"],
    buckets=(2, 5, 10, 20, 30, 50, 80, 120, 200),
)

THROUGHPUT_TOKENS = Counter(
    "llm_generated_tokens_total",
    "Total generated output tokens",
    ["backend"],
)

RUNNING_SEQS = Gauge(
    "llm_running_sequences",
    "Number of sequences currently in the running batch",
)

WAITING_SEQS = Gauge(
    "llm_waiting_sequences",
    "Number of sequences in the waiting queue",
)

BATCH_SIZE = Gauge(
    "llm_batch_size",
    "Number of sequences scheduled in the current iteration",
)

GPU_KV_UTIL = Gauge(
    "llm_kv_cache_utilization",
    "Fraction of KV cache blocks currently in use",
    ["rank"],
)
