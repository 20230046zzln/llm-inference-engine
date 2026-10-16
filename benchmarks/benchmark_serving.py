#!/usr/bin/env python3
"""Serving benchmark: measures TTFT, inter-token latency, e2e latency and
throughput against a running OpenAI-compatible endpoint.

Uses streaming SSE so Time-To-First-Token is measured precisely.

    python benchmarks/benchmark_serving.py \
        --host 127.0.0.1 --port 8000 \
        --num-prompts 500 --max-concurrency 64 \
        --input-len 512 --output-len 256
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx

# A small vocabulary used to build synthetic prompts of a target length.
VOCAB = [
    "the", "model", "system", "token", "attention", "layer", "gpu", "batch",
    "quant", "cache", "tensor", "engine", "kernel", "prompt", "decode", "memory",
]


@dataclass
class SampleStats:
    ttft_ms: float = 0.0
    total_ms: float = 0.0
    output_tokens: int = 0
    itl_ms: list[float] = field(default_factory=list)
    success: bool = True


@dataclass
class Report:
    num_prompts: int
    max_concurrency: int
    input_len: int
    output_len: int
    duration_s: float
    requests_per_s: float
    output_tokens_per_s: float
    ttft_ms: dict[str, float]
    e2e_ms: dict[str, float]
    tpot_ms: dict[str, float]


def make_prompt(input_len: int, rng: random.Random) -> str:
    return " ".join(rng.choice(VOCAB) for _ in range(input_len))


def percentile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    idx = min(len(sorted_vals) - 1, max(0, int(round(q * (len(sorted_vals) - 1)))))
    return sorted_vals[idx]


def dist(vals: list[float]) -> dict[str, float]:
    s = sorted(vals)
    return {
        "mean": sum(s) / len(s) if s else 0.0,
        "p50": percentile(s, 0.50),
        "p90": percentile(s, 0.90),
        "p95": percentile(s, 0.95),
        "p99": percentile(s, 0.99),
        "max": s[-1] if s else 0.0,
    }


async def stream_one(
    client: httpx.AsyncClient,
    base: str,
    prompt: str,
    output_len: int,
    model: str,
) -> SampleStats:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": output_len,
        "stream": True,
        "temperature": 0.7,
    }
    st = SampleStats()
    start = time.perf_counter()
    last_chunk = start
    try:
        async with client.stream("POST", f"{base}/v1/chat/completions", json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    break
                obj = json.loads(data)
                choices = obj.get("choices", [])
                if not choices:
                    continue
                delta = choices[0].get("delta", {})
                content = delta.get("content")
                if content:
                    now = time.perf_counter()
                    if st.output_tokens == 0:
                        st.ttft_ms = (now - start) * 1000.0
                    else:
                        st.itl_ms.append((now - last_chunk) * 1000.0)
                    st.output_tokens += 1
                    last_chunk = now
    except Exception as exc:
        st.success = False
        print(f"[warn] request failed: {exc}")
    st.total_ms = (time.perf_counter() - start) * 1000.0
    return st


async def run(args: argparse.Namespace) -> Report:
    base = f"http://{args.host}:{args.port}"
    rng = random.Random(args.seed)
    prompts = [make_prompt(args.input_len, rng) for _ in range(args.num_prompts)]
    sem = asyncio.Semaphore(args.max_concurrency)
    results: list[SampleStats] = []

    limits = httpx.Limits(max_connections=args.max_concurrency * 2, max_keepalive_connections=args.max_concurrency)
    async with httpx.AsyncClient(timeout=args.timeout, limits=limits) as client:
        async def _worker(prompt: str):
            async with sem:
                return await stream_one(client, base, prompt, args.output_len, args.model)

        t0 = time.perf_counter()
        results = await asyncio.gather(*[_worker(p) for p in prompts])
        duration = time.perf_counter() - t0

    ok = [r for r in results if r.success]
    total_tokens = sum(r.output_tokens for r in ok)
    ttfts = [r.ttft_ms for r in ok if r.ttft_ms]
    e2es = [r.total_ms for r in ok]
    tpots = [it for r in ok for it in r.itl_ms]

    report = Report(
        num_prompts=len(results),
        max_concurrency=args.max_concurrency,
        input_len=args.input_len,
        output_len=args.output_len,
        duration_s=duration,
        requests_per_s=len(ok) / duration,
        output_tokens_per_s=total_tokens / duration,
        ttft_ms=dist(ttfts),
        e2e_ms=dist(e2es),
        tpot_ms=dist(tpots),
    )
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--model", default="default")
    p.add_argument("--num-prompts", type=int, default=200)
    p.add_argument("--max-concurrency", type=int, default=32)
    p.add_argument("--input-len", type=int, default=512)
    p.add_argument("--output-len", type=int, default=256)
    p.add_argument("--timeout", type=int, default=600)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--result-dir", default="benchmarks/results")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    report = asyncio.run(run(args))

    print("=" * 70)
    print(f"  Prompts={report.num_prompts}  Concurrency={report.max_concurrency}")
    print(f"  Duration={report.duration_s:.1f}s  Throughput={report.requests_per_s:.2f} req/s")
    print(f"  Output tokens/s={report.output_tokens_per_s:.1f}")
    print(f"  TTFT  p50={report.ttft_ms['p50']:.1f}ms  p99={report.ttft_ms['p99']:.1f}ms")
    print(f"  TPOT  p50={report.tpot_ms['p50']:.1f}ms  p99={report.tpot_ms['p99']:.1f}ms")
    print(f"  E2E   p50={report.e2e_ms['p50']:.1f}ms  p99={report.e2e_ms['p99']:.1f}ms")
    print("=" * 70)

    os.makedirs(args.result_dir, exist_ok=True)
    tag = f"serve_in{args.input_len}_out{args.output_len}_c{args.max_concurrency}"
    out = os.path.join(args.result_dir, f"{tag}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(asdict(report), f, indent=2)
    print(f"[ok] report saved to {out}")


if __name__ == "__main__":
    main()
