#!/usr/bin/env python3
"""Offline latency model: Static Batching vs Continuous Batching.

A discrete-event simulation driven by the real
:class:`engine.scheduler.ContinuousBatchingScheduler` with a virtual clock. It
isolates the *scheduling* benefit of Continuous Batching (it does not model the
TensorRT-LLM kernel speedup, which is measured separately on hardware).

Run:
    python benchmarks/benchmark_latency.py --num-prompts 400 --seed 42
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.config import SchedulerConfig
from engine.scheduler import ContinuousBatchingScheduler, Sequence
from engine.utils.logging import setup_logging

# --------------------------------------------------------------------------- #
# GPU timing model (ms) for a ~7B model on multi-GPU. Used for relative
# comparison only; absolute hardware numbers are in docs/performance.md.
# --------------------------------------------------------------------------- #
PREFILL_TOKENS_PER_MS = 30.0          # batched prefill throughput
DECODE_BASE_MS = 4.0                  # fixed cost of a decode step
DECODE_PER_SEQ_MS = 0.18              # cost per sequence in a decode step


def decode_step_ms(batch: int) -> float:
    return DECODE_BASE_MS + DECODE_PER_SEQ_MS * batch


@dataclass
class Request:
    prompt_len: int
    output_len: int
    arrival: float
    ttft: float = 0.0
    completion: float = 0.0


def gen_requests(n: int, rng: random.Random, arrival_window_s: float) -> list[Request]:
    reqs: list[Request] = []
    for i in range(n):
        # Prompt / output lengths drawn from a log-uniform distribution.
        prompt_len = int(2 ** rng.uniform(7, 11))          # 128..2048
        output_len = int(2 ** rng.uniform(6, 9))           # 64..512
        arrival = rng.uniform(0, arrival_window_s)         # seconds
        reqs.append(Request(prompt_len=prompt_len, output_len=output_len, arrival=arrival))
    reqs.sort(key=lambda r: r.arrival)
    return reqs


# --------------------------------------------------------------------------- #
# Continuous Batching simulation
# --------------------------------------------------------------------------- #
def simulate_continuous(reqs: list[Request], max_running: int) -> list[Request]:
    t = 0.0  # virtual clock, seconds
    sched = ContinuousBatchingScheduler(
        SchedulerConfig(max_num_running=max_running, prefill_chunk_size=2048),
        clock=lambda: t,
    )
    pending = list(reqs)
    by_id = {id(r): r for r in reqs}
    seq_for_req: dict[int, Sequence] = {}

    def admit_due():
        while pending and pending[0].arrival <= t:
            r = pending.pop(0)
            seq = Sequence(
                [1] * r.prompt_len,
                max_new_tokens=r.output_len,
                arrival_time=r.arrival,
            )
            sched.add_sequence(seq)
            seq_for_req[id(r)] = seq
            by_id[id(r)] = r

    while pending or sched.running or sched.waiting:
        admit_due()
        plan = sched.schedule()
        if plan.is_empty:
            # Nothing runnable yet; jump to the next arrival.
            if pending:
                t = max(t, pending[0].arrival)
                continue
            break

        prefill_tokens = sum(c for _, c in plan.prefill_tasks)
        step_ms = prefill_tokens / PREFILL_TOKENS_PER_MS + decode_step_ms(len(plan.decode_seqs))
        t += step_ms / 1000.0

        generated: dict[str, list[int]] = {}
        for seq, _ in plan.prefill_tasks:
            generated[seq.id] = [1]
        for seq in plan.decode_seqs:
            generated[seq.id] = [1]

        finished = sched.update(plan, generated, now=t)
        for seq in finished:
            # Map back to the Request.
            for rid, s in seq_for_req.items():
                if s.id == seq.id:
                    r = by_id[rid]
                    r.ttft = (seq.first_token_time - r.arrival) * 1000.0
                    r.completion = (t - r.arrival) * 1000.0
    return reqs


# --------------------------------------------------------------------------- #
# Static Batching simulation
# --------------------------------------------------------------------------- #
def simulate_static(reqs: list[Request], batch_size: int) -> list[Request]:
    pending = list(reqs)
    t = 0.0

    while pending:
        # Collect everything that has arrived by now.
        ready = [r for r in pending if r.arrival <= t]
        if not ready:
            t = pending[0].arrival
            continue
        batch = ready[:batch_size]
        start = t
        max_prompt = max(r.prompt_len for r in batch)
        max_out = max(r.output_len for r in batch)
        prefill_ms = max_prompt / PREFILL_TOKENS_PER_MS
        batch_ms = prefill_ms + max_out * decode_step_ms(len(batch))
        end = start + batch_ms / 1000.0

        for r in batch:
            # Every request waits for the batch's (longest) prefill and the
            # batch's longest generation before it is released.
            r.ttft = (start - r.arrival) * 1000.0 + prefill_ms
            r.completion = (end - r.arrival) * 1000.0

        done_ids = {id(r) for r in batch}
        pending = [r for r in pending if id(r) not in done_ids]
        t = end
    return reqs


# --------------------------------------------------------------------------- #
# Summaries
# --------------------------------------------------------------------------- #
def summarize(reqs: list[Request]) -> dict[str, float]:
    ttfts = sorted(r.ttft for r in reqs)
    comps = sorted(r.completion for r in reqs)
    makespan = max(r.arrival + r.completion / 1000.0 for r in reqs)
    total_tokens = sum(r.output_len for r in reqs)

    def pct(vals, q):
        return vals[min(len(vals) - 1, int(q * (len(vals) - 1)))]

    return {
        "ttft_mean_ms": sum(ttfts) / len(ttfts),
        "ttft_p90_ms": pct(ttfts, 0.9),
        "completion_mean_ms": sum(comps) / len(comps),
        "completion_p90_ms": pct(comps, 0.9),
        "makespan_s": makespan,
        "output_tokens_per_s": total_tokens / makespan,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--num-prompts", type=int, default=400)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--max-running", type=int, default=256)
    p.add_argument("--arrival-window", type=float, default=2.0,
                   help="Seconds over which requests arrive (small = burst/overload)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--result-dir", default="benchmarks/results")
    args = p.parse_args()

    setup_logging("INFO")
    rng = random.Random(args.seed)

    cont = simulate_continuous(gen_requests(args.num_prompts, rng, args.arrival_window), args.max_running)
    rng = random.Random(args.seed)
    stat = simulate_static(gen_requests(args.num_prompts, rng, args.arrival_window), args.batch_size)

    cs, ss = summarize(cont), summarize(stat)

    speedup = cs["output_tokens_per_s"] / ss["output_tokens_per_s"]
    ttft_drop = 1.0 - cs["ttft_mean_ms"] / ss["ttft_mean_ms"]
    makespan_drop = 1.0 - cs["makespan_s"] / ss["makespan_s"]

    print("=" * 78)
    print(f"{'metric':<26}{'static':>16}{'continuous':>16}{'gain':>16}")
    print("-" * 78)
    rows = [
        ("TTFT mean (ms)", ss["ttft_mean_ms"], cs["ttft_mean_ms"], f"-{ttft_drop*100:.0f}%"),
        ("TTFT p90 (ms)", ss["ttft_p90_ms"], cs["ttft_p90_ms"], ""),
        ("Completion mean (ms)", ss["completion_mean_ms"], cs["completion_mean_ms"], ""),
        ("Makespan (s)", ss["makespan_s"], cs["makespan_s"], f"-{makespan_drop*100:.0f}%"),
        ("Output tokens/s", ss["output_tokens_per_s"], cs["output_tokens_per_s"], f"{speedup:.2f}x"),
    ]
    for name, a, b, g in rows:
        print(f"{name:<26}{a:>16.1f}{b:>16.1f}{g:>16}")
    print("=" * 78)
    print("Note: this isolates the scheduling benefit. Hardware results (4xA100,")
    print("fused TensorRT-LLM kernels + INT8) are reported in docs/performance.md.")

    os.makedirs(args.result_dir, exist_ok=True)
    out = os.path.join(args.result_dir, "static_vs_continuous.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"static": ss, "continuous": cs, "speedup": speedup, "ttft_drop": ttft_drop}, f, indent=2)
    print(f"[ok] saved {out}")


if __name__ == "__main__":
    main()
