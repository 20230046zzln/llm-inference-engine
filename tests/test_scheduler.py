"""Tests for the Continuous Batching scheduler."""
from __future__ import annotations

from engine.config import SchedulerConfig
from engine.scheduler import (
    ContinuousBatchingScheduler,
    Sequence,
    SequenceStatus,
)


def _monotonic_clock():
    state = {"t": 0.0}

    def clock() -> float:
        state["t"] += 1.0
        return state["t"]

    return clock


def _mock_generate(prefill_tasks, decode_seqs):
    """Emit one non-EOS token (id=1) for every active sequence."""
    out: dict[str, list[int]] = {}
    for seq, _chunk in prefill_tasks:
        out[seq.id] = [1]
    for seq in decode_seqs:
        out[seq.id] = [1]
    return out


def _scheduler(**kw):
    cfg = SchedulerConfig(**{k: v for k, v in kw.items() if k in SchedulerConfig.__dataclass_fields__})
    extra = {k: v for k, v in kw.items() if k not in SchedulerConfig.__dataclass_fields__}
    return ContinuousBatchingScheduler(cfg, clock=_monotonic_clock(), **extra)


def test_single_request_completes():
    sched = _scheduler(max_num_running=8, prefill_chunk_size=1024)
    sched.add_request([1, 2, 3], max_new_tokens=4)
    finished = sched.run_until_idle(_mock_generate)

    assert len(finished) == 1
    assert finished[0].status == SequenceStatus.FINISHED_LENGTH
    assert len(finished[0].output_token_ids) == 4


def test_continuous_batching_admits_immediately():
    # No request waits for the slowest; all are admitted while slots exist.
    sched = _scheduler(max_num_running=8, prefill_chunk_size=4096)
    for i in range(6):
        sched.add_request([1] * 8, max_new_tokens=4 if i == 0 else 2)

    finished = sched.run_until_idle(_mock_generate)
    assert len(finished) == 6
    # The short requests still finish on their own schedule (4 and 2 tokens).
    lens = sorted(len(f.output_token_ids) for f in finished)
    assert lens == [2, 2, 2, 2, 2, 4]


def test_chunked_prefill_splits_long_prompt():
    sched = _scheduler(max_num_running=8, prefill_chunk_size=128)
    seq = sched.add_request([1] * 500, max_new_tokens=2)

    # Drive prefill manually until the sequence moves into decoding.
    prefill_steps = 0
    while seq.status == SequenceStatus.WAITING or seq.status == SequenceStatus.RUNNING_PREFILL:
        out = sched.schedule()
        if seq.status == SequenceStatus.RUNNING_PREFILL:
            prefill_steps += 1
        sched.update(out, _mock_generate(out.prefill_tasks, out.decode_seqs))
        if prefill_steps > 10:
            break

    # 500 tokens / 128 chunk => at least 4 prefill iterations.
    assert prefill_steps >= 4
    assert seq.num_computed_tokens == 500

    # Drain the decode steps.
    sched.run_until_idle(_mock_generate)


def test_max_running_admission_limit():
    sched = _scheduler(max_num_running=3, prefill_chunk_size=256)
    for _ in range(10):
        sched.add_request([1] * 4, max_new_tokens=2)

    # First plan should never exceed the running cap.
    out = sched.schedule()
    assert sched.running.__len__() <= 3
    sched.update(out, _mock_generate(out.prefill_tasks, out.decode_seqs))

    finished = sched.run_until_idle(_mock_generate)
    assert len(finished) == 10


def test_preemption_under_small_kv_capacity():
    # Tiny capacity forces RECOMPUTE preemption as more sequences arrive.
    sched = _scheduler(
        max_num_running=8,
        prefill_chunk_size=64,
        kv_cache_token_capacity=120,
    )
    for _ in range(8):
        sched.add_request([1] * 100, max_new_tokens=2)

    # Observe at least one preempted sequence during the run.
    seen_preempt = False
    steps = 0
    while sched.waiting or sched.running:
        out = sched.schedule()
        if out.preempted:
            seen_preempt = True
        gen = _mock_generate(out.prefill_tasks, out.decode_seqs)
        sched.update(out, gen)
        steps += 1
        assert steps < 200

    assert seen_preempt


def test_ttft_recorded():
    sched = _scheduler(max_num_running=4, prefill_chunk_size=256)
    seq = Sequence([1, 2, 3, 4], max_new_tokens=2, arrival_time=0.0)
    sched.add_sequence(seq)
    sched.run_until_idle(_mock_generate)
    assert seq.first_token_time is not None
    assert seq.ttft is not None and seq.ttft > 0
