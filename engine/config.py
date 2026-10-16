"""Typed configuration loading for the serving engine."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    name_or_path: str = "meta-llama/Llama-2-7b-hf"
    tokenizer: str | None = None
    dtype: str = "float16"
    trust_remote_code: bool = False


@dataclass
class TrtLlmConfig:
    engine_dir: str = "/models/engines/llama2-7b-fp16-tp4"
    dtype: str = "float16"
    quantization: str = "fp16"
    max_batch_size: int = 256
    max_input_len: int = 6144
    max_output_len: int = 2048
    kv_cache_free_gpu_mem_fraction: float = 0.9
    enable_kv_cache_reuse: bool = True
    use_inflight_batching: bool = True


@dataclass
class VllmConfig:
    dtype: str = "float16"
    gpu_memory_utilization: float = 0.9
    max_num_seqs: int = 256
    max_num_batched_tokens: int = 8192
    enable_prefix_caching: bool = True
    enforce_eager: bool = False
    tensor_parallel_size: int = 1


@dataclass
class ParallelConfig:
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    backend: str = "nccl"
    distributed_init_method: str = "env://"


@dataclass
class SchedulerConfig:
    policy: str = "continuous"
    max_num_running: int = 256
    max_num_waiting: int = 4096
    prefill_chunk_size: int = 2048
    prefill_decode_mixed: bool = True
    scheduling_policy: str = "fcfs"
    long_prefill_threshold: int = 1024
    target_ttft_ms: int = 500


@dataclass
class RouterConfig:
    strategy: str = "latency_aware"
    trtllm_min_prefill_tokens: int = 512
    spill_to_vllm_on_overload: bool = True


@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    api_keys: list[str] = field(default_factory=list)
    max_model_len: int = 8192
    request_timeout_s: int = 120
    enable_metrics: bool = True


@dataclass
class EngineConfig:
    mode: str = "hybrid"  # vllm | trtllm | hybrid
    model: ModelConfig = field(default_factory=ModelConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    trtllm: TrtLlmConfig = field(default_factory=TrtLlmConfig)
    vllm: VllmConfig = field(default_factory=VllmConfig)
    parallel: ParallelConfig = field(default_factory=ParallelConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    router: RouterConfig = field(default_factory=RouterConfig)

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "EngineConfig":
        def _sub(cls, key):
            return cls(**{k: v for k, v in data.get(key, {}).items() if k in cls.__dataclass_fields__})

        return EngineConfig(
            mode=data.get("mode", "hybrid"),
            model=_sub(ModelConfig, "model"),
            server=_sub(ServerConfig, "server"),
            trtllm=_sub(TrtLlmConfig, "trtllm"),
            vllm=_sub(VllmConfig, "vllm"),
            parallel=_sub(ParallelConfig, "parallel"),
            scheduler=_sub(SchedulerConfig, "scheduler"),
            router=_sub(RouterConfig, "router"),
        )


def load_config(path: str | Path) -> EngineConfig:
    """Load a YAML config file into an :class:`EngineConfig`."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return EngineConfig.from_dict(data)
