"""NCCL process-group bootstrap for multi-GPU tensor parallelism.

Ranks are launched with MPI (``scripts/launch_distributed.sh``) or torchrun and
discover each other via environment variables (``RANK``, ``WORLD_SIZE``,
``LOCAL_RANK``, ``MASTER_ADDR``, ``MASTER_PORT``).
"""
from __future__ import annotations

import os

from engine.utils.logging import get_logger

logger = get_logger("parallel")


def init_distributed(
    backend: str = "nccl",
    init_method: str = "env://",
    *,
    set_cuda_device: bool = True,
) -> dict[str, int]:
    """Initialize the default process group and bind each rank to a CUDA device.

    Returns a dict with rank / world_size / local_rank.
    """
    import torch
    import torch.distributed as dist

    if dist.is_initialized():
        return _env()

    if init_method == "env://":
        _ensure_master_env()

    dist.init_process_group(backend=backend, init_method=init_method)
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    local_rank = int(os.environ.get("LOCAL_RANK", rank % torch.cuda.device_count()))
    if set_cuda_device and torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(local_rank)

    logger.info("Initialized %s group: rank=%d/%d local_rank=%d", backend, rank, world_size, local_rank)
    return _env()


def _env() -> dict[str, int]:
    import torch.distributed as dist

    return {
        "rank": dist.get_rank(),
        "world_size": dist.get_world_size(),
        "local_rank": int(os.environ.get("LOCAL_RANK", "0")),
    }


def _ensure_master_env() -> None:
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")


def is_initialized() -> bool:
    import torch.distributed as dist

    return dist.is_initialized()


def get_world_size() -> int:
    import torch.distributed as dist

    return dist.get_world_size() if dist.is_initialized() else 1


def get_global_rank() -> int:
    import torch.distributed as dist

    return dist.get_rank() if dist.is_initialized() else 0


def barrier() -> None:
    import torch.distributed as dist

    if dist.is_initialized():
        dist.barrier()


def destroy_distributed() -> None:
    import torch.distributed as dist

    if dist.is_initialized():
        dist.destroy_process_group()
