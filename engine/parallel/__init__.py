"""Distributed / tensor-parallel primitives."""
from engine.parallel.distributed import (
    destroy_distributed,
    get_global_rank,
    get_world_size,
    init_distributed,
    is_initialized,
)

__all__ = [
    "init_distributed",
    "destroy_distributed",
    "get_global_rank",
    "get_world_size",
    "is_initialized",
]
