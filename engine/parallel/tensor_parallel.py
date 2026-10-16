"""Tensor-parallel layers (Megatron-style) over NCCL.

A transformer block is sharded with two complementary layers:

* :class:`ColumnParallelLinear` splits the weight along ``out_features`` (used for
  the fused QKV projection and the first MLP / gate-up projection).
* :class:`RowParallelLinear` splits the weight along ``in_features`` (used for the
  attention output projection and the down projection).

The column-parallel output is **not** gathered; it feeds the row-parallel layer
directly, and a single NCCL ``all_reduce`` after the row-parallel layer produces
the global result. This minimizes communication (one all-reduce per sub-layer).

All torch/cuda imports are lazy so the module imports on CPU/CI.
"""
from __future__ import annotations

import math

from engine.parallel.distributed import get_global_rank, get_world_size
from engine.utils.logging import get_logger

logger = get_logger("tensor_parallel")


def _ensure_divisible(size: int, world_size: int, name: str) -> None:
    if size % world_size != 0:
        raise ValueError(f"{name}={size} must be divisible by world_size={world_size}")


def _all_reduce(tensor, op: str = "sum"):
    import torch.distributed as dist

    if not (dist.is_available() and dist.is_initialized()):
        return tensor
    torch_op = dist.ReduceOp.SUM if op == "sum" else dist.ReduceOp.AVG
    dist.all_reduce(tensor, op=torch_op)
    return tensor


class ColumnParallelLinear:
    """Y = X @ W ; W is sharded by columns (output features).

    Each rank computes ``Y_local = X @ W_local``. With ``gather_output=False`` the
    shard is returned directly (pair with a row-parallel layer).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool = True,
        gather_output: bool = False,
        dtype: str = "float16",
    ) -> None:
        import torch

        world_size = get_world_size()
        _ensure_divisible(out_features, world_size, "out_features")
        self.in_features = in_features
        self.out_features = out_features
        self.out_features_per_rank = out_features // world_size
        self.gather_output = gather_output
        self.world_size = world_size
        self.rank = get_global_rank()

        self.weight = torch.nn.Parameter(
            torch.empty(self.out_features_per_rank, in_features, dtype=getattr(torch, dtype))
        )
        self.bias = (
            torch.nn.Parameter(torch.empty(self.out_features_per_rank, dtype=getattr(torch, dtype)))
            if bias
            else None
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        import torch

        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.in_features
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def shard_from_full_weight(self, full_weight, full_bias=None) -> None:
        """Load a rank-local shard from an unsharded (HF) weight matrix."""
        start = self.rank * self.out_features_per_rank
        end = start + self.out_features_per_rank
        self.weight.data = full_weight[start:end].contiguous()
        if self.bias is not None and full_bias is not None:
            self.bias.data = full_bias[start:end].contiguous()

    def __call__(self, x):
        import torch
        import torch.distributed as dist
        import torch.nn.functional as F

        y_local = F.linear(x, self.weight, self.bias)
        if not self.gather_output or not dist.is_initialized():
            return y_local
        # Gather shards along the last dimension.
        gathered = [torch.empty_like(y_local) for _ in range(self.world_size)]
        dist.all_gather(gathered, y_local)
        return torch.cat(gathered, dim=-1)


class RowParallelLinear:
    """Y = X @ W ; W is sharded by rows (input features).

    Expects the sharded output of a column-parallel layer (input dim already
    partitioned). Each rank computes a partial result and an NCCL all-reduce
    sums them into the global output.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool = True,
        input_is_parallel: bool = True,
        dtype: str = "float16",
    ) -> None:
        import torch

        world_size = get_world_size()
        _ensure_divisible(in_features, world_size, "in_features")
        self.in_features = in_features
        self.out_features = out_features
        self.in_features_per_rank = in_features // world_size
        self.input_is_parallel = input_is_parallel
        self.world_size = world_size
        self.rank = get_global_rank()

        self.weight = torch.nn.Parameter(
            torch.empty(out_features, self.in_features_per_rank, dtype=getattr(torch, dtype))
        )
        # Bias lives on the all-reduced (global) output.
        self.bias = (
            torch.nn.Parameter(torch.empty(out_features, dtype=getattr(torch, dtype)))
            if bias
            else None
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        import torch

        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.in_features
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def shard_from_full_weight(self, full_weight, full_bias=None) -> None:
        start = self.rank * self.in_features_per_rank
        end = start + self.in_features_per_rank
        self.weight.data = full_weight[:, start:end].contiguous()
        if self.bias is not None and full_bias is not None:
            self.bias.data = full_bias.contiguous()

    def __call__(self, x):
        import torch
        import torch.distributed as dist
        import torch.nn.functional as F

        if self.input_is_parallel and dist.is_initialized():
            x_local = x
        else:
            start = self.rank * self.in_features_per_rank
            x_local = x[..., start : start + self.in_features_per_rank]

        y = F.linear(x_local, self.weight, None)
        # Single all-reduce across the tensor-parallel group.
        if dist.is_initialized():
            _all_reduce(y, "sum")
        if self.bias is not None:
            y = y + self.bias
        return y


class TensorParallelGroup:
    """Convenience handle holding the NCCL group and parallel metadata."""

    def __init__(self, tensor_parallel_size: int) -> None:
        self.tp_size = tensor_parallel_size
        self.rank = get_global_rank()

    def shard_attention_heads(self, num_heads: int, num_kv_heads: int) -> tuple[int, int]:
        """Return (local_query_heads, local_kv_heads) for this rank."""
        _ensure_divisible(num_heads, self.tp_size, "num_attention_heads")
        local_q = num_heads // self.tp_size
        local_kv = num_kv_heads
        if num_kv_heads >= self.tp_size:
            _ensure_divisible(num_kv_heads, self.tp_size, "num_kv_heads")
            local_kv = num_kv_heads // self.tp_size
        return local_q, local_kv
