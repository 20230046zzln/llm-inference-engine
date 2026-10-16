#!/usr/bin/env bash
# =============================================================================
# Launch the serving engine across multiple GPUs with NCCL tensor parallelism.
#
#   bash scripts/launch_distributed.sh [GPUS=4] [CONFIG=configs/server.yaml]
#
# TensorRT-LLM expects one MPI rank per GPU; each rank builds its ModelRunner
# and joins the NCCL communicator for tensor-parallel collectives.
# =============================================================================
set -euo pipefail

GPUS="${1:-4}"
CONFIG="${2:-configs/server.yaml}"

# ---- Networking / NCCL knobs -----------------------------------------------
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29500}"
# NCCL tuning for multi-GPU / NVLink (uncomment for IB clusters):
# export NCCL_IB_DISABLE=0
# export NCCL_NET_GDR_LEVEL=2
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export CUDA_DEVICE_MAX_CONNECTIONS=1

echo "Launching ${GPUS}-GPU tensor-parallel serving (config=${CONFIG})"

if command -v mpirun >/dev/null 2>&1; then
  # Preferred path for TensorRT-LLM: one rank per GPU.
  mpirun -np "${GPUS}" \
    --allow-run-as-root \
    -x MASTER_ADDR -x MASTER_PORT \
    -x NCCL_DEBUG -x CUDA_DEVICE_MAX_CONNECTIONS \
    python -m engine.server --config "${CONFIG}"
else
  # Fallback to torchrun (vLLM / pure-Porch path).
  torchrun --nproc_per_node="${GPUS}" \
    --master_addr="${MASTER_ADDR}" --master_port="${MASTER_PORT}" \
    -m engine.server --config "${CONFIG}"
fi
