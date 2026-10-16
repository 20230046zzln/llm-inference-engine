# =============================================================================
# Multi-stage image for the vLLM + TensorRT-LLM fused inference engine.
# TensorRT-LLM ships a CUDA + TensorRT base image; we layer vLLM + our code.
#   docker build -t llm-inference-engine:latest .
# =============================================================================
FROM nvcr.io/nvidia/tensorrt:24.05-py3 AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl build-essential cmake ninja-build \
        openmpi-bin libopenmpi-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/build

# Install vLLM alongside the pre-baked TensorRT-LLM stack.
RUN python -m pip install --upgrade pip && \
    python -m pip install "vllm>=0.5.0" \
        mpi4py fastapi "uvicorn[standard]" pydantic pyyaml typer \
        prometheus-client prometheus-fastapi-instrumentator

# Copy source and build the native C++ scheduler extension.
COPY . /workspace/llm-inference-engine
WORKDIR /workspace/llm-inference-engine
RUN LLM_ENGINE_BUILD_CUDA=1 python -m pip install -e . --no-build-isolation

# -----------------------------------------------------------------------------
FROM nvcr.io/nvidia/tensorrt:24.05-py3 AS runtime

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/models/hf \
    ENGINE_DIR=/models/engines

RUN apt-get update && apt-get install -y --no-install-recommends \
        openmpi-bin libopenmpi-dev curl && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local/lib/python3.10/dist-packages /usr/local/lib/python3.10/dist-packages
COPY . /app
WORKDIR /app

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --retries=5 \
    CMD curl -fsS http://localhost:8000/health || exit 1

# Single-GPU default; override entrypoint for multi-GPU (see scripts/launch_distributed.sh).
ENTRYPOINT ["python", "-m", "engine.server"]
CMD ["--config", "configs/server.yaml"]
