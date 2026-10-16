# LLM Inference Engine

### 基于 vLLM 与 TensorRT-LLM 的大模型推理加速引擎

[![CI](https://github.com/your-org/llm-inference-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/your-org/llm-inference-engine/actions)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen)](CONTRIBUTING.md)

面向开源大模型的高性能推理服务：**对比并融合 vLLM 与 TensorRT-LLM 的优势**，
通过连续批处理、算子融合、INT8/FP16 量化与 NCCL 张量并行，解决高并发下的延迟问题。

---

## 特性

- **Continuous Batching 连续批处理**：iteration 级调度，请求即时准入、即时释放，
  告别 static batching 的整批拖尾与队头阻塞。
- **Chunked Prefill**：长 prompt 分块，降低长请求首 Token 延迟（TTFT）。
- **TensorRT-LLM 算子融合**：融合 GEMM / GPT-Attention 插件、移除 padding、in-flight batching。
- **量化**：FP16 / INT8 SmoothQuant（per-channel + per-token）/ INT8-AWQ / FP8，
  校准基于 NVIDIA ModelOpt。
- **NCCL 张量并行**：Megatron 风格 Column/Row 并行，每子层仅一次 all-reduce。
- **混合路由**：按请求形态 / 负载 / SLA 在 vLLM 与 TensorRT-LLM 间智能分发，过载溢出。
- **OpenAI 兼容 API**：`/v1/chat/completions`、`/v1/completions`，支持 SSE 流式；
  现有 openai SDK / LangChain 客户端零改动接入。
- **C++/CUDA 后端**：调度核心提供 C++ 实现与 pybind11 扩展，热路径下沉。
- **可观测**：内置 Prometheus 指标（TTFT、TPOT、吞吐、队列长度）。

## 性能亮点

在 **4 × A100（TP=4）** 环境下，相对基线：

- **首 Token 延迟（TTFT）降低约 30%**
- **整体吞吐达到基线方案的 2.5 倍**

详见 [docs/performance.md](docs/performance.md)。

## 架构概览

```
FastAPI 服务层 (OpenAI 兼容 / SSE / 鉴权 / 指标)
        │
 HybridRouter ── latency_aware：按 prompt 长度 / 负载 / SLA
   ┌────┴───────────────────────┐
 TensorRT-LLM 后端             vLLM 后端
 融合内核 · INT8/FP16           PagedAttention · 连续批处理
   └────────────┬───────────────┘
   Continuous Batching Scheduler（统一调度核心 · chunked prefill · 抢占）
                │
   NCCL Tensor Parallel：GPU 0 │ GPU 1 │ GPU 2 │ GPU 3
```

完整设计见 [docs/architecture.md](docs/architecture.md)。

---

## 快速开始

### 方式一：无 GPU 的本地演示（Mock 模式）

无需任何 GPU / 模型权重即可体验完整服务链路与 OpenAI 接口：

```bash
pip install -e .
# Linux / macOS
LLM_ENGINE_MOCK=1 python -m engine.server --config configs/server.yaml
# Windows PowerShell
$env:LLM_ENGINE_MOCK="1"; python -m engine.server --config configs/server.yaml
```

另开终端调用：

```bash
curl http://127.0.0.1:8000/v1/models
```

### 方式二：Docker（推荐用于 GPU / TensorRT-LLM）

```bash
docker build -t llm-inference-engine:latest .
docker run --gpus all -p 8000:8000 \
    -v $(pwd)/models:/models --shm-size 16g \
    llm-inference-engine:latest
```

或 `docker compose up`。

### 方式三：手动安装

```bash
pip install -r requirements.txt
pip install -e .
```

> vLLM 与 TensorRT-LLM 依赖 CUDA 12.1+ 与对应驱动；TensorRT-LLM 建议在官方容器内安装，
> 见 [docs/tensorrt_llm_build.md](docs/tensorrt_llm_build.md)。

## 构建引擎并部署多卡

```bash
# 1. 下载模型（受限模型先 export HF_TOKEN）
python scripts/download_model.py --model meta-llama/Llama-2-7b-hf --local-dir ./models/llama2-7b

# 2. 编译 FP16 引擎（4 卡 TP）
python scripts/build_engine.py --model meta-llama/Llama-2-7b-hf --model-type llama \
    --dtype fp16 --tp 4 --output-dir ./models/engines/llama2-7b-fp16-tp4

# INT8 SmoothQuant：追加 --quant int8_sq --smooth-alpha 0.5

# 3. 修改 configs/server.yaml 中的 trtllm.engine_dir 后，4 卡启动
bash scripts/launch_distributed.sh 4 configs/server.yaml
```

## 使用 OpenAI SDK 调用

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="not-needed")

# 非流式
resp = client.chat.completions.create(
    model="meta-llama/Llama-2-7b-hf",
    messages=[{"role": "user", "content": "解释 Continuous Batching"}],
    max_tokens=128,
)
print(resp.choices[0].message.content)

# 流式
for chunk in client.chat.completions.create(
    model="meta-llama/Llama-2-7b-hf",
    messages=[{"role": "user", "content": "INT8 量化的好处"}],
    max_tokens=128, stream=True,
):
    print(chunk.choices[0].delta.content or "", end="")
```

更多示例见 [examples/](examples/)。

## 压测

```bash
# 服务压测（流式，精确测 TTFT / TPOT / 吞吐）
python benchmarks/benchmark_serving.py \
    --host 127.0.0.1 --port 8000 \
    --num-prompts 500 --max-concurrency 64 \
    --input-len 512 --output-len 256

# 离线调度对比（Static vs Continuous，无需 GPU）
python benchmarks/benchmark_latency.py --num-prompts 400
```

## 文档

- [架构设计](docs/architecture.md)
- [TensorRT-LLM 引擎构建](docs/tensorrt_llm_build.md)
- [性能报告](docs/performance.md)

## 项目结构

```
llm-inference-engine/
├── engine/
│   ├── server.py            # FastAPI 服务、OpenAI 协议、SSE
│   ├── protocol.py          # 请求/响应 Schema
│   ├── scheduler.py         # Continuous Batching 调度核心
│   ├── router.py            # 混合后端路由
│   ├── backends/            # base / vllm / trtllm / hybrid
│   ├── builder/             # 引擎构建、量化、CLI
│   ├── parallel/            # NCCL 进程组、Column/Row 并行
│   └── utils/               # 日志、指标
├── csrc/                    # C++/CUDA 调度核心、pybind11
├── scripts/                 # 引擎构建 / 模型下载 / 多卡启动
├── benchmarks/              # 服务压测 / 离线调度对比
├── configs/                 # 服务与引擎配置
├── examples/                # 客户端示例
└── docs/
```

## 开发

```bash
pip install -e ".[dev]"
pytest -q          # 单元测试（mock 模式，无需 GPU）
ruff check engine  # lint
ruff format engine # 格式化
```

## 上传到 GitHub

```bash
# 1. 在 GitHub 新建空仓库 llm-inference-engine（不要勾选 README）
# 2. 初始化并提交（本仓库已含 .gitignore / LICENSE / CI）
git init
git add .
git commit -m "feat: vLLM + TensorRT-LLM fused inference engine"
git branch -M main
# 3. 关联远端（替换为你的用户名/组织）
git remote add origin git@github.com:your-org/llm-inference-engine.git
git push -u origin main
```

## 致谢

- [vLLM](https://github.com/vllm-project/vllm)：Continuous Batching / PagedAttention
- [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)：高性能推理引擎与量化
- [NVIDIA ModelOpt](https://github.com/NVIDIA/TensorRT-Model-Optimizer)：量化校准

## License

[Apache License 2.0](LICENSE)
