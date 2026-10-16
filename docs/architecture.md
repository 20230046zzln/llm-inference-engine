# 架构设计

## 1. 总览

本引擎把 **vLLM 的 Continuous Batching / PagedAttention 调度能力**与 **TensorRT-LLM 的
算子融合 / 量化内核**结合，对外提供 OpenAI 兼容的推理服务，并通过 NCCL 实现多卡张量并行。

```
                         ┌────────────────────────────────────────────┐
   client (openai SDK)   │                FastAPI 服务层                │
   ───────────────────►  │  /v1/chat/completions  /v1/completions      │
                         │  SSE 流式 · 鉴权 · Prometheus 指标           │
                         └───────────────┬────────────────────────────┘
                                         │
                         ┌───────────────▼────────────────────────────┐
                         │            HybridRouter 混合路由             │
                         │  latency_aware：按 prompt 长度 / 负载 / SLA  │
                         └───────┬───────────────────────────┬─────────┘
                                 │                           │
                ┌────────────────▼─────────┐   ┌─────────────▼──────────────┐
                │  TensorRT-LLM 后端        │   │  vLLM 后端                  │
                │  融合 GEMM/Attention 内核  │   │  PagedAttention 连续批处理   │
                │  INT8 / FP16 量化          │   │  Prefix Caching             │
                └────────────────┬───────────┘   └─────────────┬──────────────┘
                                 │                             │
                         ┌───────▼─────────────────────────────▼───────┐
                         │  Continuous Batching Scheduler（统一调度核心） │
                         │  iteration 级调度 · chunked prefill · 抢占    │
                         └───────────────────────────┬─────────────────┘
                                                     │
                         ┌───────────────────────────▼─────────────────┐
                         │  NCCL Tensor Parallel（Column/Row 并行）       │
                         │  GPU 0 │ GPU 1 │ GPU 2 │ GPU 3                │
                         └──────────────────────────────────────────────┘
```

> 调度核心（`engine/scheduler.py`）有一份行为等价的 C++ 实现（`csrc/`），可编译为
> 原生扩展 `infer_engine._C`，把热路径下沉到 C++/CUDA。

## 2. 请求生命周期

1. 请求进入 FastAPI，校验并转换为 `GenerationParams`。
2. Chat 模板由 tokenizer 渲染（`apply_chat_template`）。
3. `HybridRouter` 依据 prompt 长度、两个后端的在途负载和 SLA 选择后端。
4. 请求进入 `ContinuousBatchingScheduler` 等待队列。
5. 每个 decode step 前调度器重新评估批次：
   - 继续在飞的分块 prefill；
   - 从等待队列准入新请求（受并发数与 token 预算约束）；
   - 已完成 prefill 的序列与 decode 序列混入同一批次；
   - KV 不足时抢占（RECOMPUTE）最早可抢占的序列。
6. 后端执行前向（TRT-LLM 编译内核或 vLLM），产出 token。
7. 完成的序列立即释放槽位，结果以 SSE 流式返回。

## 3. 核心机制

### 3.1 Continuous Batching（连续批处理）

传统 **static batching** 把一批请求绑定，必须等生成长度最长的请求完成，整批才释放：
短请求被长请求拖尾，期间到达的请求无法进入。

Continuous Batching 在**每个 decode step**（而非每批）重新调度：

- 新请求随时准入、完成请求立即退出，无整批栅栏；
- 一个 step 内同时包含 prefill 和 decode（mixed batching）；
- 配合 PagedAttention 的 KV 块，显存利用率显著提高。

### 3.2 Chunked Prefill

长 prompt 被切成 `prefill_chunk_size` 的小块，避免单个长 prefill 长时间占用一次前向、
阻塞整个 decode 批次，从而显著降低长请求的首 Token 延迟（TTFT）。

### 3.3 准入控制与抢占

- 以 KV 驻留 token 数（`num_computed + output_len`）为容量口径；
- 准入时若超出 KV 容量，按优先级抢占正在 decode 的序列（RECOMPUTE 策略），
  释放其 KV 并让其回到等待队列头部，之后重算。

### 3.4 混合路由（vLLM × TensorRT-LLM）

两类引擎的优势互补，`HybridRouter` 按请求形态分发：

| 特征 | 选择 | 原因 |
|---|---|---|
| 长 prompt（≥ 阈值）、计算密集 prefill | TensorRT-LLM | 融合内核 + INT8 量化，prefill 算力效率高 |
| 高并发 decode、短 prompt | vLLM | 连续批处理调度灵活，decode 吞吐高 |
| TRT-LLM 过载 | 溢出到 vLLM | 防止单引擎成为延迟瓶颈 |

### 3.5 TensorRT-LLM 算子融合与量化

- 融合 GEMM 插件、GPT-Attention 插件（融合 QKV、bias+残差、注意力 softmax）；
- 移除 padding（`remove_input_padding`），减少无效计算；
- 量化：FP16 / INT8 SmoothQuant（per-channel 权重 + per-token 激活，α 平滑）/
  INT8-AWQ / FP8，校准通过 NVIDIA ModelOpt 完成。

### 3.6 NCCL 张量并行

Transformer 层采用 Megatron 风格的两种并行线性层：

- `ColumnParallelLinear`：按输出维切分（融合 QKV、MLP 第一个投影）；
- `RowParallelLinear`：按输入维切分（注意力输出投影、MLP 下投影）；
- 列并行的输出**不做 all-gather**，直接喂给行并行，仅在行并行末尾做一次
  NCCL `all_reduce`，把通信量降到每层一次。

多卡通过 MPI（每卡一 rank，TRT-LLM 推荐）或 `torchrun`（vLLM）启动，见
`scripts/launch_distributed.sh`。

## 4. 目录结构

| 路径 | 说明 |
|---|---|
| `engine/server.py` | FastAPI 服务、OpenAI 协议、SSE 流式 |
| `engine/protocol.py` | 请求/响应 Schema（Chat / Completions） |
| `engine/scheduler.py` | Continuous Batching 调度核心（纯 Python，可单测） |
| `engine/router.py` | 混合后端路由策略 |
| `engine/backends/` | base / vLLM / TensorRT-LLM / hybrid 后端 |
| `engine/builder/` | PyTorch → TensorRT-LLM 引擎构建、量化、CLI |
| `engine/parallel/` | NCCL 进程组、Column/Row 张量并行 |
| `engine/utils/` | 日志、Prometheus 指标 |
| `csrc/` | C++/CUDA 调度核心与参考引擎、pybind11 绑定 |
| `benchmarks/` | 服务压测与离线调度对比 |
| `scripts/` | 引擎构建、模型下载、多卡启动 |
| `configs/` | 服务、TRT-LLM、vLLM 配置 |
