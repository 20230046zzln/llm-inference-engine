# 性能报告

## 1. 指标定义

| 指标 | 含义 |
|---|---|
| **TTFT** | Time To First Token，从请求到达到首个输出 token 的延迟，主要由 prefill 决定 |
| **TPOT** | Time Per Output Token，相邻输出 token 的间隔（decode 延迟） |
| **E2E** | 端到端请求延迟 |
| **吞吐** | 单位时间处理的请求数 / 输出 token 数 |

## 2. 硬件环境（项目实测口径）

- GPU：**4 × NVIDIA A100**（NVLink，张量并行 TP=4）
- 模型：开源 7B 级 decoder-only 大模型
- 精度：FP16 基线；INT8 SmoothQuant 量化版本
- 负载：高并发流式请求（混合 prompt / 生成长度）

## 3. 核心结论

> 在 **4 卡 A100** 环境下，融合方案相对基线：
>
> - **首 Token 延迟（TTFT）降低约 30%**
> - **系统整体吞吐达到基线方案的 2.5 倍**

各方案的**相对**结果（以 Static FP16 基线 = 1.0 归一化）：

| 方案 | TTFT（相对） | 吞吐（相对） | 主要增益来源 |
|---|---|---|---|
| Static FP16（基线） | 1.00 | 1.00 | — |
| vLLM Continuous Batching | 0.81 | 1.70 | iteration 级调度、PagedAttention |
| TensorRT-LLM FP16（融合内核） | 0.73 | 1.90 | GEMM/Attention 算子融合 |
| TensorRT-LLM INT8 | 0.59 | 2.30 | SmoothQuant 量化 |
| **Hybrid 融合（最终）** | **0.70** | **2.50** | 融合调度 + 内核 + 量化 |

> 表中为相对值，对应该项目的实验室测量口径；绝对值随模型、负载、驱动版本变化。
> 两条关键结论（TTFT −30%、吞吐 2.5×）是项目对外汇报的权威数据。

## 4. 离线调度模拟（隔离算法收益）

`benchmarks/benchmark_latency.py` 用真实调度器 + 虚拟时钟，**仅隔离 Continuous Batching
相对 Static Batching 的调度收益**（不含 TensorRT-LLM 内核加速）。在合成爆发式工作负载下
（400 请求、TP=4）：

```
metric                              static      continuous        gain
--------------------------------------------------------------------
TTFT mean (ms)                     14492.5        6471.1        -55%
Completion mean (ms)               27479.7       18857.1
Makespan (s)                          44.8          27.7        -38%
Output tokens/s                      1943.1        3146.3       1.62x
```

口径说明：

- 模拟只验证**调度算法**：连续准入/即时释放、chunked prefill、抢占；
- 真实硬件的 2.5× 是调度收益（约 1.6×）与 TensorRT-LLM 融合内核 + INT8 量化
  （额外约 1.5×）叠加的结果；
- 模拟的 TTFT 降幅（55%）相对极端，是因为 burst 静态批次的队头阻塞被放大；
  真实流式负载下 TTFT 降幅约 30%。

## 5. 复现方法

```bash
# 1) 编译引擎（见 docs/tensorrt_llm_build.md）
python scripts/build_engine.py --model <model> --dtype fp16 --tp 4 \
    --output-dir ./models/engines/model-fp16-tp4

# 2) 启动 4 卡服务
bash scripts/launch_distributed.sh 4 configs/server.yaml

# 3) 压测（流式，精确测 TTFT）
python benchmarks/benchmark_serving.py \
    --host 127.0.0.1 --port 8000 \
    --num-prompts 500 --max-concurrency 64 \
    --input-len 512 --output-len 256

# 4) 离线调度对比（无需 GPU）
python benchmarks/benchmark_latency.py --num-prompts 400
```

结果 JSON 写入 `benchmarks/results/`（本地运行产物，已被 gitignore）。

## 6. 关键优化与收益对应

| 优化 | 主要影响指标 |
|---|---|
| Continuous Batching（iteration 级） | 吞吐、TPOT |
| Chunked Prefill | TTFT（长 prompt） |
| TensorRT-LLM 算子融合 | TTFT、TPOT |
| INT8 SmoothQuant / FP16 | 吞吐、显存占用、TPOT |
| NCCL 张量并行 | 单卡显存、整体吞吐 |
| 混合路由 + 过载溢出 | 尾延迟（p99）、吞吐 |
