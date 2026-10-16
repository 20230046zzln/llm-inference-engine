# TensorRT-LLM 引擎构建指南

本指南把 Hugging Face 的 PyTorch 模型编译为 TensorRT-LLM 引擎（`.engine`），
完成算子融合、插件启用与 FP16/INT8 量化，供运行时直接加载。

> 推荐在 NVIDIA 官方 TensorRT 容器内构建（已包含匹配的 CUDA、TensorRT、MPI 与
> TensorRT-LLM），本仓库的 `Dockerfile` 已封装好该环境。

## 1. 准备环境

```bash
# 构建镜像（基于 nvcr.io/nvidia/tensorrt，含 TensorRT-LLM）
docker build -t llm-inference-engine:latest .

# 或在已有容器中手动安装依赖
pip install -r requirements.txt
```

## 2. 下载模型

```bash
# 非受限模型
python scripts/download_model.py --model meta-llama/Llama-2-7b-hf --local-dir ./models/llama2-7b

# 受限模型先设置 HF_TOKEN
export HF_TOKEN=hf_xxx
```

## 3. 构建 FP16 引擎（4 卡张量并行）

```bash
python scripts/build_engine.py \
    --model meta-llama/Llama-2-7b-hf \
    --model-type llama \
    --dtype fp16 \
    --tp 4 \
    --max-batch 256 \
    --max-input-len 6144 \
    --max-output-len 2048 \
    --output-dir ./models/engines/llama2-7b-fp16-tp4
```

或使用配置文件（与 `configs/engine_trtllm.yaml` 等价）：

```bash
llm-serve build build --config configs/engine_trtllm.yaml
```

构建过程包含：

1. `from_hugging_face` 把权重载入 TensorRT-LLM 模型定义；
2. 启用 GEMM / GPT-Attention 融合插件、`remove_input_padding`；
3. 设置 in-flight batching、最大 batch / 序列长度；
4. 编译并 profile 内核，序列化为每个 rank 的 `.engine`。

## 4. 构建 INT8 引擎（SmoothQuant）

INT8 采用 **SmoothQuant**：把激活通道上的离群值按系数 α 平滑到权重，再对
权重做 per-channel、激活做 per-token 的 INT8 量化。

```bash
python scripts/build_engine.py \
    --model meta-llama/Llama-2-7b-hf \
    --model-type llama \
    --quant int8_sq \
    --smooth-alpha 0.5 \
    --tp 4 \
    --output-dir ./models/engines/llama2-7b-int8sq-tp4
```

校准流程由 `engine/builder/quantization.py::calibrate_with_modelopt` 封装，
使用 NVIDIA ModelOpt：

- 校准数据集默认 `cnn_dailymail`，512 条样本、序列长度 2048；
- 先量化再导出，可在 `configs/engine_trtllm.yaml` 中调整 `calibration.*`。

其它量化：

| 模式 | 命令 | 适用 |
|---|---|---|
| FP16 | `--quant fp16` | 基准，兼容性最好 |
| INT8 SmoothQuant | `--quant int8_sq` | 权重+激活，A100 通用 |
| INT8 AWQ | `--quant int8_awq` | 权重量化，显存受限 |
| FP8 | `--quant fp8` | Hopper（H100/H800） |

## 5. 多卡（NCCL）张量并行

引擎按 `--tp` 切分（每个 rank 一个 `.engine`）。服务时每卡一 rank 启动：

```bash
bash scripts/launch_distributed.sh 4 configs/server.yaml
```

脚本优先使用 `mpirun`（TensorRT-LLM 推荐），缺失时回退到 `torchrun`（vLLM）。

## 6. 运行服务

确认 `configs/server.yaml` 中 `trtllm.engine_dir` 指向编译产物，然后：

```bash
# 单卡
python -m engine.server --config configs/server.yaml

# 4 卡
bash scripts/launch_distributed.sh 4
```

## 7. 版本适配说明

TensorRT-LLM 的 Python API 演进较快。本仓库使用的是 **Builder / BuildConfig /
ModelRunnerCpp** 路径（0.10 系列）。若官方 API 变化：

- 构建端：优先以官方 `trtllm convert` / `trtllm build` CLI 为准，
  `engine/builder/trt_engine_builder.py` 是等价的 Python 编排；
- 运行端：`ModelRunnerCpp` 是较稳定的 C++ 运行时；如需 token 级流式可改用
  `Executor` API（`tensorrt_llm.executor`）。
