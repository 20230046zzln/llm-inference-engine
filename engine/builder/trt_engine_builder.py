"""Convert a Hugging Face PyTorch model into a TensorRT-LLM engine.

Pipeline
--------
1. Load the HF model into TensorRT-LLM's model definition (``from_hugging_face``).
2. Optionally apply INT8 SmoothQuant / AWQ / FP8 quantization (with calibration).
3. Configure fused plugins (GEMM / GPT-attention), remove input padding, and set
   in-flight batching + max batch/sequence limits.
4. Build the serialized ``.engine`` plan and write the per-rank config files.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from engine.builder.quantization import QuantizationConfig, build_quant_config
from engine.utils.logging import get_logger

logger = get_logger("builder")

# Registry of TensorRT-LLM model classes keyed by config `model_type`.
MODEL_REGISTRY = {
    "llama": "LLaMAForCausalLM",
    "gpt": "GPTForCausalLM",
    "qwen": "QWenForCausalLM",
    "baichuan": "BaichuanForCausalLM",
    "chatglm": "ChatGLMForCausalLM",
}


@dataclass
class BuildSpec:
    model_type: str = "llama"
    hf_dir: str = ""
    output_dir: str = ""
    dtype: str = "float16"
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    max_batch_size: int = 256
    max_input_len: int = 6144
    max_output_len: int = 2048
    max_num_tokens: int = 16384
    use_gemm_plugin: str = "fp16"
    use_gpt_attention_plugin: str = "fp16"
    remove_input_padding: bool = True
    inflight_batching: bool = True
    operator_fusion: bool = True
    builder_opt_level: int = 3


class TrtEngineBuilder:
    def __init__(self, spec: BuildSpec, quant: QuantizationConfig | None = None) -> None:
        self.spec = spec
        self.quant = quant or QuantizationConfig()

    # ------------------------------------------------------------------ #
    def build(self) -> Path:
        from tensorrt_llm.builder import BuildConfig, build  # type: ignore
        import tensorrt_llm.models as models  # type: ignore

        spec = self.spec
        out_dir = Path(spec.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        cls_name = MODEL_REGISTRY.get(spec.model_type)
        if cls_name is None:
            raise ValueError(f"Unsupported model_type: {spec.model_type}")
        model_cls = getattr(models, cls_name)

        dtype = spec.dtype
        model = model_cls.from_hugging_face(
            spec.hf_dir,
            dtype=dtype,
            tensor_parallel_size=spec.tensor_parallel_size,
            pipeline_parallel_size=spec.pipeline_parallel_size,
        )

        # Apply quantization (INT8 SmoothQuant / AWQ / FP8).
        quant_config = build_quant_config(self.quant)
        if quant_config is not None:
            model.quantize(quant_config)

        # Plugin + fusion configuration.
        plugin_config = self._plugin_config()

        build_config = BuildConfig(
            max_input_len=spec.max_input_len,
            max_seq_len=spec.max_input_len + spec.max_output_len,
            max_batch_size=spec.max_batch_size,
            max_num_tokens=spec.max_num_tokens,
            opt_level=spec.builder_opt_level,
            plugin_config=plugin_config,
        )

        logger.info("Building TensorRT engine (this compiles + profiles kernels)...")
        engine = build(model, build_config)

        # Save per-rank engine + config.
        self._save(model, engine, out_dir)
        logger.info("Engine written to %s", out_dir)
        return out_dir

    # ------------------------------------------------------------------ #
    def _plugin_config(self) -> Any:
        from tensorrt_llm.builder import PluginConfig  # type: ignore

        spec = self.spec
        pc = PluginConfig()
        # Fused GEMM and attention plugins are what give the operator fusion win.
        pc.gemm_plugin = spec.use_gemm_plugin
        pc.gpt_attention_plugin = spec.use_gpt_attention_plugin
        pc.remove_input_padding = spec.remove_input_padding
        pc.inflight_batching = spec.inflight_batching
        # Fused QKV / MLP / bias+residual are enabled by the attention/GEMM plugins.
        pc.enable_xqa = self.spec.operator_fusion
        return pc

    def _save(self, model, engine, out_dir: Path) -> None:
        tp = self.spec.tensor_parallel_size
        pp = self.spec.pipeline_parallel_size
        for rank in range(tp * pp):
            model.save(
                engine,
                out_dir,
                rank=rank,
                save_config=(rank == 0),
            )
