"""CLI for compiling TensorRT-LLM engines.

Examples
--------
FP16, 4-way tensor parallel::

    python -m engine.builder.build_cli build \
        --hf-dir meta-llama/Llama-2-7b-hf --model-type llama \
        --output-dir /models/engines/llama2-7b-fp16-tp4 \
        --dtype fp16 --tp 4

INT8 SmoothQuant (adds calibration)::

    python -m engine.builder.build_cli build ... --quant int8_sq
"""
from __future__ import annotations

from pathlib import Path

import typer
import yaml

from engine.builder.quantization import QuantizationConfig
from engine.builder.trt_engine_builder import BuildSpec, TrtEngineBuilder
from engine.utils.logging import setup_logging

app = typer.Typer(add_completion=False, help="Build TensorRT-LLM engines.")


@app.command()
def build(
    hf_dir: str = typer.Option(..., "--hf-dir", help="Hugging Face model dir/id"),
    model_type: str = typer.Option("llama", "--model-type"),
    output_dir: str = typer.Option(..., "--output-dir"),
    dtype: str = typer.Option("fp16", "--dtype"),
    tp: int = typer.Option(1, "--tp", help="Tensor parallel size"),
    pp: int = typer.Option(1, "--pp", help="Pipeline parallel size"),
    max_batch_size: int = typer.Option(256, "--max-batch"),
    max_input_len: int = typer.Option(6144, "--max-input-len"),
    max_output_len: int = typer.Option(2048, "--max-output-len"),
    quant: str = typer.Option("fp16", "--quant", help="fp16|int8_sq|int8_awq|fp8"),
    smooth_alpha: float = typer.Option(0.5, "--smooth-alpha"),
    config: Path | None = typer.Option(None, "--config", help="Build YAML; overrides flags"),
    log_level: str = typer.Option("INFO", "--log-level"),
) -> None:
    """Compile and save a TensorRT-LLM engine plan."""
    setup_logging(log_level)

    if config is not None:
        with config.open(encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        b = data.get("build", {})
        p = data.get("parallel", {})
        m = data.get("model", {})
        spec = BuildSpec(
            model_type=m.get("model_type", "llama"),
            hf_dir=m.get("name_or_path", hf_dir),
            output_dir=b.get("output_dir", output_dir),
            dtype=b.get("dtype", "float16"),
            tensor_parallel_size=p.get("tensor_parallel_size", tp),
            pipeline_parallel_size=p.get("pipeline_parallel_size", pp),
            max_batch_size=b.get("max_batch_size", 256),
            max_input_len=b.get("max_input_len", 6144),
            max_output_len=b.get("max_output_len", 2048),
            max_num_tokens=b.get("max_num_tokens", 16384),
            use_gemm_plugin=b.get("use_gemm_plugin", "fp16"),
            use_gpt_attention_plugin=b.get("use_gpt_attention_plugin", "fp16"),
            remove_input_padding=b.get("remove_input_padding", True),
            inflight_batching=b.get("use_inflight_batching", True),
            operator_fusion=b.get("operator_fusion", True),
        )
        qcfg = QuantizationConfig(
            mode="int8_sq" if b.get("int8") else b.get("quantization", "fp16"),
            smooth_quant_alpha=b.get("smooth_quant_alpha", 0.5),
        )
    else:
        spec = BuildSpec(
            model_type=model_type,
            hf_dir=hf_dir,
            output_dir=output_dir,
            dtype=_dtype(dtype),
            tensor_parallel_size=tp,
            pipeline_parallel_size=pp,
            max_batch_size=max_batch_size,
            max_input_len=max_input_len,
            max_output_len=max_output_len,
        )
        qcfg = QuantizationConfig(mode=quant, smooth_quant_alpha=smooth_alpha)

    TrtEngineBuilder(spec, qcfg).build()
    typer.secho(f"Done: {spec.output_dir}", fg=typer.colors.GREEN)


def _dtype(name: str) -> str:
    return {"fp16": "float16", "bf16": "bfloat16", "fp32": "float32"}.get(name, name)


if __name__ == "__main__":
    app()
