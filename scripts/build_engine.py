#!/usr/bin/env python3
"""Convenience wrapper to compile a TensorRT-LLM engine.

Delegates to the same pipeline used by ``llm-build``::

    python scripts/build_engine.py \
        --model meta-llama/Llama-2-7b-hf --model-type llama \
        --dtype fp16 --tp 4 \
        --output-dir /models/engines/llama2-7b-fp16-tp4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.builder.quantization import QuantizationConfig
from engine.builder.trt_engine_builder import BuildSpec, TrtEngineBuilder
from engine.utils.logging import setup_logging

DTYPE_MAP = {"fp16": "float16", "bf16": "bfloat16", "fp32": "float32"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build a TensorRT-LLM engine.")
    p.add_argument("--model", required=True, help="Hugging Face model id/dir")
    p.add_argument("--model-type", default="llama")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--dtype", default="fp16", choices=list(DTYPE_MAP))
    p.add_argument("--tp", type=int, default=1, help="tensor parallel size")
    p.add_argument("--pp", type=int, default=1)
    p.add_argument("--max-batch", type=int, default=256)
    p.add_argument("--max-input-len", type=int, default=6144)
    p.add_argument("--max-output-len", type=int, default=2048)
    p.add_argument("--quant", default="fp16", help="fp16|int8_sq|int8_awq|fp8")
    p.add_argument("--smooth-alpha", type=float, default=0.5)
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging(args.log_level)
    spec = BuildSpec(
        model_type=args.model_type,
        hf_dir=args.model,
        output_dir=args.output_dir,
        dtype=DTYPE_MAP[args.dtype],
        tensor_parallel_size=args.tp,
        pipeline_parallel_size=args.pp,
        max_batch_size=args.max_batch,
        max_input_len=args.max_input_len,
        max_output_len=args.max_output_len,
    )
    quant = QuantizationConfig(mode=args.quant, smooth_quant_alpha=args.smooth_alpha)
    TrtEngineBuilder(spec, quant).build()
    print(f"[ok] engine written to {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
