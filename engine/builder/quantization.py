"""Quantization configuration: FP16 / INT8 SmoothQuant / INT8-AWQ / FP8.

Wraps TensorRT-LLM's quantization modes in a version-tolerant way and wires up
NVIDIA ModelOpt for calibration-based INT8 (SmoothQuant).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class QuantizationConfig:
    mode: str = "fp16"  # fp16 | int8_sq | int8_awq | fp8
    per_channel: bool = True
    per_token: bool = True
    smooth_quant_alpha: float = 0.5
    awq_group_size: int = 128
    awq_calib_size: int = 512
    fp8_kv_cache: bool = False

    @property
    def is_int8(self) -> bool:
        return self.mode in ("int8_sq", "int8_awq")

    @property
    def needs_calibration(self) -> bool:
        return self.mode in ("int8_sq", "int8_awq", "fp8")


def build_quant_config(cfg: QuantizationConfig):
    """Translate into a TensorRT-LLM ``QuantConfig``.

    Imported lazily; returns None for plain FP16 where no quant config is needed.
    """
    if cfg.mode == "fp16":
        return None

    from tensorrt_llm.quantization import QuantConfig  # type: ignore
    from tensorrt_llm.quantization import QuantMode  # type: ignore

    if cfg.mode == "int8_sq":
        mode = QuantMode.use_smooth_quant(
            per_channel=cfg.per_channel, per_token=cfg.per_token
        )
    elif cfg.mode == "int8_awq":
        mode = QuantMode.from_description(
            use_int4_weights=False, use_int8_weights=True, use_int8_activations=True
        )
    elif cfg.mode == "fp8":
        mode = QuantMode.from_description(quantize_weights=True, quantize_activations=True)
    else:
        raise ValueError(f"Unsupported quantization mode: {cfg.mode}")

    return QuantConfig(mode)


def calibrate_with_modelopt(model, cfg: QuantizationConfig, calib_dataloader):
    """Run SmoothQuant / FP8 calibration through NVIDIA ModelOpt.

    ``model`` is a PyTorch (HF) model; returns a patched model ready for export.
    """
    import modelopt.torch.quantization as mtq  # type: ignore

    if cfg.mode == "int8_sq":
        quant_cfg = mtq.SMOOTH_QUANT_CFG
        # SmoothQuant folds activation scales into weights using alpha.
        quant_cfg["quant_cfg"]["*.input_quantizer"]["enable"] = True
        quant_cfg["quant_cfg"]["*weight_quantizer"]["enable"] = True
    elif cfg.mode == "fp8":
        quant_cfg = mtq.FP8_DEFAULT_CFG
    else:
        raise ValueError(f"ModelOpt calibration not defined for {cfg.mode}")

    def forward_loop(model):
        import torch

        model.eval()
        for batch in calib_dataloader:
            with torch.no_grad():
                model(batch.to(next(model.parameters()).device))

    model = mtq.quantize(model, quant_cfg, forward_loop)
    return model
