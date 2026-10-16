"""Build entry point.

Compiles the optional C++/CUDA scheduler extension (``infer_engine._C``) when a
CUDA toolchain is available; otherwise falls back to a pure-Python install so the
serving stack and tooling remain importable on CPU / CI machines.
"""
from __future__ import annotations

import os
import sys

from setuptools import setup

# A pure-Python install keeps the package usable on CI / CPU boxes. The native
# extension is enabled explicitly with `LLM_ENGINE_BUILD_CUDA=1` inside the
# CUDA / TensorRT-LLM container (see Dockerfile).
BUILD_CUDA = os.environ.get("LLM_ENGINE_BUILD_CUDA", "0") == "1"


def _cuda_extensions():
    if not BUILD_CUDA:
        return []
    try:
        from torch.utils import cpp_extension as torch_cpp
    except ImportError:
        print("[setup] torch not found; skipping native extension build.", file=sys.stderr)
        return []

    define_macros = [("WITH_CUDA", "1")]
    extra_compile_args = {"cxx": ["-O3", "-std=c++17"]}

    return [
        torch_cpp.CppExtension(
            name="infer_engine._C",
            sources=[
                "csrc/src/engine.cpp",
                "csrc/src/batching.cpp",
                "csrc/src/bindings.cpp",
            ],
            include_dirs=["csrc/include"],
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        )
    ]


setup(
    ext_modules=_cuda_extensions(),
    cmdclass={"build_ext": _torch_build_ext()} if BUILD_CUDA else {},
)


def _torch_build_ext():
    from torch.utils.cpp_extension import BuildExtension

    return BuildExtension
