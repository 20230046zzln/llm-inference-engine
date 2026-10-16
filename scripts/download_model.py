#!/usr/bin/env python3
"""Download/cache a Hugging Face model snapshot.

    python scripts/download_model.py --model meta-llama/Llama-2-7b-hf --local-dir ./models/llama2-7b
Set HF_TOKEN for gated models.
"""
from __future__ import annotations

import argparse
import os

from huggingface_hub import snapshot_download


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--local-dir", default=None)
    p.add_argument(
        "--allow-patterns",
        nargs="*",
        default=["*.json", "*.safetensors", "*.model", "tokenizer*", "*.txt"],
    )
    args = p.parse_args()

    path = snapshot_download(
        repo_id=args.model,
        local_dir=args.local_dir,
        allow_patterns=args.allow_patterns,
        token=os.environ.get("HF_TOKEN"),
    )
    print(f"[ok] cached at {path}")


if __name__ == "__main__":
    main()
