"""Point the official OpenAI SDK at this engine (works with any OpenAI client).

    # Start the server first:
    LLM_ENGINE_MOCK=1 python -m engine.server --config configs/server.yaml
    python examples/openai_client.py
"""
from __future__ import annotations

from openai import OpenAI

# `base_url` points at the local engine; api_key is unused unless configured.
client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="not-needed")


def chat_non_stream() -> None:
    resp = client.chat.completions.create(
        model="meta-llama/Llama-2-7b-hf",
        messages=[
            {"role": "system", "content": "You are a concise assistant."},
            {"role": "user", "content": "Explain Continuous Batching in one sentence."},
        ],
        max_tokens=64,
        temperature=0.0,
    )
    print("non-stream:", resp.choices[0].message.content)


def chat_stream() -> None:
    stream = client.chat.completions.create(
        model="meta-llama/Llama-2-7b-hf",
        messages=[{"role": "user", "content": "List three benefits of INT8 quantization."}],
        max_tokens=64,
        stream=True,
    )
    print("stream: ", end="")
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            print(delta, end="", flush=True)
    print()


if __name__ == "__main__":
    chat_non_stream()
    chat_stream()
