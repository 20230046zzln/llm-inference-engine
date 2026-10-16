"""OpenAI-compatible FastAPI server.

Endpoints
---------
* ``GET  /health``                  liveness/readiness
* ``GET  /v1/models``               served model metadata
* ``POST /v1/chat/completions``     chat (stream & non-stream)
* ``POST /v1/completions``          completion (stream & non-stream)
* ``GET  /metrics``                 Prometheus metrics

Run
---
    python -m engine.server --config configs/server.yaml
    # CPU/GPU-less smoke test:
    LLM_ENGINE_MOCK=1 python -m engine.server --config configs/server.yaml
"""
from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

from engine.backends import build_backend
from engine.backends.base import BaseBackend, GenerationParams
from engine.config import EngineConfig, load_config
from engine.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatCompletionStreamChoice,
    ChatCompletionStreamResponse,
    ChatMessage,
    CompletionRequest,
    CompletionResponse,
    CompletionResponseChoice,
    DeltaMessage,
    ModelListResponse,
    UsageInfo,
)
from engine.utils.logging import get_logger, setup_logging

logger = get_logger("server")


# --------------------------------------------------------------------------- #
# Application factory
# --------------------------------------------------------------------------- #
def create_app(config: EngineConfig) -> FastAPI:
    if os.environ.get("LLM_ENGINE_MOCK") == "1" or config.mode == "mock":
        config.mode = "mock"

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        backend = build_backend(config)
        await backend.load()
        app.state.backend = backend
        app.state.config = config
        if config.server.enable_metrics:
            _instrument(app)
        logger.info("Serving backend=%s on %s:%d", backend.name, config.server.host, config.server.port)
        try:
            yield
        finally:
            await backend.shutdown()

    app = FastAPI(title="llm-inference-engine", version="0.1.0", lifespan=lifespan)

    def require_api_key(authorization: str | None = Header(default=None)) -> None:
        keys = config.server.api_keys
        if not keys:
            return
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Missing API key")
        if authorization.removeprefix("Bearer ").strip() not in keys:
            raise HTTPException(status_code=401, detail="Invalid API key")

    # ---------------------------------------------------------------- #
    @app.get("/health")
    async def health():
        backend: BaseBackend = app.state.backend
        ok = await backend.health()
        return JSONResponse(
            status_code=200 if ok else 503,
            content={"status": "ok" if ok else "unavailable", "backend": backend.name},
        )

    @app.get("/v1/models")
    async def list_models(_: None = Depends(require_api_key)):
        return ModelListResponse(
            data=[
                {
                    "id": config.model.name_or_path,
                    "object": "model",
                    "owned_by": "llm-inference-engine",
                    "backend": app.state.backend.name,
                }
            ]
        )

    # ---------------------------------------------------------------- #
    @app.post("/v1/chat/completions", dependencies=[Depends(require_api_key)])
    async def chat_completions(req: ChatCompletionRequest):
        prompt = await _render_chat(app.state.backend, req)
        params = _gen_params(
            max_tokens=req.max_tokens or 512,
            temperature=req.temperature,
            top_p=req.top_p,
            repetition_penalty=req.repetition_penalty,
            stop=req.stop,
            priority=req.priority,
        )
        if req.stream:
            return StreamingResponse(
                _chat_stream(app.state.backend, config.model.name_or_path, prompt, params),
                media_type="text/event-stream",
            )

        output = await app.state.backend.generate(f"chat-{id(req)}", prompt, params)
        return ChatCompletionResponse(
            model=config.model.name_or_path,
            choices=[
                ChatCompletionResponseChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=output.text),
                    finish_reason=output.finish_reason if output.finish_reason in (
                        "stop", "length"
                    ) else "stop",
                )
            ],
            usage=UsageInfo(
                prompt_tokens=output.prompt_tokens,
                completion_tokens=output.completion_tokens,
                total_tokens=output.prompt_tokens + output.completion_tokens,
            ),
        )

    @app.post("/v1/completions", dependencies=[Depends(require_api_key)])
    async def completions(req: CompletionRequest):
        prompt = req.prompt if isinstance(req.prompt, str) else req.prompt[0]
        params = _gen_params(
            max_tokens=req.max_tokens or 512,
            temperature=req.temperature,
            top_p=req.top_p,
            stop=req.stop,
            priority=req.priority,
        )
        if req.stream:
            return StreamingResponse(
                _completion_stream(app.state.backend, config.model.name_or_path, prompt, params),
                media_type="text/event-stream",
            )
        output = await app.state.backend.generate(f"cmpl-{id(req)}", prompt, params)
        return CompletionResponse(
            model=config.model.name_or_path,
            choices=[
                CompletionResponseChoice(
                    index=0,
                    text=output.text,
                    finish_reason=output.finish_reason if output.finish_reason in (
                        "stop", "length"
                    ) else "stop",
                )
            ],
            usage=UsageInfo(
                prompt_tokens=output.prompt_tokens,
                completion_tokens=output.completion_tokens,
                total_tokens=output.prompt_tokens + output.completion_tokens,
            ),
        )

    return app


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _gen_params(
    *,
    max_tokens: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float = 1.0,
    stop: str | list[str] | None = None,
    priority: int = 0,
) -> GenerationParams:
    stops = [stop] if isinstance(stop, str) else (stop or [])
    return GenerationParams(
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        stop_strings=stops,
        priority=priority,
    )


async def _render_chat(backend: BaseBackend, req: ChatCompletionRequest) -> str:
    tokenizer = getattr(backend, "_tokenizer", None)
    if tokenizer is not None and getattr(tokenizer, "chat_template", None):
        messages = [m.model_dump(exclude_none=True) for m in req.messages]
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            pass
    # Fallback: plain role/content concatenation.
    parts = []
    for m in req.messages:
        parts.append(f"{m.role}: {m.content or ''}")
    parts.append("assistant:")
    return "\n".join(parts)


def _sse(data: dict | str) -> str:
    payload = data if isinstance(data, str) else json.dumps(data, ensure_ascii=False)
    return f"data: {payload}\n\n"


async def _chat_stream(
    backend: BaseBackend, model: str, prompt: str, params: GenerationParams
) -> AsyncIterator[str]:
    import uuid

    request_id = f"chat-{uuid.uuid4().hex[:12]}"
    first = ChatCompletionStreamResponse(
        model=model, choices=[ChatCompletionStreamChoice(index=0, delta=DeltaMessage(role="assistant"))]
    )
    yield _sse(first.model_dump())
    async for delta in backend.stream(request_id, prompt, params):
        chunk = ChatCompletionStreamResponse(
            model=model,
            choices=[ChatCompletionStreamChoice(index=0, delta=DeltaMessage(content=delta))],
        )
        yield _sse(chunk.model_dump())
    final = ChatCompletionStreamResponse(
        model=model,
        choices=[ChatCompletionStreamChoice(index=0, delta=DeltaMessage(), finish_reason="stop")],
    )
    yield _sse(final.model_dump())
    yield "data: [DONE]\n\n"


async def _completion_stream(
    backend: BaseBackend, model: str, prompt: str, params: GenerationParams
) -> AsyncIterator[str]:
    import uuid

    request_id = f"cmpl-{uuid.uuid4().hex[:12]}"
    async for delta in backend.stream(request_id, prompt, params):
        from engine.protocol import CompletionResponse

        chunk = CompletionResponse(
            model=model,
            choices=[CompletionResponseChoice(index=0, text=delta, finish_reason=None)],
        )
        yield _sse(chunk.model_dump())
    yield "data: [DONE]\n\n"


def _instrument(app: FastAPI) -> None:
    try:
        from prometheus_fastapi_instrumentator import Instrumentator

        Instrumentator().instrument(app).expose(app, include_in_schema=False, endpoint="/metrics")
    except Exception:  # pragma: no cover - optional
        logger.debug("prometheus instrumentation unavailable")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main() -> None:
    import typer
    import uvicorn

    def run(
        config: str = typer.Option("configs/server.yaml", "--config"),
        host: str | None = typer.Option(None, "--host"),
        port: int | None = typer.Option(None, "--port"),
        log_level: str = typer.Option("INFO", "--log-level"),
    ) -> None:
        setup_logging(log_level)
        cfg = load_config(config)
        if host:
            cfg.server.host = host
        if port:
            cfg.server.port = port
        app = create_app(cfg)
        uvicorn.run(app, host=cfg.server.host, port=cfg.server.port, log_level=log_level.lower())

    typer.run(run)


if __name__ == "__main__":
    main()
