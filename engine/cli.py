"""Unified CLI: ``llm-serve`` (serving) and ``llm-build`` (engine compile).

    llm-serve serve --config configs/server.yaml
    llm-serve build --help
"""
from __future__ import annotations

import typer

from engine.builder.build_cli import app as build_app
from engine.utils.logging import setup_logging

app = typer.Typer(add_completion=False, help="vLLM + TensorRT-LLM inference engine.")
app.add_typer(build_app, name="build")


@app.command()
def serve(
    config: str = typer.Option("configs/server.yaml", "--config"),
    host: str | None = typer.Option(None, "--host"),
    port: int | None = typer.Option(None, "--port"),
    log_level: str = typer.Option("INFO", "--log-level"),
) -> None:
    """Start the OpenAI-compatible serving endpoint."""
    import uvicorn

    from engine.config import load_config
    from engine.server import create_app

    setup_logging(log_level)
    cfg = load_config(config)
    if host:
        cfg.server.host = host
    if port:
        cfg.server.port = port
    uvicorn.run(create_app(cfg), host=cfg.server.host, port=cfg.server.port, log_level=log_level.lower())


if __name__ == "__main__":
    app()
