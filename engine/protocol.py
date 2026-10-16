"""OpenAI-compatible request/response schemas.

Mirrors the subset of the OpenAI Chat/Completions API so that existing clients
(``openai`` SDK, LangChain, etc.) can point at this server unchanged.
"""
from __future__ import annotations

import time
import uuid
from typing import Any, Literal

from pydantic import BaseModel, Field


def _request_id() -> str:
    return f"cmpl-{uuid.uuid4().hex[:24]}"


# --------------------------------------------------------------------------- #
# Chat completions
# --------------------------------------------------------------------------- #
class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str | None = None
    name: str | None = None


class ChatCompletionRequest(BaseModel):
    model: str = "default"
    messages: list[ChatMessage]
    temperature: float = 0.7
    top_p: float = 1.0
    n: int = 1
    max_tokens: int | None = 512
    stream: bool = False
    repetition_penalty: float = 1.0
    stop: str | list[str] | None = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    user: str | None = None
    # Extra hint consumed by the hybrid router.
    priority: int = 0


class CompletionRequest(BaseModel):
    model: str = "default"
    prompt: str | list[str]
    temperature: float = 0.7
    top_p: float = 1.0
    max_tokens: int | None = 512
    stream: bool = False
    stop: str | list[str] | None = None
    priority: int = 0


# --------------------------------------------------------------------------- #
# Response models
# --------------------------------------------------------------------------- #
class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Literal["stop", "length"] | None = None


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=_request_id)
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChatCompletionResponseChoice]
    usage: UsageInfo = Field(default_factory=UsageInfo)


class CompletionResponseChoice(BaseModel):
    index: int
    text: str
    finish_reason: Literal["stop", "length"] | None = None


class CompletionResponse(BaseModel):
    id: str = Field(default_factory=_request_id)
    object: str = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[CompletionResponseChoice]
    usage: UsageInfo = Field(default_factory=UsageInfo)


# --------------------------------------------------------------------------- #
# Streaming chunks (SSE)
# --------------------------------------------------------------------------- #
class DeltaMessage(BaseModel):
    role: str | None = None
    content: str | None = None


class ChatCompletionStreamChoice(BaseModel):
    index: int
    delta: DeltaMessage
    finish_reason: str | None = None


class ChatCompletionStreamResponse(BaseModel):
    id: str = Field(default_factory=_request_id)
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChatCompletionStreamChoice]


class ModelListResponse(BaseModel):
    object: str = "list"
    data: list[dict[str, Any]]
