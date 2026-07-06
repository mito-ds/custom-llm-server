"""
Mock OpenAI-compatible chat completions server for testing Mito's LiteLLM integration.

Mito sends requests here when LITELLM_BASE_URL points at this server.
No real LLM calls — responses are deterministic placeholders.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any, AsyncIterator, List, Optional

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel

API_KEY = "mito-dev-key"  # match LITELLM_API_KEY in your zprofile
HOST = "127.0.0.1"
PORT = 8080

app = FastAPI(title="Mito Custom LLM Mock Server")


class ChatMessage(BaseModel):
    role: str
    content: Any


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    stream: bool = False
    response_format: Optional[dict] = None


def _new_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex[:12]}"


def _last_user_message(messages: List[ChatMessage]) -> str:
    for message in reversed(messages):
        if message.role == "user":
            if isinstance(message.content, str):
                return message.content
            if isinstance(message.content, list):
                parts = []
                for part in message.content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        parts.append(str(part.get("text", "")))
                return " ".join(parts).strip()
    return ""


def _is_agent_response_request(response_format: Optional[dict]) -> bool:
    if not response_format or response_format.get("type") != "json_schema":
        return False

    json_schema = response_format.get("json_schema") or {}
    return json_schema.get("name") == "agent_response"


def _agent_response_placeholder(user_message: str) -> str:
    preview = user_message[:200] + ("..." if len(user_message) > 200 else "")
    return json.dumps(
        {
            "type": "finished_task",
            "message": (
                "Placeholder agent response from your custom local server. "
                f"Received: {preview or '(empty prompt)'}"
            ),
            "cell_update": None,
            "get_cell_output_cell_id": None,
            "next_steps": None,
            "analysis_assumptions": None,
            "streamlit_app_prompt": None,
            "question": None,
            "answers": None,
            "scratchpad_code": None,
            "scratchpad_summary": None,
            "mcp_tool_call": None,
            "skill_name": None,
        }
    )


def _placeholder_content(model: str, user_message: str, response_format: Optional[dict]) -> str:
    if _is_agent_response_request(response_format):
        return _agent_response_placeholder(user_message)

    if response_format and response_format.get("type") == "json_schema":
        return json.dumps(
            {
                "summary": "Placeholder JSON from custom local server",
                "model": model,
                "echo": user_message[:200],
            }
        )

    preview = user_message[:120] + ("..." if len(user_message) > 120 else "")
    return (
        "This is a placeholder response from your custom local LLM server.\n\n"
        f"Model: {model}\n"
        f"Received: {preview or '(empty prompt)'}"
    )


def _check_auth(authorization: Optional[str]) -> None:
    if authorization != f"Bearer {API_KEY}":
        raise HTTPException(status_code=401, detail="Invalid API key")


def _completion_payload(model: str, content: str) -> dict:
    return {
        "id": _new_id(),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": max(1, len(content.split())),
            "total_tokens": 10 + max(1, len(content.split())),
        },
    }


async def _stream_chunks(model: str, content: str) -> AsyncIterator[str]:
    chunk_id = _new_id()
    words = content.split(" ")

    for index, word in enumerate(words):
        token = word if index == 0 else f" {word}"
        payload = {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": token},
                    "finish_reason": None,
                }
            ],
        }
        yield f"data: {json.dumps(payload)}\n\n"

    final_payload = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {},
                "finish_reason": "stop",
            }
        ],
    }
    yield f"data: {json.dumps(final_payload)}\n\n"
    yield "data: [DONE]\n\n"


async def _handle_chat_completions(
    body: ChatCompletionRequest,
    authorization: Optional[str],
) -> Response:
    _check_auth(authorization)

    user_message = _last_user_message(body.messages)
    content = _placeholder_content(body.model, user_message, body.response_format)

    if body.stream:
        return StreamingResponse(
            _stream_chunks(body.model, content),
            media_type="text/event-stream",
        )

    return JSONResponse(_completion_payload(body.model, content))


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "service": "mito-custom-llm-server"}


@app.get("/v1/models")
async def list_models(authorization: Optional[str] = Header(default=None)) -> dict:
    _check_auth(authorization)
    models = [
        "openai/gpt-4.1",
        "anthropic/claude-haiku-4-5-20251001",
    ]
    return {
        "object": "list",
        "data": [
            {"id": model, "object": "model", "created": int(time.time()), "owned_by": "custom"}
            for model in models
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions_v1(
    body: ChatCompletionRequest,
    authorization: Optional[str] = Header(default=None),
) -> Response:
    return await _handle_chat_completions(body, authorization)


@app.post("/chat/completions")
async def chat_completions(
    body: ChatCompletionRequest,
    authorization: Optional[str] = Header(default=None),
) -> Response:
    return await _handle_chat_completions(body, authorization)


def main() -> None:
    import uvicorn

    uvicorn.run("main:app", host=HOST, port=PORT, reload=True)


if __name__ == "__main__":
    main()
