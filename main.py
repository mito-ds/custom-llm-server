"""
Mock OpenAI-compatible chat completions server for testing Mito's LiteLLM integration.

Mito sends requests here when LITELLM_BASE_URL points at this server.
No real LLM calls — responses are deterministic placeholders.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from typing import Any, AsyncIterator, List, Optional

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse, Response, StreamingResponse
from mito_ai_core.agent.utils import serialize_agent_response
from mito_ai_core.completions.models import AgentResponse, CellUpdate
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


class AddCodeCellRequest(BaseModel):
    code: str
    after_cell_id: str = "new cell"
    message: str = "Adding a new code cell."
    code_summary: str = "Add code cell"


def _new_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex[:12]}"


def _message_text(message: ChatMessage) -> str:
    if isinstance(message.content, str):
        return message.content
    if isinstance(message.content, list):
        parts = []
        for part in message.content:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(str(part.get("text", "")))
        return " ".join(parts).strip()
    return ""


def _last_user_message(messages: List[ChatMessage]) -> str:
    for message in reversed(messages):
        if message.role == "user":
            return _message_text(message)
    return ""


_TASK_SECTION_RE = re.compile(r"<Task>\s*(.*?)\s*</Task>", re.DOTALL)


def _extract_task_text(prompt: str) -> Optional[str]:
    match = _TASK_SECTION_RE.search(prompt)
    if not match:
        return None
    return match.group(1).strip()


def _is_tool_result_prompt(prompt: str) -> bool:
    return "<Tool Result>" in prompt


def _original_user_task(messages: List[ChatMessage]) -> str:
    """Return the user's task from the first agent execution prompt."""
    for message in messages:
        if message.role != "user":
            continue
        text = _message_text(message)
        if _is_tool_result_prompt(text):
            continue
        task = _extract_task_text(text)
        if task:
            return task
    return _last_user_message(messages)


def _is_agent_response_request(response_format: Optional[dict]) -> bool:
    if not response_format or response_format.get("type") != "json_schema":
        return False

    json_schema = response_format.get("json_schema") or {}
    return json_schema.get("name") == "agent_response"


def _build_add_code_cell_agent_response(
    code: str,
    *,
    after_cell_id: str = "new cell",
    message: str = "Adding a new code cell.",
    code_summary: str = "Add code cell",
) -> AgentResponse:
    return AgentResponse(
        type="cell_update",
        message=message,
        cell_update=CellUpdate(
            type="new",
            after_cell_id=after_cell_id,
            code=code,
            code_summary=code_summary,
            cell_type="code",
        ),
        get_cell_output_cell_id=None,
        next_steps=None,
        analysis_assumptions=None,
        streamlit_app_prompt=None,
        question=None,
        answers=None,
        scratchpad_code=None,
        scratchpad_summary=None,
        mcp_tool_call=None,
        skill_name=None,
    )


def _build_finished_task_agent_response(message: str) -> AgentResponse:
    return AgentResponse(
        type="finished_task",
        message=message,
        cell_update=None,
        get_cell_output_cell_id=None,
        next_steps=None,
        analysis_assumptions=None,
        streamlit_app_prompt=None,
        question=None,
        answers=None,
        scratchpad_code=None,
        scratchpad_summary=None,
        mcp_tool_call=None,
        skill_name=None,
    )


def _agent_response_to_json(response: AgentResponse) -> str:
    return json.dumps(serialize_agent_response(response))


def _extract_last_cell_id_from_messages(messages: List[ChatMessage]) -> str:
    decoder = json.JSONDecoder()
    for message in reversed(messages):
        text = _message_text(message)
        for index, char in enumerate(text):
            if char != "[":
                continue
            try:
                parsed, _end = decoder.raw_decode(text, index)
            except json.JSONDecodeError:
                continue
            if not isinstance(parsed, list) or not parsed:
                continue
            if not all(
                isinstance(item, dict) and "id" in item and "cell_type" in item
                for item in parsed
            ):
                continue
            return str(parsed[-1]["id"])
    return "new cell"


def _default_code_from_task(task: str) -> str:
    fenced_match = re.search(r"```(?:python)?\s*\n(.*?)```", task, re.DOTALL)
    if fenced_match:
        return fenced_match.group(1).strip()

    preview = task.strip() or "Hello from custom LLM server"
    return f'print({preview[:200]!r})'


def _should_return_finished_task(messages: List[ChatMessage]) -> bool:
    last_user_message = _last_user_message(messages)
    if _is_tool_result_prompt(last_user_message):
        return True

    for message in messages:
        if message.role != "assistant":
            continue
        text = _message_text(message)
        if '"type": "cell_update"' in text or '"type":"cell_update"' in text:
            return True
    return False


def _agent_response_for_messages(messages: List[ChatMessage]) -> str:
    user_task = _original_user_task(messages)

    if _should_return_finished_task(messages):
        preview = user_task[:200] + ("..." if len(user_task) > 200 else "")
        response = _build_finished_task_agent_response(
            "Finished adding a code cell from your custom local server. "
            f"Task: {preview or '(empty prompt)'}"
        )
        return _agent_response_to_json(response)

    code = _default_code_from_task(user_task)
    after_cell_id = _extract_last_cell_id_from_messages(messages)
    response = _build_add_code_cell_agent_response(
        code,
        after_cell_id=after_cell_id,
        message="Adding a code cell from your custom local server.",
        code_summary="Add code cell",
    )
    return _agent_response_to_json(response)


def _placeholder_content(
    model: str,
    user_message: str,
    response_format: Optional[dict],
    messages: List[ChatMessage],
) -> str:
    if _is_agent_response_request(response_format):
        return _agent_response_for_messages(messages)

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

    user_message = _original_user_task(body.messages)
    content = _placeholder_content(
        body.model,
        user_message,
        body.response_format,
        body.messages,
    )

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


@app.post("/v1/mito/add-code-cell")
async def add_code_cell(
    body: AddCodeCellRequest,
    authorization: Optional[str] = Header(default=None),
) -> dict:
    """Build a valid Mito AgentResponse that inserts a new code cell."""
    _check_auth(authorization)
    response = _build_add_code_cell_agent_response(
        body.code,
        after_cell_id=body.after_cell_id,
        message=body.message,
        code_summary=body.code_summary,
    )
    return {"agent_response": serialize_agent_response(response)}


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
