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

_TASK_RE = re.compile(r"<Task>\s*(.*?)\s*</Task>", re.DOTALL)
_NOTEBOOK_RE = re.compile(r"<Notebook>\s*(\[.*?\])\s*</Notebook>", re.DOTALL)
_FENCED_CODE_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)

app = FastAPI(title="Mito Custom LLM Mock Server")


class ChatMessage(BaseModel):
    role: str
    content: Any


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    stream: bool = False
    response_format: Optional[dict] = None


def _message_text(message: ChatMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            str(part.get("text", ""))
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        ).strip()
    return ""


def _user_prompts(messages: List[ChatMessage]) -> List[str]:
    return [_message_text(m) for m in messages if m.role == "user"]


def _agent_response(type_: str, message: str, **fields: Any) -> AgentResponse:
    defaults = {
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
    defaults.update(fields)
    return AgentResponse(type=type_, message=message, **defaults)


def _build_content(
    model: str,
    messages: List[ChatMessage],
    response_format: Optional[dict],
) -> str:
    user_prompts = _user_prompts(messages)

    task = ""
    for prompt in user_prompts:
        if "<Tool Result>" in prompt:
            continue
        if match := _TASK_RE.search(prompt):
            task = match.group(1).strip()
            break
    if not task and user_prompts:
        task = user_prompts[-1]

    json_schema = (response_format or {}).get("json_schema") or {}
    is_agent = (
        response_format
        and response_format.get("type") == "json_schema"
        and json_schema.get("name") == "agent_response"
    )

    if is_agent:
        last_prompt = user_prompts[-1] if user_prompts else ""
        if "<Tool Result>" in last_prompt:
            preview = task[:200] + ("..." if len(task) > 200 else "")
            response = _agent_response(
                "finished_task",
                "Finished adding a code cell from your custom local server. "
                f"Task: {preview or '(empty prompt)'}",
            )
            return json.dumps(serialize_agent_response(response))

        if match := _FENCED_CODE_RE.search(task):
            code = match.group(1).strip()
        else:
            text = task.strip() or "Hello from custom LLM server"
            code = f"print({text[:200]!r})"

        after_cell_id = "new cell"
        for prompt in reversed(user_prompts):
            if match := _NOTEBOOK_RE.search(prompt):
                try:
                    cells = json.loads(match.group(1))
                except json.JSONDecodeError:
                    continue
                if cells:
                    after_cell_id = str(cells[-1]["id"])
                    break

        response = _agent_response(
            "cell_update",
            "Adding a code cell from your custom local server.",
            cell_update=CellUpdate(
                type="new",
                after_cell_id=after_cell_id,
                code=code,
                code_summary="Add code cell",
                cell_type="code",
            ),
        )
        return json.dumps(serialize_agent_response(response))

    if response_format and response_format.get("type") == "json_schema":
        return json.dumps(
            {
                "summary": "Placeholder JSON from custom local server",
                "model": model,
                "echo": task[:200],
            }
        )

    preview = task[:120] + ("..." if len(task) > 120 else "")
    return (
        "This is a placeholder response from your custom local LLM server.\n\n"
        f"Model: {model}\n"
        f"Received: {preview or '(empty prompt)'}"
    )


def _check_auth(authorization: Optional[str]) -> None:
    if authorization != f"Bearer {API_KEY}":
        raise HTTPException(status_code=401, detail="Invalid API key")


async def _stream_response(model: str, content: str) -> AsyncIterator[str]:
    chunk_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    for index, word in enumerate(content.split(" ")):
        delta = word if index == 0 else f" {word}"
        payload = {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [{"index": 0, "delta": {"content": delta}, "finish_reason": None}],
        }
        yield f"data: {json.dumps(payload)}\n\n"

    final = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    yield f"data: {json.dumps(final)}\n\n"
    yield "data: [DONE]\n\n"


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "service": "mito-custom-llm-server"}


@app.post("/v1/chat/completions")
@app.post("/chat/completions")
async def chat_completions(
    body: ChatCompletionRequest,
    authorization: Optional[str] = Header(default=None),
) -> Response:
    _check_auth(authorization)
    content = _build_content(body.model, body.messages, body.response_format)

    if body.stream:
        return StreamingResponse(_stream_response(body.model, content), media_type="text/event-stream")

    completion_tokens = max(1, len(content.split()))
    return JSONResponse(
        {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": body.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": completion_tokens,
                "total_tokens": 10 + completion_tokens,
            },
        }
    )


def main() -> None:
    import uvicorn

    uvicorn.run("main:app", host=HOST, port=PORT, reload=True)


if __name__ == "__main__":
    main()
