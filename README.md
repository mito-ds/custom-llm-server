# Mito Custom LLM Server

A mock OpenAI-compatible chat completions server for testing Mito's LiteLLM integration. No real LLM calls — responses are deterministic placeholders.

## Requirements

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) (recommended)

## Install

```bash
uv sync
```

## Run

```bash
uv run main.py
```

The server starts at `http://127.0.0.1:8080`.

Verify it's running:

```bash
curl http://127.0.0.1:8080/health
```

## Use with Mito

Point Mito at this server in your environment:

```shellscript
export LITELLM_BASE_URL="http://127.0.0.1:8080/v1"
export LITELLM_API_KEY="mito-dev-key"
export LITELLM_MODELS="litellm/openai/gpt-4.1,litellm/anthropic/claude-haiku-4-5-20251001"
```

The API key must match the value in `main.py` (`mito-dev-key` by default).

## Add a code cell in Mito

This server uses `mito-ai-core` to build valid `AgentResponse` payloads for notebook cell insertion.

### Agent mode (automatic)

When Mito runs in agent mode, the mock server now:

1. Returns a `cell_update` response on the first agent turn (inserts a new code cell).
2. Returns `finished_task` on the next turn after Mito applies and runs the cell.

### Direct endpoint

Build a cell-insertion payload without going through chat completions:

```bash
curl -X POST http://127.0.0.1:8080/v1/mito/add-code-cell \
  -H "Authorization: Bearer mito-dev-key" \
  -H "Content-Type: application/json" \
  -d '{
    "code": "import pandas as pd\nprint(pd.__version__)",
    "after_cell_id": "new cell",
    "message": "Adding pandas version check.",
    "code_summary": "Check pandas version"
  }'
```

`after_cell_id` can be an existing cell id from the notebook, or `"new cell"` to insert at the top.