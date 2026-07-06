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
uv run mito-mock-llm
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