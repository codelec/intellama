"""Minimal Ollama-compatible server backed by OpenVINO GenAI.

Serves a single OpenVINO IR model (e.g. downloaded from the "OpenVINO" org on
Hugging Face, already converted - no optimum-intel export needed) behind an
HTTP API that mirrors Ollama's wire format closely enough for most Ollama
clients (Open WebUI, ollama-python, curl, etc.) to work against it unmodified.

Module layout:
    state.py       shared process-wide state (pipeline, tokenizer, settings)
    utils.py       small helpers shared across modules
    schemas.py     request bodies for /api/generate and /api/chat
    catalog.py     curated list of known-good OpenVINO models
    prompts.py     prompt assembly + NPU prompt-length truncation
    generation.py  GenerationConfig mapping and token streaming
    thinking.py    <think>...</think> separation
    responses.py   Ollama-shaped response helpers
    api_meta.py    metadata/discovery routes
    api_inference.py  /api/generate and /api/chat routes
    app.py         FastAPI app assembly (routers + catch-all fallback)
    cli.py         argument parsing, model loading, uvicorn entry point
"""

from .app import app
from .cli import main

__all__ = ["app", "main"]
