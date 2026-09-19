#!/usr/bin/env python3
"""
Minimal Ollama-compatible server backed by OpenVINO GenAI.

Serves a single OpenVINO IR model (e.g. downloaded from the "OpenVINO" org on
Hugging Face, already converted - no optimum-intel export needed) behind an
HTTP API that mirrors Ollama's wire format closely enough for most Ollama
clients (Open WebUI, ollama-python, curl, etc.) to work against it unmodified:

    GET  /api/version
    GET  /api/tags
    GET  /api/ps
    POST /api/show
    POST /api/generate   (stream: true -> newline-delimited JSON, like Ollama)
    POST /api/chat       (stream: true -> newline-delimited JSON, like Ollama)

This is intentionally minimal: one model, one device, one request at a time
(concurrent requests are queued, not parallelized - OpenVINO GenAI's
LLMPipeline is not safe to call concurrently from multiple threads).

This file is just the entry point; the implementation lives in the
`ov_ollama` package next to it (see ov_ollama/__init__.py for the module
layout).

Setup:
    python3 -m venv --system-site-packages venv
    ./venv/bin/pip install openvino-genai huggingface_hub fastapi "uvicorn[standard]"

    # Download a pre-converted model (no conversion needed):
    ./venv/bin/python -c "from huggingface_hub import snapshot_download; \\
        snapshot_download('OpenVINO/Qwen3-8B-int4-cw-ov', local_dir='models/Qwen3-8B-int4-cw-ov')"

Run:
    ./venv/bin/python server.py --model-dir models/Qwen3-8B-int4-cw-ov \\
        --device GPU --served-name qwen3:8b --port 11434

Try it:
    curl http://localhost:11434/api/generate -d '{"model":"qwen3:8b","prompt":"Why is the sky blue?","stream":false}'
    curl http://localhost:11434/api/chat -d '{"model":"qwen3:8b","messages":[{"role":"user","content":"hi"}]}'
"""

from ov_ollama.app import app  # re-exported so `uvicorn server:app` keeps working
from ov_ollama.cli import main

__all__ = ["app", "main"]


if __name__ == "__main__":
    main()
