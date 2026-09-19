"""Ollama-shaped response helpers shared by /api/generate and /api/chat."""

import json
from typing import Any, Dict

from fastapi.responses import StreamingResponse

from .utils import now_iso


def stats_fields(elapsed_ns: int, token_count: int) -> Dict[str, Any]:
    return {
        "total_duration": elapsed_ns,
        "load_duration": 0,
        "prompt_eval_count": 0,
        "prompt_eval_duration": 0,
        "eval_count": token_count,
        "eval_duration": elapsed_ns,
    }


def preflight_error_response(model_name: str, key: str, error_msg: str, stream: bool):
    """Builds an error response for failures that happen before generation
    ever starts (e.g. NPU prompt-length handling raising unexpectedly), in
    the same shape as a normal done:true/error response, so clients that
    only understand this server's regular error contract still handle it."""
    body: Dict[str, Any] = {
        "model": model_name,
        "created_at": now_iso(),
        key: "" if key == "response" else {"role": "assistant", "content": ""},
        "done": True,
        "done_reason": "error",
        "error": error_msg,
    }
    if key == "response":
        body["context"] = []
    body.update(stats_fields(0, 0))
    if not stream:
        return body
    return StreamingResponse(iter([json.dumps(body) + "\n"]), media_type="application/x-ndjson")
