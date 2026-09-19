"""Generation routes: /api/generate and /api/chat."""

import json
import time
from typing import Any, Dict

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from .generation import run_generation
from .prompts import build_chat_prompt, build_generate_prompt
from .responses import preflight_error_response, stats_fields
from .schemas import ChatRequest, GenerateRequest
from .state import STATE
from .thinking import collect_split, split_thinking_stream
from .utils import now_iso

router = APIRouter()


@router.post("/api/generate")
def api_generate(req: GenerateRequest):
    model_name = req.model or STATE["served_name"]
    # think=False is the only way to opt out of separation (mirrors real
    # Ollama's default-on behavior for reasoning models); unspecified/True/a
    # level string all keep it, since this server can't suppress the
    # model's own <think> generation, only report it separately or drop it.
    include_thinking = req.think is not False
    try:
        prompt = req.prompt
        if req.system:
            prompt = f"{req.system}\n\n{prompt}"
        prompt = build_generate_prompt(prompt)
    except Exception as exc:  # noqa: BLE001
        return preflight_error_response(model_name, "response", str(exc), req.stream)

    t0 = time.perf_counter()
    stream = run_generation(prompt, req.options)

    if not req.stream:
        thinking_text, content_text = collect_split(stream)
        elapsed_ns = int((time.perf_counter() - t0) * 1e9)
        body = {
            "model": model_name,
            "created_at": now_iso(),
            "response": content_text,
            "done": True,
            "done_reason": "error" if stream.error else "stop",
            "context": [],
        }
        if include_thinking and thinking_text:
            body["thinking"] = thinking_text
        body.update(stats_fields(elapsed_ns, stream.token_count))
        if stream.error:
            body["error"] = stream.error
        return body

    def event_gen():
        for kind, text in split_thinking_stream(stream):
            if kind == "thinking" and not include_thinking:
                continue
            chunk: Dict[str, Any] = {"model": model_name, "created_at": now_iso(), "done": False}
            if kind == "thinking":
                chunk["thinking"] = text
                chunk["response"] = ""
            else:
                chunk["response"] = text
            yield json.dumps(chunk) + "\n"
        elapsed_ns = int((time.perf_counter() - t0) * 1e9)
        final = {
            "model": model_name,
            "created_at": now_iso(),
            "response": "",
            "done": True,
            "done_reason": "error" if stream.error else "stop",
            "context": [],
        }
        final.update(stats_fields(elapsed_ns, stream.token_count))
        if stream.error:
            final["error"] = stream.error
        yield json.dumps(final) + "\n"

    return StreamingResponse(event_gen(), media_type="application/x-ndjson")


@router.post("/api/chat")
def api_chat(req: ChatRequest):
    model_name = req.model or STATE["served_name"]
    include_thinking = req.think is not False
    try:
        messages = [m.model_dump() for m in req.messages]
        prompt = build_chat_prompt(messages)
    except Exception as exc:  # noqa: BLE001
        return preflight_error_response(model_name, "message", str(exc), req.stream)

    t0 = time.perf_counter()
    # apply_chat_template=False: build_chat_prompt() already rendered the
    # full template above, so the pipeline must not apply it a second time.
    stream = run_generation(prompt, req.options, apply_chat_template=False)

    if not req.stream:
        thinking_text, content_text = collect_split(stream)
        elapsed_ns = int((time.perf_counter() - t0) * 1e9)
        message: Dict[str, Any] = {"role": "assistant", "content": content_text}
        if include_thinking and thinking_text:
            message["thinking"] = thinking_text
        body = {
            "model": model_name,
            "created_at": now_iso(),
            "message": message,
            "done": True,
            "done_reason": "error" if stream.error else "stop",
        }
        body.update(stats_fields(elapsed_ns, stream.token_count))
        if stream.error:
            body["error"] = stream.error
        return body

    def event_gen():
        for kind, text in split_thinking_stream(stream):
            if kind == "thinking" and not include_thinking:
                continue
            message: Dict[str, Any] = {"role": "assistant"}
            if kind == "thinking":
                message["thinking"] = text
                message["content"] = ""
            else:
                message["content"] = text
            yield json.dumps(
                {"model": model_name, "created_at": now_iso(), "message": message, "done": False}
            ) + "\n"
        elapsed_ns = int((time.perf_counter() - t0) * 1e9)
        final = {
            "model": model_name,
            "created_at": now_iso(),
            "message": {"role": "assistant", "content": ""},
            "done": True,
            "done_reason": "error" if stream.error else "stop",
        }
        final.update(stats_fields(elapsed_ns, stream.token_count))
        if stream.error:
            final["error"] = stream.error
        yield json.dumps(final) + "\n"

    return StreamingResponse(event_gen(), media_type="application/x-ndjson")
