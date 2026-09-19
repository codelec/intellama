"""Generation routes: /api/generate and /api/chat."""

import asyncio
import json
import time
from typing import Any, Dict

from fastapi import APIRouter, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse

from .generation import TokenStream, run_generation
from .prompts import build_chat_prompt, build_generate_prompt
from .responses import preflight_error_response, stats_fields
from .schemas import ChatRequest, GenerateRequest
from .state import STATE
from .thinking import collect_split, split_thinking_stream
from .utils import now_iso

router = APIRouter()

# Holds references to in-flight disconnect-watcher tasks. asyncio only keeps
# a weak reference to tasks created via create_task(), so without this a
# watcher could be garbage-collected mid-flight; each task removes itself
# once it's done (see _watch_for_disconnect below).
_disconnect_watchers: set = set()


async def _cancel_on_disconnect(request: Request, stream: TokenStream, poll_interval: float = 0.5) -> None:
    """Cancels `stream`'s generation as soon as the client disconnects.

    Without this, an abandoned request (e.g. a client that gave up waiting
    and retried - VS Code's chat UI does this) keeps running to completion
    and holding the shared pipeline lock (STATE["gen_lock"]) for a response
    nobody will ever read, starving every other request queued behind it.
    """
    try:
        while not stream.finished.is_set():
            if await request.is_disconnected():
                stream.cancel()
                return
            await asyncio.sleep(poll_interval)
    except Exception:  # noqa: BLE001
        pass  # best-effort: a watcher failure must never affect the response


def _watch_for_disconnect(request: Request, stream: TokenStream) -> None:
    task = asyncio.create_task(_cancel_on_disconnect(request, stream))
    _disconnect_watchers.add(task)
    task.add_done_callback(_disconnect_watchers.discard)


def _debug_log(label: str, text: str, limit: int = 2000) -> None:
    """Opt-in (--debug) logging of raw prompts/messages and raw model output.

    Useful for diagnosing malformed or unexpected responses - e.g. a
    client's hidden session/title-generation requests, or a chat template
    mismatch causing the model to echo structural markup instead of
    answering - since none of that is otherwise visible outside the chat
    client's own (often opaque) UI.
    """
    if not STATE.get("debug"):
        return
    truncated = text if len(text) <= limit else text[:limit] + f"... [{len(text) - limit} more chars]"
    print(f"[DEBUG] {label}:\n{truncated}\n", flush=True)


@router.post("/api/generate")
async def api_generate(req: GenerateRequest, request: Request):
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
        prompt = await run_in_threadpool(build_generate_prompt, prompt)
    except Exception as exc:  # noqa: BLE001
        return preflight_error_response(model_name, "response", str(exc), req.stream)

    _debug_log("api/generate raw prompt sent to model", prompt)
    t0 = time.perf_counter()
    stream = run_generation(prompt, req.options)
    _watch_for_disconnect(request, stream)

    if not req.stream:
        thinking_text, content_text = await run_in_threadpool(collect_split, stream)
        _debug_log("api/generate raw model output", f"thinking={thinking_text!r}\ncontent={content_text!r}")
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
        debug_parts = []
        for kind, text in split_thinking_stream(stream):
            debug_parts.append((kind, text))
            if kind == "thinking" and not include_thinking:
                continue
            chunk: Dict[str, Any] = {"model": model_name, "created_at": now_iso(), "done": False}
            if kind == "thinking":
                chunk["thinking"] = text
                chunk["response"] = ""
            else:
                chunk["response"] = text
            yield json.dumps(chunk) + "\n"
        thinking_text = "".join(t for k, t in debug_parts if k == "thinking")
        content_text = "".join(t for k, t in debug_parts if k == "content")
        _debug_log("api/generate raw model output (streamed)", f"thinking={thinking_text!r}\ncontent={content_text!r}")
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
async def api_chat(req: ChatRequest, request: Request):
    model_name = req.model or STATE["served_name"]
    include_thinking = req.think is not False
    try:
        messages = [m.model_dump() for m in req.messages]
        prompt = await run_in_threadpool(build_chat_prompt, messages)
    except Exception as exc:  # noqa: BLE001
        return preflight_error_response(model_name, "message", str(exc), req.stream)

    _debug_log("api/chat raw incoming messages", json.dumps(messages, ensure_ascii=False, indent=2))
    _debug_log("api/chat rendered prompt sent to model", prompt)
    t0 = time.perf_counter()
    # apply_chat_template=False: build_chat_prompt() already rendered the
    # full template above, so the pipeline must not apply it a second time.
    stream = run_generation(prompt, req.options, apply_chat_template=False)
    _watch_for_disconnect(request, stream)

    if not req.stream:
        thinking_text, content_text = await run_in_threadpool(collect_split, stream)
        _debug_log("api/chat raw model output", f"thinking={thinking_text!r}\ncontent={content_text!r}")
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
        debug_parts = []
        for kind, text in split_thinking_stream(stream):
            debug_parts.append((kind, text))
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
        thinking_text = "".join(t for k, t in debug_parts if k == "thinking")
        content_text = "".join(t for k, t in debug_parts if k == "content")
        _debug_log("api/chat raw model output (streamed)", f"thinking={thinking_text!r}\ncontent={content_text!r}")
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
