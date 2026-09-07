#!/usr/bin/env python3
"""
Minimal Ollama-compatible server backed by OpenVINO GenAI.

Serves a single OpenVINO IR model (e.g. downloaded from the "OpenVINO" org on
Hugging Face, already converted - no optimum-intel export needed) behind an
HTTP API that mirrors Ollama's wire format closely enough for most Ollama
clients (Open WebUI, ollama-python, curl, etc.) to work against it unmodified:

    GET  /api/version
    GET  /api/tags
    POST /api/show
    POST /api/generate   (stream: true -> newline-delimited JSON, like Ollama)
    POST /api/chat       (stream: true -> newline-delimited JSON, like Ollama)

This is intentionally minimal: one model, one device, one request at a time
(concurrent requests are queued, not parallelized - OpenVINO GenAI's
LLMPipeline is not safe to call concurrently from multiple threads).

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

import argparse
import json
import os
import queue
import threading
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import openvino_genai as ov_genai
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

try:
    import openvino as ov
except ImportError:
    ov = None

app = FastAPI(title="ov-ollama")

# Curated, hand-picked starting points - not fetched from anywhere live, just
# a small list of pre-converted OpenVINO IR models known to work well for
# common use cases. Extend this list as you try more models.
MODEL_RECOMMENDATIONS = [
    {
        "id": "OpenVINO/Qwen3-8B-int4-cw-ov",
        "display_name": "Qwen3 8B (int4, channel-wise)",
        "use_case": "general chat / reasoning",
        "approx_size_gb": 4.5,
        "recommended_devices": ["GPU", "CPU", "NPU"],
        "notes": "Good general-purpose reasoning-capable chat model. Emits <think> blocks. "
                 "Channel-wise int4 quantization also makes it NPU-compatible.",
    },
    {
        "id": "OpenVINO/Qwen2.5-Coder-7B-Instruct-int4-ov",
        "display_name": "Qwen2.5-Coder 7B (int4)",
        "use_case": "coding assistant",
        "approx_size_gb": 4.0,
        "recommended_devices": ["GPU", "CPU"],
        "notes": "Strong code completion/chat model, snappy at 7B; no reasoning-trace overhead.",
    },
    {
        "id": "OpenVINO/Qwen3-VL-8B-Instruct-int8-ov",
        "display_name": "Qwen3-VL 8B (int8)",
        "use_case": "vision + text",
        "approx_size_gb": 9.0,
        "recommended_devices": ["GPU"],
        "notes": "Vision-language model with good OCR/fine-detail retention. Needs a GPU with enough VRAM; "
                 "this server currently only wires up text generation, not image inputs.",
    },
    {
        "id": "OpenVINO/SmolLM3-3B-int4-cw-ov",
        "display_name": "SmolLM3 3B (int4, channel-wise)",
        "use_case": "lightweight chat on constrained hardware",
        "approx_size_gb": 2.0,
        "recommended_devices": ["NPU", "CPU", "GPU"],
        "notes": "Smallest/fastest option here; also the most portable across CPU/GPU/NPU.",
    },
    {
        "id": "OpenVINO/Mistral-7B-Instruct-v0.3-int4-ov",
        "display_name": "Mistral 7B Instruct (int4)",
        "use_case": "general chat",
        "approx_size_gb": 4.0,
        "recommended_devices": ["GPU", "CPU"],
        "notes": "Non-reasoning alternative to Qwen3 if you don't want <think> blocks in output.",
    },
]

STATE: Dict[str, Any] = {
    "pipe": None,
    "tokenizer": None,
    "served_name": "model",
    "device": "CPU",
    "model_dir": "",
    "max_new_tokens": 512,
    "gen_lock": threading.Lock(),  # serializes calls into the shared LLMPipeline
    "is_npu": False,
    "max_prompt_len": 1024,  # openvino_genai's NPU default; overridable via --max-prompt-len
    "min_response_len": 128,  # openvino_genai's NPU default; overridable via --min-response-len
}


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# --------------------------------------------------------------------------- #
# Request schemas (loose on purpose - we only require what we actually use)
# --------------------------------------------------------------------------- #


class GenerateRequest(BaseModel):
    model: str = ""
    prompt: str = ""
    stream: bool = True
    options: Dict[str, Any] = {}
    system: Optional[str] = None
    think: Optional[Union[bool, str]] = None


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = ""
    messages: List[ChatMessage] = []
    stream: bool = True
    options: Dict[str, Any] = {}
    think: Optional[Union[bool, str]] = None


# --------------------------------------------------------------------------- #
# Generation plumbing
# --------------------------------------------------------------------------- #


def build_generation_config(options: Dict[str, Any], apply_chat_template: bool = True):
    """Maps Ollama's `options` fields onto openvino_genai's GenerationConfig.

    `apply_chat_template` defaults to True in openvino_genai's own
    GenerationConfig, meaning pipe.generate(str, ...) re-applies the model's
    chat template to whatever string it's given. That's desired for
    /api/generate (mirrors real Ollama's default templated-prompt behavior),
    but /api/chat already renders the full template itself via
    build_chat_prompt() - without disabling it here, the template would be
    applied twice, inflating token counts (and confusing NPU prompt-length
    truncation, which has no way to predict the second pass).
    """
    pipe = STATE["pipe"]
    config = pipe.get_generation_config()
    config.apply_chat_template = apply_chat_template
    config.max_new_tokens = int(options.get("num_predict") or STATE["max_new_tokens"])
    if "temperature" in options:
        config.temperature = float(options["temperature"])
        config.do_sample = config.temperature > 0
    if "top_p" in options:
        config.top_p = float(options["top_p"])
    if "top_k" in options:
        config.top_k = int(options["top_k"])
    if "repeat_penalty" in options:
        config.repetition_penalty = float(options["repeat_penalty"])
    stop = options.get("stop")
    if stop:
        config.stop_strings = set(stop if isinstance(stop, list) else [stop])
    return config


# --------------------------------------------------------------------------- #
# NPU prompt-length handling
#
# The NPU backend uses static input shapes: it hard-errors if a prompt
# exceeds MAX_PROMPT_LEN (openvino_genai default: 1024 tokens), which is
# easy to hit with clients that inject large system prompts/tool schemas or
# long chat histories (e.g. VS Code Copilot Chat). That error surfaces deep
# inside pipe.generate(), on the streaming worker thread, after the HTTP
# response has already started - which is enough to make picky clients
# report a generic "stream ended unexpectedly" error instead of a useful
# message. We proactively count tokens and truncate before ever calling
# generate(), so oversized requests degrade gracefully (dropped history)
# instead of failing outright. This is a no-op on CPU/GPU, which have no
# static prompt-length limit.
# --------------------------------------------------------------------------- #

NPU_TOKEN_SAFETY_MARGIN = 32  # headroom for template/special-token overhead across re-renders


def count_tokens(text: str) -> int:
    return STATE["tokenizer"].encode(text, add_special_tokens=False).input_ids.shape[1]


def npu_prompt_budget() -> int:
    """Target max input-prompt tokens on NPU: MAX_PROMPT_LEN minus the tokens
    reserved for the response and a small safety margin."""
    return max(STATE["max_prompt_len"] - STATE["min_response_len"] - NPU_TOKEN_SAFETY_MARGIN, 1)


def _truncate_text_to_tokens(text: str, max_tokens: int, keep_end: bool = True) -> str:
    """Shrinks `text` to at most `max_tokens` tokens and re-decodes it to a
    string. Keeps the tail by default - the most relevant part of a
    conversational turn that has to be cut down (e.g. the actual question,
    after a long pasted file). Pass keep_end=False to keep the head instead,
    which suits system prompts better (core instructions usually come
    first, with supplementary content like tool schemas appended after)."""
    if max_tokens <= 0:
        return ""
    ids = STATE["tokenizer"].encode(text, add_special_tokens=False).input_ids
    if ids.shape[1] <= max_tokens:
        return text
    row = ids.data[0]
    kept = (row[-max_tokens:] if keep_end else row[:max_tokens]).tolist()
    return STATE["tokenizer"].decode(kept)


def build_generate_prompt(prompt: str) -> str:
    """For NPU, truncates an already-assembled prompt (system + user text) so
    it fits the pipeline's static MAX_PROMPT_LEN. No-op on CPU/GPU."""
    if not STATE["is_npu"]:
        return prompt
    budget = npu_prompt_budget()
    if count_tokens(prompt) <= budget:
        return prompt
    print(
        f"[TRUNCATED] prompt trimmed to the last {budget} tokens to fit NPU "
        f"MAX_PROMPT_LEN={STATE['max_prompt_len']} (see --max-prompt-len)",
        flush=True,
    )
    return _truncate_text_to_tokens(prompt, budget)


def build_chat_prompt(messages: List[Dict[str, Any]]) -> str:
    """Applies the chat template. On NPU, truncates the conversation so the
    rendered prompt fits MAX_PROMPT_LEN - clients with large system prompts
    (e.g. VS Code Copilot Chat routinely injects large tool schemas) or
    long-running conversations routinely exceed NPU's default 1024-token
    static limit. No-op on CPU/GPU, which have no such limit. Truncation
    proceeds in escalating steps, stopping as soon as the prompt fits:
      1. Drop the oldest non-system turns, one at a time.
      2. Shrink the single remaining (most recent) message's content.
      3. Shrink the system message's content too (it can be the dominant
         source of tokens by itself, e.g. large tool/schema definitions).
    """
    tokenizer = STATE["tokenizer"]

    def render(sys_msgs: List[Dict[str, Any]], rest_msgs: List[Dict[str, Any]]) -> str:
        return tokenizer.apply_chat_template(sys_msgs + rest_msgs, add_generation_prompt=True)

    prompt = render([], messages)
    if not STATE["is_npu"]:
        return prompt

    budget = npu_prompt_budget()
    if count_tokens(prompt) <= budget:
        return prompt

    original_len = len(messages)
    system = [m for m in messages if m.get("role") == "system"]
    rest = [m for m in messages if m.get("role") != "system"]

    # Step 1: drop the oldest non-system turns first, keeping the latest one.
    while len(rest) > 1:
        rest = rest[1:]
        prompt = render(system, rest)
        if count_tokens(prompt) <= budget:
            print(
                f"[TRUNCATED] chat history trimmed from {original_len} to {len(system) + len(rest)} "
                f"messages to fit NPU MAX_PROMPT_LEN={STATE['max_prompt_len']} (see --max-prompt-len)",
                flush=True,
            )
            return prompt

    last = dict(rest[-1]) if rest else None

    # Step 2: down to system + at most one message: shrink that message's content.
    if last is not None:
        overhead = count_tokens(render(system, [dict(last, content="")]))
        last["content"] = _truncate_text_to_tokens(last["content"], max(budget - overhead, 0))
        prompt = render(system, [last])
        if count_tokens(prompt) <= budget:
            print(
                f"[TRUNCATED] chat history trimmed to a single message (content truncated) to fit "
                f"NPU MAX_PROMPT_LEN={STATE['max_prompt_len']} (see --max-prompt-len)",
                flush=True,
            )
            return prompt

    # Step 3: still too big - the system message itself is the culprit (e.g. a
    # large injected tool schema). Shrink it too, keeping its head (core
    # instructions usually come first), and split the remaining budget with
    # the last message's content (keeping its tail).
    if system:
        empty_last = [dict(last, content="")] if last else []
        base_overhead = count_tokens(render([dict(system[0], content="")], empty_last))
        remaining = max(budget - base_overhead, 0)
        sys_budget = remaining // 2 if last else remaining
        msg_budget = remaining - sys_budget
        shrunk_system = dict(system[0])
        shrunk_system["content"] = _truncate_text_to_tokens(shrunk_system["content"], sys_budget, keep_end=False)
        final_messages = [shrunk_system]
        if last is not None:
            last["content"] = _truncate_text_to_tokens(last["content"], msg_budget)
            final_messages.append(last)
        prompt = render([], final_messages)
        print(
            f"[TRUNCATED] system prompt and last message both truncated to fit NPU "
            f"MAX_PROMPT_LEN={STATE['max_prompt_len']} (see --max-prompt-len)",
            flush=True,
        )
    return prompt


class TokenStream:
    """Bridges openvino_genai's blocking generate()+callback API into a plain
    Python iterator, so it can be consumed from a streaming HTTP response."""

    def __init__(self):
        self.q: "queue.Queue[Optional[str]]" = queue.Queue()
        self.error: Optional[str] = None
        self.token_count = 0

    def _callback(self, subword: str):
        self.q.put(subword)
        self.token_count += 1
        return ov_genai.StreamingStatus.RUNNING

    def _run(self, prompt: str, config):
        # Held for the whole generation, not just setup, so concurrent
        # requests queue up instead of hitting the shared pipeline at once.
        with STATE["gen_lock"]:
            try:
                STATE["pipe"].generate(prompt, config, self._callback)
            except Exception as exc:  # noqa: BLE001
                self.error = str(exc)
            finally:
                self.q.put(None)  # sentinel: generation finished

    def start(self, prompt: str, config):
        threading.Thread(target=self._run, args=(prompt, config), daemon=True).start()
        return self

    def __iter__(self):
        while True:
            item = self.q.get()
            if item is None:
                return
            yield item


def run_generation(prompt: str, options: Dict[str, Any], apply_chat_template: bool = True) -> TokenStream:
    config = build_generation_config(options, apply_chat_template=apply_chat_template)
    return TokenStream().start(prompt, config)


# --------------------------------------------------------------------------- #
# Ollama-shaped response helpers
# --------------------------------------------------------------------------- #


def stats_fields(elapsed_ns: int, token_count: int) -> Dict[str, Any]:
    return {
        "total_duration": elapsed_ns,
        "load_duration": 0,
        "prompt_eval_count": 0,
        "prompt_eval_duration": 0,
        "eval_count": token_count,
        "eval_duration": elapsed_ns,
    }


# --------------------------------------------------------------------------- #
# Reasoning (<think>...</think>) separation
#
# Qwen3 (and other reasoning models) prepend a <think>...</think> block to
# every response by default. Real Ollama separates this into its own
# `thinking` field (see https://docs.ollama.com/capabilities/thinking) so
# clients can render it as a collapsible reasoning trace; clients like VS
# Code's Ollama provider expect that separation and render raw <think> tags
# left inline in `content`/`response` as garbled plain text. We split it out
# here so `content`/`response` only ever contains the final answer.
# --------------------------------------------------------------------------- #

THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"


def split_thinking_stream(token_iter):
    """Wraps a raw token iterator (arbitrary-length string chunks) and
    re-yields (kind, text) pairs, kind being "thinking" or "content",
    splitting out a leading <think>...</think> block if present. The tag
    itself may be split across multiple incoming chunks (subword tokens),
    so a small buffer is kept to detect it reliably; leading whitespace
    before the opening tag is tolerated."""
    buf = ""
    state = "before"  # "before" -> "thinking" -> "after"
    for chunk in token_iter:
        buf += chunk
        while True:
            if state == "before":
                stripped = buf.lstrip()
                if stripped.startswith(THINK_OPEN):
                    buf = stripped[len(THINK_OPEN):]
                    state = "thinking"
                    continue
                if THINK_OPEN.startswith(stripped):
                    break  # could still become "<think>" with more data
                state = "after"
                continue
            if state == "thinking":
                idx = buf.find(THINK_CLOSE)
                if idx != -1:
                    if idx:
                        yield ("thinking", buf[:idx])
                    buf = buf[idx + len(THINK_CLOSE):].lstrip("\n")
                    state = "after"
                    continue
                # Hold back a possible partial closing tag at the buffer's end.
                keep = 0
                for k in range(1, min(len(THINK_CLOSE), len(buf)) + 1):
                    if buf.endswith(THINK_CLOSE[:k]):
                        keep = k
                if len(buf) > keep:
                    yield ("thinking", buf[: len(buf) - keep])
                    buf = buf[len(buf) - keep :]
                break
            if state == "after":
                if buf:
                    yield ("content", buf)
                    buf = ""
                break
    if buf:
        yield ("thinking" if state == "thinking" else "content", buf)


def collect_split(stream: TokenStream) -> Tuple[str, str]:
    """Consumes a full TokenStream and returns (thinking_text, content_text)."""
    thinking_parts: List[str] = []
    content_parts: List[str] = []
    for kind, text in split_thinking_stream(stream):
        (thinking_parts if kind == "thinking" else content_parts).append(text)
    return "".join(thinking_parts), "".join(content_parts)


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


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #


@app.get("/")
def root():
    return {"status": "ok", "backend": "openvino-genai", "model": STATE["served_name"], "device": STATE["device"]}


@app.get("/api/version")
def version():
    return {"version": "0.1.0-openvino"}


@app.get("/api/tags")
def tags():
    return {
        "models": [
            {
                "name": STATE["served_name"],
                "model": STATE["served_name"],
                "modified_at": now_iso(),
                "size": 0,
                "digest": "sha256:openvino-genai-served",
                "details": {
                    "format": "openvino",
                    "family": "unknown",
                    "families": None,
                    "parameter_size": "",
                    "quantization_level": "",
                },
            }
        ]
    }


@app.get("/api/ps")
def ps():
    """Mirrors real Ollama's /api/ps, which lists currently loaded models.

    This server always has exactly one model loaded (for its lifetime), so
    it reports that single model as running. `size_vram` is set to the full
    model size when served from GPU/NPU (fully offloaded to the accelerator)
    and 0 for CPU (resident in regular RAM, not VRAM). `expires_at` uses
    Ollama's sentinel for "never unloads" (equivalent to `keep_alive: -1`),
    since this server has no idle-unload behavior.
    """
    size = _dir_size_bytes(STATE["model_dir"]) if STATE["model_dir"] else None
    size = size or 0
    return {
        "models": [
            {
                "name": STATE["served_name"],
                "model": STATE["served_name"],
                "size": size,
                "digest": "sha256:openvino-genai-served",
                "details": {
                    "parent_model": "",
                    "format": "openvino",
                    "family": "unknown",
                    "families": None,
                    "parameter_size": "",
                    "quantization_level": "",
                },
                "expires_at": "0001-01-01T00:00:00Z",
                "size_vram": size if STATE["device"].upper() != "CPU" else 0,
            }
        ]
    }


@app.post("/api/show")
def show(_: Dict[str, Any] = None):
    return {
        "modelfile": f"# served via openvino_genai from {STATE['model_dir']}",
        "parameters": "",
        "template": "",
        "details": {"format": "openvino", "family": "unknown"},
        "model_info": {"general.architecture": "openvino", "ov.device": STATE["device"]},
    }


def _dir_size_bytes(path: str) -> Optional[int]:
    try:
        total = 0
        for root, _dirs, files in os.walk(path):
            for f in files:
                total += os.path.getsize(os.path.join(root, f))
        return total or None
    except Exception:
        return None


@app.get("/api/experimental/model-recommendations")
def model_recommendations():
    """Mirrors real Ollama's /api/experimental/model-recommendations wire
    format (https://ollama.com/api/experimental/model-recommendations). This
    exact path is fetched by the official ollama-vscode extension that backs
    VS Code Copilot Chat's "Ollama" model provider, and by Ollama's desktop
    app, to highlight recommended models.

    Real schema: {"recommendations": [{"model": str, "description": str,
    "context_length"?: int, "max_output_tokens"?: int, "vram_bytes"?: int}]}.
    Every entry MUST include a string "model" field - real clients silently
    drop any entry that doesn't have one, which previously made this
    endpoint appear to return zero recommendations.

    Only the first entry (the model this server currently has loaded) is
    guaranteed to actually work if selected, since it matches this server's
    /api/tags and this server can't /api/pull anything else on demand. The
    rest of MODEL_RECOMMENDATIONS is included for informational purposes.
    """
    served_entry: Dict[str, Any] = {
        "model": STATE["served_name"],
        "description": f"Currently loaded on this server (device: {STATE['device']}), backed by OpenVINO GenAI.",
        "max_output_tokens": STATE["max_new_tokens"],
    }
    size = _dir_size_bytes(STATE["model_dir"]) if STATE["model_dir"] else None
    if size:
        served_entry["vram_bytes"] = size

    catalog_entries = []
    for model in MODEL_RECOMMENDATIONS:
        entry = {
            "model": model["id"],
            "description": f"{model['display_name']} - {model['use_case']}. {model['notes']}",
        }
        if model.get("approx_size_gb"):
            entry["vram_bytes"] = int(model["approx_size_gb"] * (1024 ** 3))
        catalog_entries.append(entry)

    return {"recommendations": [served_entry] + catalog_entries}


@app.post("/api/generate")
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


@app.post("/api/chat")
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


# --------------------------------------------------------------------------- #
# Fallback: anything else Ollama's real API supports but this server doesn't
# (e.g. /api/pull, /api/embed, /api/ps, /api/copy, /api/delete, /api/create).
# Registered last, so it only catches requests that didn't match a route
# above - logs to stdout/server.log instead of failing silently.
# --------------------------------------------------------------------------- #


@app.api_route("/{full_path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"])
async def not_implemented(full_path: str, request: Request):
    path = "/" + full_path
    print(f"[UNIMPLEMENTED] {request.method} {path} was requested but is not supported by this server", flush=True)
    return JSONResponse(
        {"error": f"{path} is not implemented by this minimal OpenVINO-backed Ollama server"},
        status_code=404,
    )


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def main():
    parser = argparse.ArgumentParser(description="Minimal Ollama-compatible server backed by OpenVINO GenAI")
    parser.add_argument("--model-dir", required=True, help="Path to a local OpenVINO IR model directory")
    parser.add_argument("--device", default="CPU", help="OpenVINO device: CPU, GPU, NPU, or e.g. AUTO:GPU,NPU,CPU")
    parser.add_argument("--served-name", default="model", help="Model name clients should request, e.g. qwen3:8b")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=11434)
    parser.add_argument("--max-new-tokens", type=int, default=512, help="Default cap when a client doesn't set num_predict")
    parser.add_argument(
        "--max-prompt-len",
        type=int,
        default=None,
        help="NPU only (requires --device NPU exactly): max input prompt tokens the static pipeline is "
             "compiled for (openvino_genai default: 1024). Raise this if using NPU with clients that send "
             "large system prompts/chat histories (e.g. VS Code Copilot Chat); conversations that still "
             "exceed it are truncated automatically (oldest turns dropped first). Note: very long NPU "
             "prompts can also degrade output quality even when they fit within this limit (see "
             "openvino.genai issue #3255), so raising this isn't a substitute for keeping conversations short.",
    )
    parser.add_argument(
        "--min-response-len",
        type=int,
        default=None,
        help="NPU only (requires --device NPU exactly): min response tokens the static pipeline reserves "
             "(openvino_genai default: 128).",
    )
    args = parser.parse_args()

    is_npu = args.device.strip().upper() == "NPU"
    pipeline_kwargs: Dict[str, Any] = {}
    if is_npu:
        if args.max_prompt_len is not None:
            pipeline_kwargs["MAX_PROMPT_LEN"] = args.max_prompt_len
        if args.min_response_len is not None:
            pipeline_kwargs["MIN_RESPONSE_LEN"] = args.min_response_len
    elif args.max_prompt_len is not None or args.min_response_len is not None:
        print(
            "[WARN] --max-prompt-len/--min-response-len only take effect with --device NPU (exactly); "
            f"ignoring them for --device {args.device}",
            flush=True,
        )

    print(f"Loading '{args.model_dir}' on {args.device} ...")
    t0 = time.perf_counter()
    pipe = ov_genai.LLMPipeline(args.model_dir, args.device, **pipeline_kwargs)
    print(f"Loaded in {time.perf_counter() - t0:.1f}s")

    STATE["pipe"] = pipe
    STATE["tokenizer"] = pipe.get_tokenizer()
    STATE["served_name"] = args.served_name
    STATE["device"] = args.device
    STATE["model_dir"] = args.model_dir
    STATE["max_new_tokens"] = args.max_new_tokens
    STATE["is_npu"] = is_npu
    STATE["max_prompt_len"] = args.max_prompt_len or 1024
    STATE["min_response_len"] = args.min_response_len or 128

    print(f"Serving '{args.served_name}' on http://{args.host}:{args.port} (Ollama-compatible API)")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
