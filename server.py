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
import queue
import threading
import time
from typing import Any, Dict, List, Optional

import openvino_genai as ov_genai
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

app = FastAPI(title="ov-ollama")

STATE: Dict[str, Any] = {
    "pipe": None,
    "tokenizer": None,
    "served_name": "model",
    "device": "CPU",
    "model_dir": "",
    "max_new_tokens": 512,
    "gen_lock": threading.Lock(),  # serializes calls into the shared LLMPipeline
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


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = ""
    messages: List[ChatMessage] = []
    stream: bool = True
    options: Dict[str, Any] = {}


# --------------------------------------------------------------------------- #
# Generation plumbing
# --------------------------------------------------------------------------- #


def build_generation_config(options: Dict[str, Any]):
    """Maps Ollama's `options` fields onto openvino_genai's GenerationConfig."""
    pipe = STATE["pipe"]
    config = pipe.get_generation_config()
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


def run_generation(prompt: str, options: Dict[str, Any]) -> TokenStream:
    config = build_generation_config(options)
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


def collect_full_text(stream: TokenStream) -> str:
    return "".join(stream)


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


@app.post("/api/show")
def show(_: Dict[str, Any] = None):
    return {
        "modelfile": f"# served via openvino_genai from {STATE['model_dir']}",
        "parameters": "",
        "template": "",
        "details": {"format": "openvino", "family": "unknown"},
        "model_info": {"general.architecture": "openvino", "ov.device": STATE["device"]},
    }


@app.post("/api/generate")
def api_generate(req: GenerateRequest):
    model_name = req.model or STATE["served_name"]
    prompt = req.prompt
    if req.system:
        prompt = f"{req.system}\n\n{prompt}"

    t0 = time.perf_counter()
    stream = run_generation(prompt, req.options)

    if not req.stream:
        full_text = collect_full_text(stream)
        elapsed_ns = int((time.perf_counter() - t0) * 1e9)
        body = {
            "model": model_name,
            "created_at": now_iso(),
            "response": full_text,
            "done": True,
            "done_reason": "error" if stream.error else "stop",
            "context": [],
        }
        body.update(stats_fields(elapsed_ns, stream.token_count))
        if stream.error:
            body["error"] = stream.error
        return body

    def event_gen():
        for token in stream:
            yield json.dumps(
                {"model": model_name, "created_at": now_iso(), "response": token, "done": False}
            ) + "\n"
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
    messages = [m.model_dump() for m in req.messages]
    prompt = STATE["tokenizer"].apply_chat_template(messages, add_generation_prompt=True)

    t0 = time.perf_counter()
    stream = run_generation(prompt, req.options)

    if not req.stream:
        full_text = collect_full_text(stream)
        elapsed_ns = int((time.perf_counter() - t0) * 1e9)
        body = {
            "model": model_name,
            "created_at": now_iso(),
            "message": {"role": "assistant", "content": full_text},
            "done": True,
            "done_reason": "error" if stream.error else "stop",
        }
        body.update(stats_fields(elapsed_ns, stream.token_count))
        if stream.error:
            body["error"] = stream.error
        return body

    def event_gen():
        for token in stream:
            yield json.dumps(
                {
                    "model": model_name,
                    "created_at": now_iso(),
                    "message": {"role": "assistant", "content": token},
                    "done": False,
                }
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
    args = parser.parse_args()

    print(f"Loading '{args.model_dir}' on {args.device} ...")
    t0 = time.perf_counter()
    pipe = ov_genai.LLMPipeline(args.model_dir, args.device)
    print(f"Loaded in {time.perf_counter() - t0:.1f}s")

    STATE["pipe"] = pipe
    STATE["tokenizer"] = pipe.get_tokenizer()
    STATE["served_name"] = args.served_name
    STATE["device"] = args.device
    STATE["model_dir"] = args.model_dir
    STATE["max_new_tokens"] = args.max_new_tokens

    print(f"Serving '{args.served_name}' on http://{args.host}:{args.port} (Ollama-compatible API)")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
