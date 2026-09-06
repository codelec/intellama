# ov-ollama

A minimal, single-file HTTP server that speaks a compatible subset of
[Ollama](https://ollama.com)'s REST API, backed by
[OpenVINO GenAI](https://github.com/openvinotoolkit/openvino.genai) instead of
`llama.cpp`. It lets you run OpenVINO IR models pulled straight from Hugging
Face (e.g. Qwen3, Llama, Phi, Mistral) on Intel **CPU, GPU (Arc/Xe iGPU), or
NPU (Core Ultra "AI Boost")**, and talk to it with any Ollama-compatible
client (`curl`, `ollama-python`, Open WebUI, etc.) without changing that
client's code.

This project is **not affiliated with Ollama**. It re-implements just enough
of Ollama's wire format to be a drop-in backend for common use cases; it is
not a full clone (see [Limitations](#limitations)).

## Features

- Real token-by-token streaming, matching Ollama's newline-delimited JSON
  (NDJSON) response format for `stream: true` requests.
- Runs on CPU, GPU, or NPU by passing a single `--device` flag (including
  OpenVINO `AUTO:GPU,NPU,CPU`-style fallback chains).
- No model conversion required — works directly with pre-converted OpenVINO
  IR models published on Hugging Face.
- Logs a clear `[UNIMPLEMENTED]` message (and returns a proper 404) whenever
  a client calls an Ollama endpoint this server doesn't implement, instead of
  failing silently or crashing.

## Requirements

- Linux (developed and tested on Debian; other distros with OpenVINO Python
  wheel support should work the same way). Not tested on Windows/macOS.
- Python 3.9+
- Optional, for hardware acceleration:
  - **GPU**: Intel compute-runtime / Level-Zero driver installed, and your
    user added to the `render` group (`sudo usermod -aG render $USER`, then
    log out/in).
  - **NPU**: a Core Ultra CPU with the in-kernel `intel_vpu`/`ivpu` driver
    (mainlined since Linux 6.4) plus NPU firmware (on Debian/Ubuntu:
    `sudo apt install firmware-misc-nonfree`). Verify with
    `ls /dev/accel/accel0` — if that path exists, the NPU is visible.
  - CPU-only works everywhere OpenVINO's Python wheel installs, no extra
    drivers needed.

## Installation

```bash
git clone <this-repo-url>
cd <repo-dir>

python3 -m venv --system-site-packages venv
./venv/bin/pip install --upgrade pip
./venv/bin/pip install openvino openvino-genai huggingface_hub fastapi "uvicorn[standard]"
```

Notes:
- `--system-site-packages` lets the venv reuse an OpenVINO install you
  already have system-wide instead of re-downloading it. If you don't have
  one, drop that flag — the `pip install` above will fetch OpenVINO fresh
  into the venv either way.
- On Debian/Ubuntu, running `pip install` **without** a venv fails with
  `error: externally-managed-environment` (PEP 668). Always use a venv as
  shown above (or, not recommended, pass `--break-system-packages`).

## Downloading a model

No conversion step is needed — Hugging Face's
[`OpenVINO` org](https://huggingface.co/OpenVINO) publishes pre-converted,
often pre-quantized (int4/int8) IR builds of popular models:

```bash
./venv/bin/python -c "
from huggingface_hub import snapshot_download
snapshot_download('OpenVINO/Qwen3-8B-int4-cw-ov', local_dir='models/Qwen3-8B-int4-cw-ov')
"
```

Swap the repo id for any other model in that org (different sizes/families
are available). Larger models take proportionally longer to download and
more RAM/VRAM to run.

To convert a different Hugging Face model yourself instead of using a
pre-converted one:

```bash
./venv/bin/pip install "optimum-intel[openvino]"
./venv/bin/optimum-cli export openvino --model <hf-model-id> --weight-format int4 models/<name>
```

## Running the server

```bash
./venv/bin/python server.py \
    --model-dir models/Qwen3-8B-int4-cw-ov \
    --device GPU \
    --served-name qwen3:8b \
    --port 11434
```

Flags:

| Flag | Default | Description |
|---|---|---|
| `--model-dir` | *(required)* | Path to a local OpenVINO IR model directory |
| `--device` | `CPU` | OpenVINO device: `CPU`, `GPU`, `NPU`, or a fallback chain like `AUTO:GPU,NPU,CPU` |
| `--served-name` | `model` | Model name clients should request, e.g. `qwen3:8b` |
| `--host` | `0.0.0.0` | Bind address |
| `--port` | `11434` | Bind port |
| `--max-new-tokens` | `512` | Default generation cap when a client doesn't set `num_predict` |

If real Ollama is already installed and running as a service, it likely owns
port `11434` already — either stop it (`sudo systemctl stop ollama`) or run
this server on a different port (e.g. `--port 11435`) so both can coexist.

NPU-specific notes: the NPU requires static input shapes (handled internally
by OpenVINO GenAI) and only sustains a handful of concurrent requests before
they start queuing at the driver level — fine for single-user use, not for
high-concurrency serving.

## API endpoints implemented

| Method | Path | Notes |
|---|---|---|
| GET | `/` | Basic health/status check |
| GET | `/api/version` | Returns a static version string |
| GET | `/api/tags` | Lists the single served model |
| POST | `/api/show` | Returns basic model metadata |
| POST | `/api/generate` | Prompt completion; supports `stream: true/false` |
| POST | `/api/chat` | Chat completion; applies the model's chat template automatically; supports `stream: true/false` |

Any other Ollama endpoint (`/api/pull`, `/api/push`, `/api/create`,
`/api/copy`, `/api/delete`, `/api/embed`/`/api/embeddings`, `/api/ps`, ...) is
**not implemented**. Calling one returns HTTP 404 with a JSON error body, and
the server prints a line like:

```
[UNIMPLEMENTED] POST /api/pull was requested but is not supported by this server
```

so you can see at a glance which endpoints a client is trying to use that
this server doesn't cover.

## Usage examples

Non-streaming generate:

```bash
curl -H "Content-Type: application/json" http://localhost:11434/api/generate -d '{
  "model": "qwen3:8b",
  "prompt": "Why is the sky blue?",
  "stream": false,
  "options": {"num_predict": 128}
}'
```

Streaming chat (each line is a separate JSON object; the last line has
`"done": true`):

```bash
curl -N -H "Content-Type: application/json" http://localhost:11434/api/chat -d '{
  "model": "qwen3:8b",
  "messages": [{"role": "user", "content": "Give me 3 dinner ideas."}],
  "stream": true
}'
```

`options` fields recognized: `num_predict`, `temperature`, `top_p`, `top_k`,
`repeat_penalty`, `stop`. Unrecognized options are silently ignored rather
than erroring out.

### Using it with Ollama SDKs / Open WebUI

Anything that lets you point at a custom Ollama host works. For example,
with `ollama-python`:

```python
import ollama
client = ollama.Client(host="http://localhost:11434")
response = client.chat(model="qwen3:8b", messages=[{"role": "user", "content": "hi"}])
print(response["message"]["content"])
```

For Open WebUI, set its Ollama API base URL to `http://<host>:<port>`.

## Limitations

This is intentionally a minimal server, not a full Ollama replacement:

- Serves exactly **one model on one device per process**. Run multiple
  instances on different ports to serve multiple models simultaneously.
- **One generation at a time**: concurrent requests are serialized behind an
  internal lock rather than run in parallel, since `LLMPipeline` isn't safe
  to call concurrently from multiple threads.
- **No tool/function calling** — OpenVINO GenAI has no native tool-calling
  support, and this server doesn't add a prompt-based shim for it.
- **No model management** (`/api/pull`, `/api/delete`, `/api/create`, etc.) —
  download and point `--model-dir` at models manually instead.
- **No embeddings API.**
- Reasoning models (e.g. Qwen3) include their raw `<think>...</think>` block
  in the response; this server doesn't strip or separate it.

## Troubleshooting

- **`address already in use` on startup**: something else (often the real
  `ollama` service) already owns that port. Use `--port` to pick another one,
  or `sudo systemctl stop ollama` if you don't need it running.
- **`ModuleNotFoundError: No module named 'openvino_genai'`**: you're not
  running with the venv's Python. Use `./venv/bin/python server.py ...`.
- **`error: externally-managed-environment` during `pip install`**: you tried
  to install outside a venv (Debian/Ubuntu blocks this by default via PEP
  668). Recreate a venv as shown in [Installation](#installation).
- **GPU not detected / falls back to CPU**: check that Level-Zero/compute
  runtime is installed and your user is in the `render` group; run `id` to
  confirm group membership took effect (may require re-login).
- **NPU not detected**: check `ls /dev/accel/accel0` exists, and `dmesg |
  grep -i vpu` for firmware load errors; install `firmware-misc-nonfree` on
  Debian/Ubuntu if the firmware failed to load.
- **Requests hang / never return a response**: check the server's console
  output — the first request after startup includes model warmup time, and
  NPU/GPU compilation on first load can take longer than CPU.

## License

Add a license of your choice (e.g. MIT) before publishing this repository —
none is included by default.
