"""Command-line entry point: parses flags, loads the model, starts uvicorn."""

import argparse
import time
from typing import Any, Dict

import openvino_genai as ov_genai
import uvicorn

from .app import app
from .state import STATE


def build_arg_parser() -> argparse.ArgumentParser:
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
    return parser


def load_model(args: argparse.Namespace) -> None:
    """Loads the pipeline and publishes it (plus derived settings) to STATE."""
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


def main() -> None:
    args = build_arg_parser().parse_args()
    load_model(args)
    print(f"Serving '{args.served_name}' on http://{args.host}:{args.port} (Ollama-compatible API)")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
