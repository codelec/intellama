"""Curated, hand-picked starting points - not fetched from anywhere live, just
a small list of pre-converted OpenVINO IR models known to work well for common
use cases. Extend this list as you try more models."""

from typing import Any, Dict, List

MODEL_RECOMMENDATIONS: List[Dict[str, Any]] = [
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
