"""Metadata/discovery routes: everything except generation."""

from typing import Any, Dict

from fastapi import APIRouter

from .catalog import MODEL_RECOMMENDATIONS
from .state import STATE
from .utils import dir_size_bytes, now_iso

router = APIRouter()


@router.get("/")
def root():
    return {"status": "ok", "backend": "openvino-genai", "model": STATE["served_name"], "device": STATE["device"]}


@router.get("/api/version")
def version():
    return {"version": "0.1.0-openvino"}


@router.get("/api/tags")
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


@router.get("/api/ps")
def ps():
    """Mirrors real Ollama's /api/ps, which lists currently loaded models.

    This server always has exactly one model loaded (for its lifetime), so
    it reports that single model as running. `size_vram` is set to the full
    model size when served from GPU/NPU (fully offloaded to the accelerator)
    and 0 for CPU (resident in regular RAM, not VRAM). `expires_at` uses
    Ollama's sentinel for "never unloads" (equivalent to `keep_alive: -1`),
    since this server has no idle-unload behavior.
    """
    size = dir_size_bytes(STATE["model_dir"]) if STATE["model_dir"] else None
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


@router.post("/api/show")
def show(_: Dict[str, Any] = None):
    return {
        "modelfile": f"# served via openvino_genai from {STATE['model_dir']}",
        "parameters": "",
        "template": "",
        "details": {"format": "openvino", "family": "unknown"},
        "model_info": {"general.architecture": "openvino", "ov.device": STATE["device"]},
    }


@router.get("/api/experimental/model-recommendations")
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
    size = dir_size_bytes(STATE["model_dir"]) if STATE["model_dir"] else None
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
