"""Generation plumbing: Ollama options -> GenerationConfig, and streaming."""

import queue
import threading
from typing import Any, Dict, Optional

import openvino_genai as ov_genai

from .state import STATE


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
