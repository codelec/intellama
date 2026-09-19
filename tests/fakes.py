"""Test doubles standing in for openvino_genai's LLMPipeline/Tokenizer.

These let the test suite exercise the full FastAPI request/response path
(including NPU prompt-truncation and <think> splitting) without needing real
model weights or CPU/GPU/NPU hardware. Tokenization here is a simple,
deterministic whitespace split - one "token" per word - which is all the
code under test actually needs (it only calls .shape, .data, and decode() on
whatever encode() returns).
"""

import re
import types
from typing import Any, Dict, List, Optional

import numpy as np
import openvino_genai as ov_genai


class _FakeInputIds:
    """Mimics the real tokenizer's input_ids tensor closely enough for the
    truncation code in ov_ollama.prompts: a 2D, single-row array supporting
    .shape and row-slicing + .tolist() (via numpy, like the real backend's
    tensor rows)."""

    def __init__(self, tokens: List[str]):
        self._array = np.array(tokens, dtype=object).reshape(1, -1)
        self.shape = self._array.shape

    @property
    def data(self):
        return self._array


class _FakeEncoded:
    def __init__(self, tokens: List[str]):
        self.input_ids = _FakeInputIds(tokens)


class FakeTokenizer:
    """Whitespace tokenizer: encode() splits on spaces, decode() rejoins with
    spaces. Good enough to deterministically exercise truncation logic that
    only cares about token counts and being able to round-trip text."""

    def encode(self, text: str, add_special_tokens: bool = False):
        tokens = text.split(" ") if text else []
        return _FakeEncoded(tokens)

    def decode(self, ids: List[str]) -> str:
        return " ".join(ids)

    def apply_chat_template(self, messages: List[Dict[str, Any]], add_generation_prompt: bool = True) -> str:
        parts = [f"<{m.get('role', '')}>{m.get('content', '')}</{m.get('role', '')}>" for m in messages]
        if add_generation_prompt:
            parts.append("<assistant>")
        return "\n".join(parts)


class FakePipe:
    """Stands in for ov_genai.LLMPipeline.

    By default, streams `response_text` back word-by-word via the callback,
    exactly reconstructible as "".join(chunks) == response_text (each chunk
    keeps its trailing separator). Records the prompt/config it was last
    called with so tests can assert on what actually reached generate() -
    e.g. that NPU truncation was applied before the pipeline ever saw the
    prompt.
    """

    DEFAULT_RESPONSE = (
        "<think>Let me think about this step by step.</think>\nHere is the final answer."
    )

    def __init__(self, response_text: Optional[str] = None, raise_error: Optional[str] = None):
        self.response_text = response_text if response_text is not None else self.DEFAULT_RESPONSE
        self.raise_error = raise_error
        self.last_prompt: Optional[str] = None
        self.last_config: Any = None
        self.calls: List[Dict[str, Any]] = []

    def get_generation_config(self):
        return types.SimpleNamespace(
            apply_chat_template=True,
            max_new_tokens=512,
            temperature=1.0,
            do_sample=True,
            top_p=1.0,
            top_k=0,
            repetition_penalty=1.0,
            stop_strings=set(),
        )

    def get_tokenizer(self):
        return FakeTokenizer()

    def generate(self, prompt: str, config: Any, callback):
        self.last_prompt = prompt
        self.last_config = config
        self.calls.append({"prompt": prompt, "config": config})
        if self.raise_error:
            raise RuntimeError(self.raise_error)
        # Split keeping trailing whitespace attached to each word so
        # "".join(chunks) reconstructs response_text exactly, mirroring how
        # subword streaming concatenates back into the full text.
        chunks = re.findall(r"\S+\s*", self.response_text)
        for chunk in chunks:
            status = callback(chunk)
            if status != ov_genai.StreamingStatus.RUNNING:
                break
