"""Process-wide state for the single loaded model.

This server intentionally serves exactly one model on one device per process,
so a module-level dict is enough: it's populated once at startup by
`cli.main()` and read from everywhere else.
"""

import threading
from typing import Any, Dict

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
    "debug": False,  # overridable via --debug; logs raw prompts/messages and raw model output
}
