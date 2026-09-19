"""Prompt assembly and NPU prompt-length handling.

The NPU backend uses static input shapes: it hard-errors if a prompt exceeds
MAX_PROMPT_LEN (openvino_genai default: 1024 tokens), which is easy to hit
with clients that inject large system prompts/tool schemas or long chat
histories (e.g. VS Code Copilot Chat). That error surfaces deep inside
pipe.generate(), on the streaming worker thread, after the HTTP response has
already started - which is enough to make picky clients report a generic
"stream ended unexpectedly" error instead of a useful message. We proactively
count tokens and truncate before ever calling generate(), so oversized
requests degrade gracefully (dropped history) instead of failing outright.
This is a no-op on CPU/GPU, which have no static prompt-length limit.
"""

from typing import Any, Dict, List

from .state import STATE

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
    # Kept pristine so step 3 can re-truncate the real content with its own
    # budget, instead of re-truncating whatever step 2 left behind below
    # (which can be emptied out entirely if the system prompt alone is huge).
    original_last_content = last["content"] if last is not None else None

    # Step 2: down to system + at most one message: shrink that message's content.
    if last is not None:
        overhead = count_tokens(render(system, [dict(last, content="")]))
        last["content"] = _truncate_text_to_tokens(original_last_content, max(budget - overhead, 0))
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
            last["content"] = _truncate_text_to_tokens(original_last_content, msg_budget)
            final_messages.append(last)
        prompt = render([], final_messages)
        print(
            f"[TRUNCATED] system prompt and last message both truncated to fit NPU "
            f"MAX_PROMPT_LEN={STATE['max_prompt_len']} (see --max-prompt-len)",
            flush=True,
        )
    return prompt
