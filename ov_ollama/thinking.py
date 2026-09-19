"""Reasoning (<think>...</think>) separation.

Qwen3 (and other reasoning models) prepend a <think>...</think> block to
every response by default. Real Ollama separates this into its own
`thinking` field (see https://docs.ollama.com/capabilities/thinking) so
clients can render it as a collapsible reasoning trace; clients like VS
Code's Ollama provider expect that separation and render raw <think> tags
left inline in `content`/`response` as garbled plain text. We split it out
here so `content`/`response` only ever contains the final answer.
"""

from typing import List, Tuple

from .generation import TokenStream

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
