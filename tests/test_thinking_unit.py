"""Pure unit tests for ov_ollama.thinking.split_thinking_stream, independent
of the HTTP layer. Focuses on the tag being split awkwardly across chunk
boundaries, since that's the tricky part of streaming subword tokens."""

from ov_ollama.thinking import split_thinking_stream


def run(chunks):
    return list(split_thinking_stream(iter(chunks)))


def test_open_tag_split_across_many_chunks():
    chunks = ["<", "th", "ink", ">", "reasoning", "</", "think", ">", "answer"]
    result = run(chunks)
    thinking = "".join(t for k, t in result if k == "thinking")
    content = "".join(t for k, t in result if k == "content")
    assert thinking == "reasoning"
    assert content == "answer"


def test_leading_whitespace_before_think_tag_is_tolerated():
    chunks = ["  \n", "<think>", "reasoning", "</think>", "answer"]
    result = run(chunks)
    thinking = "".join(t for k, t in result if k == "thinking")
    content = "".join(t for k, t in result if k == "content")
    assert thinking == "reasoning"
    assert content == "answer"


def test_no_think_tag_passes_through_as_content():
    chunks = ["just ", "a ", "normal ", "reply"]
    result = run(chunks)
    assert all(k == "content" for k, _ in result)
    assert "".join(t for _, t in result) == "just a normal reply"


def test_unterminated_think_block_at_eof_is_still_reported_as_thinking():
    """If the stream ends mid-<think> block (e.g. generation cut off by
    max_new_tokens), whatever was buffered should be surfaced as thinking
    rather than silently dropped."""
    chunks = ["<think>", "partial reasoning that never closes"]
    result = run(chunks)
    assert result == [("thinking", "partial reasoning that never closes")]


def test_empty_think_block():
    chunks = ["<think></think>", "answer only"]
    result = run(chunks)
    assert ("thinking", "") not in result  # empty thinking segments aren't emitted
    content = "".join(t for k, t in result if k == "content")
    assert content == "answer only"


def test_close_tag_split_at_every_possible_boundary():
    """Sweeps every possible split point of "</think>" to make sure the
    partial-tag holdback logic never leaks a fragment of the tag into the
    thinking text or fails to detect the tag."""
    close = "</think>"
    for i in range(1, len(close)):
        chunks = ["<think>", "reasoning", close[:i], close[i:], "answer"]
        result = run(chunks)
        thinking = "".join(t for k, t in result if k == "thinking")
        content = "".join(t for k, t in result if k == "content")
        assert thinking == "reasoning", f"split at {i} produced thinking={thinking!r}"
        assert content == "answer", f"split at {i} produced content={content!r}"


def test_single_chunk_containing_entire_response():
    chunks = ["<think>all reasoning here</think>final content here"]
    result = run(chunks)
    thinking = "".join(t for k, t in result if k == "thinking")
    content = "".join(t for k, t in result if k == "content")
    assert thinking == "all reasoning here"
    assert content == "final content here"
