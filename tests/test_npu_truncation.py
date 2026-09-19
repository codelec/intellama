"""Ultra-long prompt tests for NPU prompt-length truncation.

The NPU backend compiles a static MAX_PROMPT_LEN and hard-errors past it.
ov_ollama.prompts proactively truncates oversized prompts/chat histories so
requests degrade gracefully instead of failing mid-stream. These tests throw
genuinely large inputs (thousands of "tokens" under the FakeTokenizer's
one-word-per-token scheme) at /api/generate and /api/chat to exercise every
truncation step, and confirm CPU/GPU devices are unaffected (no static
limit there).

All assertions use ov_ollama.prompts.count_tokens/npu_prompt_budget directly
(the same functions - and the same STATE-installed tokenizer - the app code
uses) so they hold regardless of the fake tokenizer's exact word-splitting
quirks; the invariant under test is "the algorithm's own budget is honored
and a normal (non-error) response is still produced", not a hand-derived
token count.
"""

from ov_ollama.prompts import count_tokens, npu_prompt_budget

WORD_COUNT = 6000  # comfortably an "ultra-long" prompt under any reasonable budget


def _words(prefix: str, n: int) -> str:
    return " ".join(f"{prefix}{i}" for i in range(n))


def test_generate_ultra_long_prompt_is_truncated_on_npu(make_client, capsys):
    c = make_client(is_npu=True, max_prompt_len=200, min_response_len=50)
    huge_prompt = _words("tailword", WORD_COUNT)  # tailword0 tailword1 ... tailwordN-1

    r = c.post("/api/generate", json={"prompt": huge_prompt, "stream": False})

    assert r.status_code == 200
    body = r.json()
    assert body["done_reason"] == "stop"  # truncation must not surface as an error

    final_prompt = c.fake_pipe.last_prompt
    budget = npu_prompt_budget()
    assert count_tokens(final_prompt) <= budget
    # Tail is kept (most relevant/recent content), head is dropped.
    assert final_prompt.rstrip().endswith(f"tailword{WORD_COUNT - 1}")
    assert "tailword0 " not in final_prompt

    assert "[TRUNCATED]" in capsys.readouterr().out


def test_generate_ultra_long_prompt_untouched_on_cpu(make_client, capsys):
    """Control case: the same ultra-long prompt on a non-NPU device must
    pass through unmodified - there's no static prompt-length limit."""
    c = make_client(is_npu=False, device="CPU")
    huge_prompt = _words("word", WORD_COUNT)

    r = c.post("/api/generate", json={"prompt": huge_prompt, "stream": False})

    assert r.status_code == 200
    assert r.json()["done_reason"] == "stop"
    assert c.fake_pipe.last_prompt == huge_prompt
    assert "[TRUNCATED]" not in capsys.readouterr().out


def test_generate_ultra_long_system_plus_prompt_is_truncated(make_client):
    """A huge system field (e.g. an injected tool schema) combined with a
    normal prompt must still respect the NPU budget."""
    c = make_client(is_npu=True, max_prompt_len=200, min_response_len=50)
    huge_system = _words("schema", WORD_COUNT)

    r = c.post(
        "/api/generate",
        json={"prompt": "What's the weather?", "system": huge_system, "stream": False},
    )

    assert r.status_code == 200
    assert r.json()["done_reason"] == "stop"
    budget = npu_prompt_budget()
    assert count_tokens(c.fake_pipe.last_prompt) <= budget


def test_chat_ultra_long_history_drops_oldest_turns_first(make_client, capsys):
    """Step 1: with many past turns, the oldest are dropped first, keeping
    the most recent user message intact."""
    c = make_client(is_npu=True, max_prompt_len=300, min_response_len=50)

    messages = [{"role": "system", "content": "Be terse."}]
    for i in range(200):
        messages.append({"role": "user", "content": f"turn {i} " + _words("filler", 20)})
        messages.append({"role": "assistant", "content": f"reply {i}"})
    messages.append({"role": "user", "content": "FINAL_MARKER_QUESTION"})

    original_user_turns = sum(1 for m in messages if m["role"] == "user")

    r = c.post("/api/chat", json={"messages": messages, "stream": False})

    assert r.status_code == 200
    assert r.json()["done_reason"] == "stop"

    final_prompt = c.fake_pipe.last_prompt
    budget = npu_prompt_budget()
    assert count_tokens(final_prompt) <= budget
    assert "FINAL_MARKER_QUESTION" in final_prompt
    assert final_prompt.count("<user>") < original_user_turns

    out = capsys.readouterr().out
    assert "[TRUNCATED]" in out
    assert "chat history trimmed from" in out


def test_chat_ultra_long_single_message_content_is_shrunk(make_client, capsys):
    """Step 2: down to system + a single (already-latest) message, the
    message's own content gets shrunk (tail kept) rather than the turn
    being dropped entirely."""
    c = make_client(is_npu=True, max_prompt_len=200, min_response_len=50)

    huge_message = _words("head", WORD_COUNT // 2) + " TAIL_MARKER"
    messages = [
        {"role": "system", "content": "Be terse."},
        {"role": "user", "content": huge_message},
    ]

    r = c.post("/api/chat", json={"messages": messages, "stream": False})

    assert r.status_code == 200
    assert r.json()["done_reason"] == "stop"

    final_prompt = c.fake_pipe.last_prompt
    budget = npu_prompt_budget()
    assert count_tokens(final_prompt) <= budget
    assert "Be terse." in final_prompt  # short system message survives untouched
    assert "TAIL_MARKER" in final_prompt  # tail of the oversized message is kept
    assert "head0 " not in final_prompt  # head of the oversized message is dropped

    out = capsys.readouterr().out
    assert "chat history trimmed to a single message" in out


def test_chat_ultra_long_system_and_message_both_shrunk(make_client, capsys):
    """Step 3: when even the single remaining message can't make it fit
    because the system message itself is huge (e.g. a large tool schema),
    both get shrunk - system keeps its head, the message keeps its tail."""
    c = make_client(is_npu=True, max_prompt_len=100, min_response_len=20)

    huge_system = "SYS_HEAD_MARKER " + _words("schema", WORD_COUNT // 2)
    huge_message = _words("head", WORD_COUNT // 2) + " USER_TAIL_MARKER"
    messages = [
        {"role": "system", "content": huge_system},
        {"role": "user", "content": huge_message},
    ]

    r = c.post("/api/chat", json={"messages": messages, "stream": False})

    assert r.status_code == 200
    assert r.json()["done_reason"] == "stop"

    final_prompt = c.fake_pipe.last_prompt
    budget = npu_prompt_budget()
    assert count_tokens(final_prompt) <= budget
    assert "SYS_HEAD_MARKER" in final_prompt  # system keeps its head
    assert "USER_TAIL_MARKER" in final_prompt  # message keeps its tail
    assert f"schema{WORD_COUNT // 2 - 1}" not in final_prompt  # system tail dropped
    assert "head0 " not in final_prompt  # message head dropped

    out = capsys.readouterr().out
    assert "system prompt and last message both truncated" in out


def test_chat_ultra_long_history_untouched_on_gpu(make_client, capsys):
    """Control case mirroring the drop-oldest-turns scenario, but on a
    non-NPU device: the full history must reach the pipeline unmodified."""
    c = make_client(is_npu=False, device="GPU")

    messages = [{"role": "system", "content": "Be terse."}]
    for i in range(50):
        messages.append({"role": "user", "content": f"turn {i} " + _words("filler", 20)})
        messages.append({"role": "assistant", "content": f"reply {i}"})
    messages.append({"role": "user", "content": "FINAL_MARKER_QUESTION"})

    r = c.post("/api/chat", json={"messages": messages, "stream": False})

    assert r.status_code == 200
    final_prompt = c.fake_pipe.last_prompt
    assert final_prompt.count("<user>") == sum(1 for m in messages if m["role"] == "user")
    assert "[TRUNCATED]" not in capsys.readouterr().out
