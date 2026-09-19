"""End-to-end simulation of the VS Code Copilot Chat "Ollama" provider
(the official ollama-vscode extension) talking to this server.

This isn't a literal capture of the extension's network traffic, but it
scripts the documented/observed request sequence closely enough to catch
integration regressions a unit test per-route wouldn't:

  1. GET  /api/version                        - reachability/compat check
  2. GET  /api/tags                            - populate the model picker
  3. GET  /api/experimental/model-recommendations - suggested-models panel
  4. POST /api/show                            - model details for the picker
  5. POST /api/chat (stream: true)             - an actual chat turn, with a
     large injected system prompt (VS Code routinely attaches a big tool/
     function-calling schema as part of the system message) plus prior
     conversation history.
  6. A growing multi-turn conversation, eventually large enough to need NPU
     truncation, while still producing a well-formed NDJSON stream.

Each step asserts the response shape a real client would rely on to keep
functioning (status code, content-type, NDJSON contract, non-empty model
list, etc.).
"""

import json

from .conftest import parse_ndjson
from .fakes import FakePipe

TOOL_SCHEMA_SYSTEM_PROMPT = "You are a coding assistant with tool access.\n\nTOOLS:\n" + "\n".join(
    json.dumps(
        {
            "name": f"tool_{i}",
            "description": "A representative tool definition, repeated many times to " * 3,
            "parameters": {
                "type": "object",
                "properties": {f"arg_{j}": {"type": "string", "description": "x" * 40} for j in range(8)},
            },
        }
    )
    for i in range(80)
)


def _assert_well_formed_ndjson_chat_stream(events):
    assert events, "expected at least one streamed event"
    for e in events[:-1]:
        assert e["done"] is False
        assert e["message"]["role"] == "assistant"
    final = events[-1]
    assert final["done"] is True
    assert final["done_reason"] in ("stop", "error")
    assert "model" in final and "created_at" in final


def test_vscode_extension_startup_and_discovery_sequence(make_client):
    """Steps 1-4: what the extension does before a user has typed anything."""
    c = make_client(served_name="qwen3:8b", device="GPU")

    version = c.get("/api/version")
    assert version.status_code == 200
    assert isinstance(version.json().get("version"), str)

    tags = c.get("/api/tags")
    assert tags.status_code == 200
    model_names = [m["name"] for m in tags.json()["models"]]
    assert "qwen3:8b" in model_names

    recs = c.get("/api/experimental/model-recommendations")
    assert recs.status_code == 200
    assert all(isinstance(r.get("model"), str) and r["model"] for r in recs.json()["recommendations"])

    show = c.post("/api/show", json={"name": "qwen3:8b"})
    assert show.status_code == 200
    assert "modelfile" in show.json()


def test_vscode_extension_chat_turn_with_large_tool_schema_system_prompt(make_client):
    """Step 5: a single chat turn carrying a large injected tool-schema
    system prompt (as VS Code Copilot Chat does), streamed NDJSON."""
    c = make_client(served_name="qwen3:8b", device="GPU")

    messages = [
        {"role": "system", "content": TOOL_SCHEMA_SYSTEM_PROMPT},
        {"role": "user", "content": "Open the file utils.py and summarize it."},
    ]
    r = c.post(
        "/api/chat",
        json={"model": "qwen3:8b", "messages": messages, "stream": True},
    )
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("application/x-ndjson")

    events = parse_ndjson(r)
    _assert_well_formed_ndjson_chat_stream(events)

    content = "".join(e["message"].get("content", "") for e in events)
    thinking = "".join(e["message"].get("thinking", "") for e in events)
    assert content == "Here is the final answer."
    assert thinking == "Let me think about this step by step."

    # The full tool schema must have reached the model unmodified on a
    # non-NPU device (no static prompt-length limit to worry about).
    assert "tool_0" in c.fake_pipe.last_prompt
    assert "tool_79" in c.fake_pipe.last_prompt


def test_vscode_extension_growing_conversation_triggers_npu_truncation(make_client):
    """Step 6: the extension keeps appending turns as the conversation
    grows. On NPU this eventually exceeds the static prompt budget - the
    server must keep responding normally (not error out or hang), per the
    truncation contract in ov_ollama/prompts.py."""
    c = make_client(
        served_name="qwen3:8b",
        device="NPU",
        is_npu=True,
        max_prompt_len=512,
        min_response_len=64,
    )

    messages = [{"role": "system", "content": TOOL_SCHEMA_SYSTEM_PROMPT}]
    responses = []
    for turn in range(40):
        messages.append({"role": "user", "content": f"Turn {turn}: please look at file_{turn}.py and explain it."})
        r = c.post(
            "/api/chat",
            json={"model": "qwen3:8b", "messages": messages, "stream": True, "think": False},
        )
        assert r.status_code == 200
        events = parse_ndjson(r)
        _assert_well_formed_ndjson_chat_stream(events)
        assert all("thinking" not in e["message"] for e in events)  # think: false honored
        final_content = "".join(e["message"].get("content", "") for e in events)
        assert final_content == "Here is the final answer."
        messages.append({"role": "assistant", "content": final_content})
        responses.append(r)

    # By the last turn, the rendered prompt must respect the NPU budget even
    # though the accumulated conversation is now far larger than it.
    from ov_ollama.prompts import count_tokens, npu_prompt_budget

    assert count_tokens(c.fake_pipe.last_prompt) <= npu_prompt_budget()
    # The most recent user turn is always preserved even when older history
    # gets dropped to make room.
    assert "Turn 39" in c.fake_pipe.last_prompt


def test_vscode_extension_handles_generation_error_gracefully(make_client):
    """If the pipeline itself fails mid-conversation, the extension should
    still get a well-formed NDJSON stream ending in done:true/error rather
    than a broken connection."""
    c = make_client(pipe=FakePipe(raise_error="device lost"), served_name="qwen3:8b")
    r = c.post(
        "/api/chat",
        json={"model": "qwen3:8b", "messages": [{"role": "user", "content": "hi"}], "stream": True},
    )
    events = parse_ndjson(r)
    assert events[-1]["done"] is True
    assert events[-1]["done_reason"] == "error"
    assert "device lost" in events[-1]["error"]
