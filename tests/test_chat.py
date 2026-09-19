"""Tests for POST /api/chat."""

from .conftest import parse_ndjson
from .fakes import FakePipe


def test_non_streaming_chat_returns_message_with_thinking(client):
    r = client.post(
        "/api/chat",
        json={"messages": [{"role": "user", "content": "hi"}], "stream": False},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["message"]["role"] == "assistant"
    assert body["message"]["content"] == "Here is the final answer."
    assert body["message"]["thinking"] == "Let me think about this step by step."
    assert body["done_reason"] == "stop"


def test_apply_chat_template_disabled_for_chat(client):
    """/api/chat renders the template itself via build_chat_prompt(), so the
    pipeline must not re-apply it (that would double-template and confuse
    NPU token budgeting)."""
    client.post("/api/chat", json={"messages": [{"role": "user", "content": "hi"}], "stream": False})
    assert client.fake_pipe.last_config.apply_chat_template is False
    assert client.fake_pipe.last_prompt == "<user>hi</user>\n<assistant>"


def test_chat_template_includes_full_history_and_system_message(client):
    messages = [
        {"role": "system", "content": "Be terse."},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "how are you"},
    ]
    client.post("/api/chat", json={"messages": messages, "stream": False})
    assert client.fake_pipe.last_prompt == (
        "<system>Be terse.</system>\n"
        "<user>hi</user>\n"
        "<assistant>hello</assistant>\n"
        "<user>how are you</user>\n"
        "<assistant>"
    )


def test_streaming_chat_reconstructs_message(client):
    r = client.post(
        "/api/chat",
        json={"messages": [{"role": "user", "content": "hi"}], "stream": True},
    )
    assert r.status_code == 200
    events = parse_ndjson(r)
    assert all(e["done"] is False for e in events[:-1])
    assert events[-1]["done"] is True

    content = "".join(e["message"].get("content", "") for e in events)
    thinking = "".join(e["message"].get("thinking", "") for e in events)
    assert content == "Here is the final answer."
    assert thinking == "Let me think about this step by step."


def test_streaming_chat_think_false_omits_thinking(client):
    r = client.post(
        "/api/chat",
        json={"messages": [{"role": "user", "content": "hi"}], "stream": True, "think": False},
    )
    events = parse_ndjson(r)
    assert all("thinking" not in e["message"] for e in events)


def test_thinking_tag_split_across_stream_chunks(make_client):
    """The FakePipe streams word-by-word, so a multi-word <think> block
    naturally exercises the tag-spanning-chunks logic in split_thinking_stream
    end-to-end through the real HTTP response."""
    pipe = FakePipe(response_text="<think>reasoning across several words</think>final reply here")
    c = make_client(pipe=pipe)
    r = c.post("/api/chat", json={"messages": [{"role": "user", "content": "hi"}], "stream": False})
    body = r.json()
    assert body["message"]["thinking"] == "reasoning across several words"
    assert body["message"]["content"] == "final reply here"


def test_response_with_no_think_tag_has_no_thinking_field(make_client):
    pipe = FakePipe(response_text="just a plain answer, no reasoning block")
    c = make_client(pipe=pipe)
    r = c.post("/api/chat", json={"messages": [{"role": "user", "content": "hi"}], "stream": False})
    body = r.json()
    assert "thinking" not in body["message"]
    assert body["message"]["content"] == "just a plain answer, no reasoning block"


def test_chat_preflight_error_when_tokenizer_fails(make_client, monkeypatch):
    """Simulates a failure before generation starts (e.g. a chat-template
    error) and checks it's reported in the same done:true/error shape as a
    normal response, per preflight_error_response()'s contract."""
    import ov_ollama.api_inference as api_inference

    def boom(_messages):
        raise ValueError("template exploded")

    monkeypatch.setattr(api_inference, "build_chat_prompt", boom)
    c = make_client()
    r = c.post("/api/chat", json={"messages": [{"role": "user", "content": "hi"}], "stream": False})
    body = r.json()
    assert body["done"] is True
    assert body["done_reason"] == "error"
    assert "template exploded" in body["error"]


def test_chat_generation_error_surfaces_in_response(make_client):
    c = make_client(pipe=FakePipe(raise_error="boom"))
    r = c.post("/api/chat", json={"messages": [{"role": "user", "content": "hi"}], "stream": False})
    body = r.json()
    assert body["done_reason"] == "error"
    assert "boom" in body["error"]
