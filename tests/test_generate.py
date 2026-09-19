"""Tests for POST /api/generate."""

from .conftest import parse_ndjson
from .fakes import FakePipe


def test_non_streaming_splits_thinking_from_response(client):
    r = client.post("/api/generate", json={"prompt": "hi", "stream": False})
    assert r.status_code == 200
    body = r.json()
    assert body["done"] is True
    assert body["done_reason"] == "stop"
    assert body["response"] == "Here is the final answer."
    assert body["thinking"] == "Let me think about this step by step."
    assert body["context"] == []


def test_non_streaming_think_false_drops_thinking(client):
    r = client.post("/api/generate", json={"prompt": "hi", "stream": False, "think": False})
    body = r.json()
    assert "thinking" not in body
    assert body["response"] == "Here is the final answer."


def test_system_field_is_prepended_to_prompt(client):
    r = client.post(
        "/api/generate",
        json={"prompt": "What's next?", "system": "You are terse.", "stream": False},
    )
    assert r.status_code == 200
    assert client.fake_pipe.last_prompt == "You are terse.\n\nWhat's next?"


def test_streaming_reconstructs_full_response(client):
    r = client.post("/api/generate", json={"prompt": "hi", "stream": True})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("application/x-ndjson")
    events = parse_ndjson(r)

    assert all(e["done"] is False for e in events[:-1])
    assert events[-1]["done"] is True
    assert events[-1]["done_reason"] == "stop"

    thinking = "".join(e.get("thinking", "") for e in events)
    content = "".join(e.get("response", "") for e in events)
    assert thinking == "Let me think about this step by step."
    assert content == "Here is the final answer."


def test_streaming_think_false_omits_thinking_events(client):
    r = client.post("/api/generate", json={"prompt": "hi", "stream": True, "think": False})
    events = parse_ndjson(r)
    assert all("thinking" not in e for e in events)
    content = "".join(e.get("response", "") for e in events)
    assert content == "Here is the final answer."


def test_options_are_mapped_onto_generation_config(client):
    r = client.post(
        "/api/generate",
        json={
            "prompt": "hi",
            "stream": False,
            "options": {
                "num_predict": 64,
                "temperature": 0.0,
                "top_p": 0.5,
                "top_k": 10,
                "repeat_penalty": 1.2,
                "stop": ["STOP"],
            },
        },
    )
    assert r.status_code == 200
    config = client.fake_pipe.last_config
    assert config.max_new_tokens == 64
    assert config.temperature == 0.0
    assert config.do_sample is False  # temperature <= 0 disables sampling
    assert config.top_p == 0.5
    assert config.top_k == 10
    assert config.repetition_penalty == 1.2
    assert config.stop_strings == {"STOP"}


def test_apply_chat_template_enabled_for_generate(client):
    """Unlike /api/chat, /api/generate should let the pipeline apply the
    model's chat template itself (it hasn't been rendered already)."""
    client.post("/api/generate", json={"prompt": "hi", "stream": False})
    assert client.fake_pipe.last_config.apply_chat_template is True


def test_generation_error_surfaces_in_non_streaming_response(make_client):
    c = make_client(pipe=FakePipe(raise_error="pipeline exploded"))
    r = c.post("/api/generate", json={"prompt": "hi", "stream": False})
    assert r.status_code == 200
    body = r.json()
    assert body["done"] is True
    assert body["done_reason"] == "error"
    assert "pipeline exploded" in body["error"]


def test_generation_error_surfaces_in_streaming_response(make_client):
    c = make_client(pipe=FakePipe(raise_error="pipeline exploded"))
    r = c.post("/api/generate", json={"prompt": "hi", "stream": True})
    events = parse_ndjson(r)
    final = events[-1]
    assert final["done"] is True
    assert final["done_reason"] == "error"
    assert "pipeline exploded" in final["error"]
