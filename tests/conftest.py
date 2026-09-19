import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest
from fastapi.testclient import TestClient

# Make the repo root importable (tests/ is not a package under it).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ov_ollama.app import app  # noqa: E402
from ov_ollama.state import STATE  # noqa: E402

from .fakes import FakePipe  # noqa: E402


@pytest.fixture
def state_snapshot():
    """Snapshots STATE before a test and restores it after, so tests can
    freely mutate the shared, module-level STATE dict without leaking
    changes (e.g. is_npu, max_prompt_len) into other tests."""
    original = dict(STATE)
    yield STATE
    STATE.clear()
    STATE.update(original)


def _install_fake_pipe(state: Dict[str, Any], **overrides) -> FakePipe:
    pipe = overrides.pop("pipe", None) or FakePipe(
        response_text=overrides.pop("response_text", None),
        raise_error=overrides.pop("raise_error", None),
    )
    state["pipe"] = pipe
    state["tokenizer"] = pipe.get_tokenizer()
    state["served_name"] = overrides.pop("served_name", "test-model")
    state["device"] = overrides.pop("device", "CPU")
    state["model_dir"] = overrides.pop("model_dir", "")
    state["max_new_tokens"] = overrides.pop("max_new_tokens", 512)
    state["is_npu"] = overrides.pop("is_npu", False)
    state["max_prompt_len"] = overrides.pop("max_prompt_len", 1024)
    state["min_response_len"] = overrides.pop("min_response_len", 128)
    assert not overrides, f"Unknown fixture overrides: {sorted(overrides)}"
    return pipe


@pytest.fixture
def client(state_snapshot):
    """A TestClient backed by a FakePipe on a plain (non-NPU) device, the
    common case for /api/generate and /api/chat tests."""
    fake_pipe = _install_fake_pipe(state_snapshot)
    test_client = TestClient(app)
    test_client.fake_pipe = fake_pipe  # convenient handle for assertions
    return test_client


@pytest.fixture
def make_client(state_snapshot):
    """Factory fixture for tests that need non-default STATE (e.g. NPU mode
    with a tight max_prompt_len/min_response_len budget, or a custom served
    name/device for /api/ps and /api/show assertions)."""

    def _make(**overrides) -> TestClient:
        fake_pipe = _install_fake_pipe(state_snapshot, **overrides)
        test_client = TestClient(app)
        test_client.fake_pipe = fake_pipe
        return test_client

    return _make


def parse_ndjson(response) -> List[Dict[str, Any]]:
    """Parses an Ollama-style newline-delimited JSON streaming response body
    into a list of decoded objects, in order."""
    lines = [line for line in response.text.split("\n") if line.strip()]
    return [json.loads(line) for line in lines]
