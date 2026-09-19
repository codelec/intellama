"""Tests for client-disconnect cancellation.

These exercise the cancellation plumbing (TokenStream.cancel() and
ov_ollama.api_inference._cancel_on_disconnect) directly rather than through
a full HTTP round-trip, since simulating a genuine mid-stream client
disconnect through FastAPI's TestClient isn't practical. What actually
matters - and is fully testable at this level - is: the watcher notices a
disconnect, cancels the stream, generation stops early, and the shared
STATE["gen_lock"] is released for the next queued request.
"""

import asyncio
import time

from ov_ollama.api_inference import _cancel_on_disconnect
from ov_ollama.generation import TokenStream
from ov_ollama.state import STATE

from .fakes import FakePipe


class _FakeRequest:
    """Stands in for fastapi.Request: is_disconnected() flips to True once
    `disconnect_after` calls have been made, mimicking a client that gives
    up partway through a long-running generation."""

    def __init__(self, disconnect_after: int = 1):
        self._calls = 0
        self._disconnect_after = disconnect_after

    async def is_disconnected(self) -> bool:
        self._calls += 1
        return self._calls > self._disconnect_after


def _install_slow_pipe(state):
    """A paced pipe that emits many chunks, so there's time to cancel
    mid-stream instead of the whole (otherwise instant) response completing
    before a test gets a chance to act."""
    pipe = FakePipe(response_text=" ".join(f"word{i}" for i in range(2000)), delay=0.002)
    state["pipe"] = pipe
    state["tokenizer"] = pipe.get_tokenizer()
    return pipe


def test_cancel_on_disconnect_stops_generation_early(state_snapshot):
    pipe = _install_slow_pipe(state_snapshot)
    config = pipe.get_generation_config()
    stream = TokenStream().start("hi", config)

    request = _FakeRequest(disconnect_after=1)
    asyncio.run(_cancel_on_disconnect(request, stream, poll_interval=0.01))

    assert stream.cancelled.is_set()
    assert stream.finished.wait(timeout=5)
    # Cut off early - must not have streamed all 2000 words.
    assert stream.token_count < 2000
    assert stream.error is None  # cancellation isn't reported as an error


def test_cancel_releases_gen_lock_for_next_request(state_snapshot):
    pipe = _install_slow_pipe(state_snapshot)
    config = pipe.get_generation_config()
    stream = TokenStream().start("hi", config)

    time.sleep(0.05)  # let generation get going
    stream.cancel()
    assert stream.finished.wait(timeout=5)

    # The shared lock must be free again immediately after cancellation -
    # this is the actual fix for requests queuing up behind an abandoned one.
    assert STATE["gen_lock"].acquire(blocking=False)
    STATE["gen_lock"].release()


def test_watcher_stops_without_cancelling_once_generation_finishes_on_its_own(state_snapshot):
    """If generation finishes normally before the client ever disconnects,
    the watcher must notice via `stream.finished` and stop cleanly instead
    of cancelling a request that already succeeded."""
    pipe = FakePipe(response_text="short reply")
    state_snapshot["pipe"] = pipe
    state_snapshot["tokenizer"] = pipe.get_tokenizer()
    config = pipe.get_generation_config()
    stream = TokenStream().start("hi", config)
    assert stream.finished.wait(timeout=5)

    request = _FakeRequest(disconnect_after=10_000)  # never disconnects
    asyncio.run(_cancel_on_disconnect(request, stream, poll_interval=0.01))
    assert not stream.cancelled.is_set()


def test_watcher_survives_disconnect_check_errors(state_snapshot):
    """A broken is_disconnected() (e.g. an ASGI edge case) must not affect
    the actual response - the watcher is best-effort only."""

    class _BrokenRequest:
        async def is_disconnected(self) -> bool:
            raise RuntimeError("boom")

    pipe = _install_slow_pipe(state_snapshot)
    config = pipe.get_generation_config()
    stream = TokenStream().start("hi", config)

    # Must not raise, and must not have cancelled anything.
    asyncio.run(_cancel_on_disconnect(_BrokenRequest(), stream, poll_interval=0.01))
    assert not stream.cancelled.is_set()
    stream.cancel()  # cleanup so the background thread doesn't linger
