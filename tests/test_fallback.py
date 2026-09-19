"""Tests for the catch-all fallback that covers real Ollama endpoints this
server doesn't implement (e.g. /api/pull, /api/embed). Well-behaved clients
(including the VS Code Ollama provider, which probes a few optional
endpoints) must get a clean 404 + JSON error body, never a 500 or hang."""

import pytest


@pytest.mark.parametrize(
    "method,path",
    [
        ("POST", "/api/pull"),
        ("POST", "/api/push"),
        ("POST", "/api/create"),
        ("POST", "/api/copy"),
        ("POST", "/api/delete"),
        ("POST", "/api/embed"),
        ("POST", "/api/embeddings"),
        ("GET", "/api/unknown-thing"),
    ],
)
def test_unimplemented_endpoints_return_404_json(client, method, path):
    r = client.request(method, path, json={})
    assert r.status_code == 404
    body = r.json()
    assert path in body["error"]


def test_fallback_does_not_shadow_real_routes(client):
    """Sanity check that route registration order is preserved: the
    catch-all must never intercept a real endpoint."""
    r = client.get("/api/tags")
    assert r.status_code == 200
    assert "error" not in r.json()
