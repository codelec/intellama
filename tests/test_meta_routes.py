"""Tests for the metadata/discovery routes in ov_ollama/api_meta.py.

Several of these mirror what the VS Code Copilot Chat "Ollama" provider
(the official ollama-vscode extension) queries when it starts up: /api/tags
to populate the model picker, /api/show for model details, and
/api/experimental/model-recommendations to highlight suggestions.
"""


def test_root_reports_status(client):
    r = client.get("/")
    assert r.status_code == 200
    body = r.json()
    assert body == {
        "status": "ok",
        "backend": "openvino-genai",
        "model": "test-model",
        "device": "CPU",
    }


def test_version(client):
    r = client.get("/api/version")
    assert r.status_code == 200
    assert "version" in r.json()


def test_tags_lists_served_model(client):
    r = client.get("/api/tags")
    assert r.status_code == 200
    models = r.json()["models"]
    assert len(models) == 1
    assert models[0]["name"] == "test-model"
    assert models[0]["details"]["format"] == "openvino"


def test_ps_reports_zero_vram_on_cpu(client):
    r = client.get("/api/ps")
    assert r.status_code == 200
    model = r.json()["models"][0]
    assert model["name"] == "test-model"
    assert model["size_vram"] == 0
    assert model["expires_at"] == "0001-01-01T00:00:00Z"


def test_ps_reports_full_size_as_vram_on_gpu(make_client, tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "weights.bin").write_bytes(b"x" * 4096)

    c = make_client(device="GPU", model_dir=str(model_dir))
    model = c.get("/api/ps").json()["models"][0]
    assert model["size"] == 4096
    assert model["size_vram"] == 4096


def test_show_reports_device_and_model_dir(make_client, tmp_path):
    c = make_client(model_dir=str(tmp_path / "my-model"), device="GPU")
    r = c.post("/api/show", json={})
    assert r.status_code == 200
    body = r.json()
    assert "my-model" in body["modelfile"]
    assert body["model_info"]["ov.device"] == "GPU"


def test_model_recommendations_served_model_first(client):
    r = client.get("/api/experimental/model-recommendations")
    assert r.status_code == 200
    recs = r.json()["recommendations"]
    assert recs[0]["model"] == "test-model"
    assert len(recs) > 1


def test_model_recommendations_every_entry_has_string_model_field(client):
    """Real Ollama/VS Code clients silently drop any recommendation entry
    without a string "model" field - a regression here would make the
    endpoint appear to return nothing to those clients."""
    r = client.get("/api/experimental/model-recommendations")
    for entry in r.json()["recommendations"]:
        assert isinstance(entry.get("model"), str) and entry["model"]
