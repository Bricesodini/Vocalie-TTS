"""Tests for /v1/tts/models endpoint."""

from __future__ import annotations


def test_list_models_qwen3(api_client):
    resp = api_client.get("/v1/tts/models", params={"engine": "qwen3"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["engine"] == "qwen3"
    assert isinstance(data["models"], list)


def test_list_models_qwen3_custom_variant(api_client):
    """Engine variant should resolve to its backend and return models."""
    resp = api_client.get("/v1/tts/models", params={"engine": "qwen3_custom"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["engine"] == "qwen3_custom"
    assert isinstance(data["models"], list)


def test_list_models_chatterbox(api_client):
    resp = api_client.get("/v1/tts/models", params={"engine": "chatterbox"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["engine"] == "chatterbox"
    assert isinstance(data["models"], list)


def test_list_models_chatterbox_native_variant(api_client):
    resp = api_client.get("/v1/tts/models", params={"engine": "chatterbox_native"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["engine"] == "chatterbox_native"
    assert isinstance(data["models"], list)


def test_list_models_not_found(api_client):
    resp = api_client.get("/v1/tts/models", params={"engine": "nonexistent_backend"})
    assert resp.status_code == 404


def test_qwen3_models_have_ids_and_labels(api_client):
    """Qwen3 models should include the HF model identifiers."""
    resp = api_client.get("/v1/tts/models", params={"engine": "qwen3"})
    assert resp.status_code == 200
    models = resp.json()["models"]
    assert len(models) > 0
    for m in models:
        assert "id" in m
        assert "label" in m
        assert isinstance(m["id"], str)
        assert isinstance(m["label"], str)


def test_chatterbox_models_have_ids_and_labels(api_client):
    """Chatterbox models should include the HF model identifiers."""
    resp = api_client.get("/v1/tts/models", params={"engine": "chatterbox"})
    assert resp.status_code == 200
    models = resp.json()["models"]
    assert len(models) > 0
    for m in models:
        assert "id" in m
        assert "label" in m
        assert isinstance(m["id"], str)
        assert isinstance(m["label"], str)


def test_qwen3_capabilities_supports_voice_design(api_client):
    """Qwen3_custom should expose supports_voice_design capability."""
    resp = api_client.get("/v1/tts/engine_schema", params={"engine": "qwen3_custom"})
    assert resp.status_code == 200
    caps = resp.json().get("capabilities", {})
    assert caps.get("supports_voice_design") is True


def test_chatterbox_no_voice_design(api_client):
    """Chatterbox should NOT expose supports_voice_design."""
    resp = api_client.get("/v1/tts/engine_schema", params={"engine": "chatterbox_native"})
    assert resp.status_code == 200
    caps = resp.json().get("capabilities", {})
    assert caps.get("supports_voice_design") is not True