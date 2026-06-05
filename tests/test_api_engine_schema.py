from __future__ import annotations


def test_engine_schema_chatterbox_gap(api_client):
    client = api_client

    resp = client.get("/v1/tts/engine_schema", params={"engine": "chatterbox_native"})
    assert resp.status_code == 200
    fields = resp.json()["fields"]
    assert any(field["key"] == "chunk_gap_ms" for field in fields)


def test_engine_schema_qwen3_gap(api_client):
    client = api_client

    resp = client.get("/v1/tts/engine_schema", params={"engine": "qwen3_custom"})
    assert resp.status_code == 200
    fields = resp.json()["fields"]
    assert any(field["key"] == "chunk_gap_ms" for field in fields)


def test_engine_schema_cosyvoice_instruct(api_client):
    client = api_client

    resp = client.get("/v1/tts/engine_schema", params={"engine": "cosyvoice_instruct"})
    assert resp.status_code == 200
    data = resp.json()
    fields = data["fields"]
    caps = data.get("capabilities") or {}
    # CosyVoice instruct supports emotion
    assert caps.get("supports_instruct") is True
    assert caps.get("supports_emotion") is True
    # Has chunk_gap (inter_chunk_gap)
    assert any(field["key"] == "chunk_gap_ms" for field in fields)


def test_engine_schema_cosyvoice_clone(api_client):
    client = api_client

    resp = client.get("/v1/tts/engine_schema", params={"engine": "cosyvoice_clone"})
    assert resp.status_code == 200
    caps = resp.json().get("capabilities") or {}
    assert caps.get("supports_ref") is True
    assert caps.get("supports_streaming") is True


def test_engine_schema_cosyvoice_cross(api_client):
    client = api_client

    resp = client.get("/v1/tts/engine_schema", params={"engine": "cosyvoice_cross"})
    assert resp.status_code == 200
    caps = resp.json().get("capabilities") or {}
    assert caps.get("supports_cross_lingual") is True
    assert caps.get("supports_ref") is True


def test_legacy_engines_not_found(api_client):
    """Legacy engine IDs return 404 after cleanup."""
    client = api_client

    for engine in ("bark", "piper", "xtts_v2"):
        resp = client.get("/v1/tts/engine_schema", params={"engine": engine})
        assert resp.status_code == 404, f"Legacy engine {engine} should return 404"