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


def test_engine_schema_piper_no_ref(api_client):
    client = api_client

    resp = client.get("/v1/tts/engine_schema", params={"engine": "piper"})
    assert resp.status_code == 200
    capabilities = resp.json()["capabilities"]
    assert capabilities.get("supports_ref") is False


def test_engine_schema_xtts_requires_ref(api_client):
    client = api_client

    resp = client.get("/v1/tts/engine_schema", params={"engine": "xtts_v2"})
    assert resp.status_code == 200
    constraints = resp.json().get("constraints") or {}
    required = constraints.get("required") or []
    assert "voice_id" in required


def test_engine_schema_bark_fields(api_client):
    client = api_client

    resp = client.get("/v1/tts/engine_schema", params={"engine": "bark"})
    assert resp.status_code == 200
    keys = {field["key"] for field in resp.json()["fields"]}
    assert {"voice_preset", "text_temp", "waveform_temp", "seed", "device"}.issubset(keys)
