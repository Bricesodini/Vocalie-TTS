from __future__ import annotations

def test_health_and_info(api_client):
    client = api_client
    resp = client.get("/v1/health")
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["status"] == "ok"
    assert "timestamp" in payload

    resp = client.get("/v1/info")
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["name"] == "chatterbox-tts-fr"
    assert "work_dir" in payload
    assert "output_dir" in payload
    assert "presets_dir" in payload
