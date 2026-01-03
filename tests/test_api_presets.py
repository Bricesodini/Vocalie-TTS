from __future__ import annotations


def test_presets_crud(api_client):
    client = api_client

    create_resp = client.post(
        "/v1/presets",
        json={
            "id": "test-preset",
            "label": "Test Preset",
            "state": {
                "preparation": {"text_raw": "Bonjour"},
                "engine": {"engine_id": "piper", "params": {}},
            },
        },
    )
    assert create_resp.status_code == 200

    list_resp = client.get("/v1/presets")
    assert list_resp.status_code == 200
    preset_ids = {item["id"] for item in list_resp.json()["presets"]}
    assert "test-preset" in preset_ids

    get_resp = client.get("/v1/presets/test-preset")
    assert get_resp.status_code == 200
    payload = get_resp.json()
    assert payload["id"] == "test-preset"
    assert payload["state"]["preparation"]["text_raw"] == "Bonjour"

    update_resp = client.put(
        "/v1/presets/test-preset",
        json={
            "label": "Updated",
            "state": {
                "preparation": {"text_raw": "Salut"},
                "engine": {"engine_id": "piper", "params": {}},
            },
        },
    )
    assert update_resp.status_code == 200

    get_resp = client.get("/v1/presets/test-preset")
    assert get_resp.status_code == 200
    assert get_resp.json()["state"]["preparation"]["text_raw"] == "Salut"

    delete_resp = client.delete("/v1/presets/test-preset")
    assert delete_resp.status_code == 200

    missing_resp = client.get("/v1/presets/test-preset")
    assert missing_resp.status_code == 404
