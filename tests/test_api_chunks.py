from __future__ import annotations


def test_chunks_snapshot_marker_preview(api_client):
    client = api_client

    snapshot_resp = client.post("/v1/chunks/snapshot", json={"text_interpreted": "Bonjour le monde"})
    assert snapshot_resp.status_code == 200
    snapshot_text = snapshot_resp.json()["snapshot_text"]
    assert "Bonjour" in snapshot_text

    insert_resp = client.post(
        "/v1/chunks/apply_marker",
        json={"snapshot_text": snapshot_text, "action": "insert", "position": 7},
    )
    assert insert_resp.status_code == 200
    updated = insert_resp.json()["snapshot_text_updated"]
    assert "[[CHUNK]]" in updated

    preview_resp = client.post("/v1/chunks/preview", json={"snapshot_text": updated})
    assert preview_resp.status_code == 200
    chunks = preview_resp.json()["chunks"]
    assert len(chunks) >= 1
