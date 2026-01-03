from __future__ import annotations


def test_prep_adjust_and_interpret(api_client):
    client = api_client

    adjust_resp = client.post("/v1/prep/adjust", json={"text_raw": "Bonjour  monde"})
    assert adjust_resp.status_code == 200
    adjusted = adjust_resp.json()["text_adjusted"]
    assert isinstance(adjusted, str)

    interpret_resp = client.post(
        "/v1/prep/interpret",
        json={"text_adjusted": adjusted, "glossary_enabled": False},
    )
    assert interpret_resp.status_code == 200
    payload = interpret_resp.json()
    assert payload["text_interpreted"] == adjusted
