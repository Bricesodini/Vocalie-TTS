from __future__ import annotations


def test_health_is_public(api_client):
    resp = api_client.get("/v1/health", headers={"X-API-Key": ""})
    assert resp.status_code == 200


def test_protected_endpoint_requires_api_key(api_client):
    resp = api_client.get("/v1/capabilities", headers={"X-API-Key": ""})
    assert resp.status_code == 403
    assert resp.json()["detail"] == "forbidden"


def test_protected_endpoint_accepts_valid_api_key(api_client):
    resp = api_client.get("/v1/capabilities")
    assert resp.status_code == 200


def test_preflight_options_does_not_require_api_key(api_client):
    resp = api_client.options(
        "/v1/capabilities",
        headers={
            "Origin": "http://localhost:3018",
            "Access-Control-Request-Method": "GET",
            "X-API-Key": "",
        },
    )
    assert resp.status_code in {200, 204}
