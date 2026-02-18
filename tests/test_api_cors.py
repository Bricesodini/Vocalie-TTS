from __future__ import annotations


def test_cors_allows_localhost(api_client):
    resp = api_client.options(
        "/v1/health",
        headers={
            "Origin": "http://localhost:3018",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert resp.status_code in {200, 204}
    assert resp.headers.get("access-control-allow-origin") == "http://localhost:3018"


def test_cors_blocks_unknown_origin(api_client):
    resp = api_client.options(
        "/v1/health",
        headers={
            "Origin": "http://evil.com",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert resp.status_code in {200, 204, 400}
    assert resp.headers.get("access-control-allow-origin") is None
