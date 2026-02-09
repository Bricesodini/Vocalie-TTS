from __future__ import annotations

import backend.app as backend_app


def test_docs_routes_disabled_by_default():
    assert backend_app.app.docs_url is None
    assert backend_app.app.redoc_url is None
    assert backend_app.app.openapi_url is None


def test_trusted_host_rejects_unexpected_host(api_client):
    resp = api_client.get("/v1/health", headers={"Host": "evil.example.com"})
    assert resp.status_code == 400
