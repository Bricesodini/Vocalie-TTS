from __future__ import annotations

import backend.config as backend_config
import backend.rate_limit as rate_limit
import backend.services.job_service as job_service
from starlette.requests import Request


def test_rate_limit_not_applied_to_health(api_client):
    for _ in range(30):
        resp = api_client.get("/v1/health")
        assert resp.status_code == 200


def test_rate_limit_applied_to_tts_jobs(api_client, monkeypatch):
    monkeypatch.setattr(backend_config, "MAX_CONCURRENT_JOBS", 1000, raising=False)
    monkeypatch.setattr(rate_limit, "VOCALIE_RATE_LIMIT_RPS", 1.0, raising=False)
    monkeypatch.setattr(rate_limit, "VOCALIE_RATE_LIMIT_BURST", 2, raising=False)

    def fake_run_tts_job(**_kwargs):
        raise RuntimeError("should_not_run")

    monkeypatch.setattr(job_service, "run_tts_job", fake_run_tts_job)

    statuses = []
    for _ in range(20):
        resp = api_client.post(
            "/v1/tts/jobs",
            json={
                "text": "Bonjour",
                "engine": "chatterbox_finetune_fr",
                "direction": {"enabled": False},
            },
        )
        statuses.append(resp.status_code)
    assert 429 in statuses


def _make_request(*, peer: str, xff: str | None = None, api_key: str | None = None) -> Request:
    headers: list[tuple[bytes, bytes]] = []
    if xff:
        headers.append((b"x-forwarded-for", xff.encode("ascii")))
    if api_key:
        headers.append((b"x-api-key", api_key.encode("ascii")))
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": "/v1/tts/jobs",
        "raw_path": b"/v1/tts/jobs",
        "query_string": b"",
        "headers": headers,
        "client": (peer, 12345),
        "server": ("testserver", 80),
    }
    return Request(scope)


def test_rate_limit_uses_forwarded_ip_only_from_trusted_proxy(monkeypatch):
    rate_limit._BUCKETS.clear()
    monkeypatch.setattr(rate_limit, "VOCALIE_TRUSTED_PROXIES", ["testclient"], raising=False)

    req_a = _make_request(peer="testclient", xff="203.0.113.1")
    req_b = _make_request(peer="testclient", xff="198.51.100.2")

    assert rate_limit.consume(req_a, rps=0.01, burst=1) is True
    assert rate_limit.consume(req_b, rps=0.01, burst=1) is True
    assert rate_limit.consume(req_a, rps=0.01, burst=1) is False


def test_rate_limit_ignores_forwarded_ip_from_untrusted_peer(monkeypatch):
    rate_limit._BUCKETS.clear()
    monkeypatch.setattr(rate_limit, "VOCALIE_TRUSTED_PROXIES", ["127.0.0.1"], raising=False)

    req_a = _make_request(peer="198.18.0.10", xff="203.0.113.1")
    req_b = _make_request(peer="198.18.0.10", xff="198.51.100.2")

    assert rate_limit.consume(req_a, rps=0.01, burst=1) is True
    assert rate_limit.consume(req_b, rps=0.01, burst=1) is False


def test_rate_limit_bucket_includes_api_key(monkeypatch):
    rate_limit._BUCKETS.clear()
    monkeypatch.setattr(rate_limit, "VOCALIE_TRUSTED_PROXIES", [], raising=False)

    req_a = _make_request(peer="198.18.0.10", api_key="alpha-key")
    req_b = _make_request(peer="198.18.0.10", api_key="beta-key")

    assert rate_limit.consume(req_a, rps=0.01, burst=1) is True
    assert rate_limit.consume(req_b, rps=0.01, burst=1) is True
    assert rate_limit.consume(req_a, rps=0.01, burst=1) is False
