from __future__ import annotations

import backend.config as backend_config
import backend.rate_limit as rate_limit
import backend.services.job_service as job_service


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
