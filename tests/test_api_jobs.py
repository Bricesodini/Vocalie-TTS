from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf

import backend.services.job_service as job_service


def test_job_lifecycle(api_client, monkeypatch, tmp_path):
    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    def fake_run_tts_job(
        *,
        job_id,
        text,
        engine,
        voice,
        model,
        language,
        direction_enabled,
        direction_marker,
        options,
        export,
        editing,
        progress_cb,
    ):
        progress_cb(0.3)
        path = output_dir / "fake.wav"
        sr = 24000
        tone = np.zeros(sr // 10, dtype=np.float32)
        sf.write(str(path), tone, sr)
        progress_cb(1.0)
        return {
            "output_path": path,
            "edited_path": None,
            "session_dir": tmp_path / "work",
            "engine": engine,
            "voice": voice,
            "model": model,
            "duration_s": 0.1,
            "sample_rate": sr,
            "size_bytes": int(path.stat().st_size),
            "created_at": datetime.now(timezone.utc),
            "job_id": job_id,
        }

    monkeypatch.setattr(job_service, "run_tts_job", fake_run_tts_job)

    resp = api_client.post(
        "/v1/tts/jobs",
        json={
            "text": "Bonjour",
            "engine": "chatterbox_finetune_fr",
            "direction": {"enabled": False},
        },
    )
    assert resp.status_code == 200
    job_id = resp.json()["job_id"]

    status = None
    for _ in range(20):
        status_resp = api_client.get(f"/v1/jobs/{job_id}")
        assert status_resp.status_code == 200
        status = status_resp.json()
        if status["status"] in {"done", "error"}:
            break
        time.sleep(0.05)

    assert status is not None
    assert status["status"] == "done"
    assert status["asset_id"]

    asset_id = status["asset_id"]
    meta_resp = api_client.get(f"/v1/assets/{asset_id}/meta")
    assert meta_resp.status_code == 200
    meta = meta_resp.json()
    assert meta["file_name"] == "fake.wav"

    asset_resp = api_client.get(f"/v1/assets/{asset_id}")
    assert asset_resp.status_code == 200
    assert asset_resp.headers["content-type"].startswith("audio/")


def test_job_accepts_bark_without_voice(api_client, monkeypatch, tmp_path):
    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    def fake_run_tts_job(
        *,
        job_id,
        text,
        engine,
        voice,
        model,
        language,
        direction_enabled,
        direction_marker,
        options,
        export,
        editing,
        progress_cb,
    ):
        progress_cb(1.0)
        path = output_dir / "fake_bark.wav"
        sr = 24000
        tone = np.zeros(sr // 10, dtype=np.float32)
        sf.write(str(path), tone, sr)
        return {
            "output_path": path,
            "edited_path": None,
            "session_dir": tmp_path / "work",
            "engine": engine,
            "voice": voice,
            "model": model,
            "duration_s": 0.1,
            "sample_rate": sr,
            "size_bytes": int(path.stat().st_size),
            "created_at": datetime.now(timezone.utc),
            "job_id": job_id,
        }

    monkeypatch.setattr(job_service, "run_tts_job", fake_run_tts_job)

    resp = api_client.post(
        "/v1/tts/jobs",
        json={
            "text": "Hello",
            "engine": "bark",
            "direction": {"enabled": False},
            "options": {"voice_preset": "v2/en_speaker_6"},
        },
    )
    assert resp.status_code == 200


def test_qwen3_gap_forwarded_to_pipeline(api_client, monkeypatch):
    captured = {}

    def fake_create_job(payload):
        captured["payload"] = payload
        return {"job_id": "job_test", "status": "queued"}

    monkeypatch.setattr(job_service.JOB_STORE, "create_job", fake_create_job)

    resp = api_client.post(
        "/v1/tts/jobs",
        json={
            "text": "Bonjour",
            "engine": "qwen3_custom",
            "post_params": {"chunk_gap_ms": 250},
            "direction": {"enabled": True},
        },
    )
    assert resp.status_code == 200
    assert captured["payload"]["engine"] == "qwen3_custom"
    assert captured["payload"]["options"]["inter_chunk_gap_ms"] == 250
