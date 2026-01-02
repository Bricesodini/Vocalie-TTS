from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import backend.app as backend_app
import backend.config as backend_config
import backend.services.asset_service as asset_service
import backend.services.job_service as job_service
import backend.services.preset_service as preset_service
import backend.services.tts_service as tts_service


collect_ignore = ["tts_test.py"]


@pytest.fixture
def api_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    work_dir = tmp_path / "work"
    output_dir = tmp_path / "output"
    presets_dir = tmp_path / "presets"
    assets_meta_dir = output_dir / ".assets"
    ref_dir = tmp_path / "Ref_audio"

    for path in (work_dir, output_dir, presets_dir, assets_meta_dir, ref_dir):
        path.mkdir(parents=True, exist_ok=True)
    (ref_dir / "voice.wav").write_bytes(b"RIFF0000WAVEfmt ")

    monkeypatch.setenv("CHATTERBOX_REF_DIR", str(ref_dir))

    monkeypatch.setattr(backend_config, "WORK_DIR", work_dir, raising=False)
    monkeypatch.setattr(backend_config, "OUTPUT_DIR", output_dir, raising=False)
    monkeypatch.setattr(backend_config, "PRESETS_DIR", presets_dir, raising=False)
    monkeypatch.setattr(backend_config, "ASSETS_META_DIR", assets_meta_dir, raising=False)

    monkeypatch.setattr(backend_app, "WORK_DIR", work_dir, raising=False)

    monkeypatch.setattr(tts_service, "WORK_DIR", work_dir, raising=False)
    monkeypatch.setattr(tts_service, "OUTPUT_DIR", output_dir, raising=False)

    monkeypatch.setattr(job_service, "OUTPUT_DIR", output_dir, raising=False)

    monkeypatch.setattr(asset_service, "OUTPUT_DIR", output_dir, raising=False)
    monkeypatch.setattr(asset_service, "ASSETS_META_DIR", assets_meta_dir, raising=False)

    monkeypatch.setattr(preset_service, "PRESETS_DIR", presets_dir, raising=False)

    return TestClient(backend_app.app)
