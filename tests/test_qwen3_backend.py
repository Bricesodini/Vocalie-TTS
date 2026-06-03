import subprocess
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from tts_backends.base import BackendUnavailableError, coerce_bool
from tts_backends import qwen3_backend as qb


def test_ensure_wav_ref_passes_through_wav(monkeypatch, tmp_path):
    """_ensure_wav_ref returns the original path for .wav files."""
    wav = tmp_path / "voice.wav"
    wav.write_bytes(b"RIFF\x00\x00\x00\x00WAVEfmt ")
    out = qb._ensure_wav_ref(str(wav), tmp_path)
    assert out == str(wav)


def test_ensure_wav_ref_converts_with_ffmpeg(monkeypatch, tmp_path):
    """_ensure_wav_ref uses ffmpeg for non-wav files with proper flags."""
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(qb.shutil, "which", lambda _: "/usr/bin/ffmpeg")
    monkeypatch.setattr(qb.subprocess, "run", fake_run)

    out = qb._ensure_wav_ref(str(tmp_path / "voice.mp3"), tmp_path)
    assert out.endswith("qwen3_ref_voice.wav")
    assert calls
    cmd = calls[0]
    assert "-ac" in cmd and "1" in cmd
    assert "-ar" in cmd and "24000" in cmd
    assert "-sample_fmt" in cmd and "s16" in cmd
    assert "-af" in cmd
    assert "loudnorm" in cmd


def test_ensure_wav_ref_raises_without_ffmpeg(monkeypatch, tmp_path):
    """_ensure_wav_ref raises when ffmpeg is missing and file is not .wav."""
    monkeypatch.setattr(qb.shutil, "which", lambda _: None)
    with pytest.raises(BackendUnavailableError, match="ffmpeg"):
        qb._ensure_wav_ref(str(tmp_path / "voice.mp3"), tmp_path)


def test_qwen3_default_models():
    """QWEN3_DEFAULT_MODELS contains expected entries."""
    assert "custom_voice" in qb.QWEN3_DEFAULT_MODELS
    assert "voice_clone" in qb.QWEN3_DEFAULT_MODELS
    assert "voice_design" in qb.QWEN3_DEFAULT_MODELS
    assert qb.QWEN3_DEFAULT_MODELS["custom_voice"].startswith("Qwen/")


def test_qwen3_engine_variants():
    from tts_backends.qwen3_backend import Qwen3Backend
    variants = Qwen3Backend.engine_variants()
    assert len(variants) == 2
    ids = [v["id"] for v in variants]
    assert "qwen3_custom" in ids
    assert "qwen3_clone" in ids


def test_qwen3_list_models():
    from tts_backends import get_backend
    backend = get_backend("qwen3")
    models = backend.list_models()
    assert len(models) >= 3
    assert any(m.id == "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice" for m in models)


def test_qwen3_supports_ref_for_engine():
    from tts_backends import get_backend
    backend = get_backend("qwen3")
    assert backend.supports_ref_for_engine("qwen3_custom") is False
    assert backend.supports_ref_for_engine("qwen3_clone") is True


def test_qwen3_resolve_engine_params():
    from tts_backends import get_backend
    backend = get_backend("qwen3")
    params = backend.resolve_engine_params("qwen3_clone", {})
    assert params["qwen3_mode"] == "voice_clone"

    params = backend.resolve_engine_params("qwen3_custom", {})
    assert params["qwen3_mode"] == "custom_voice"

    # User-override takes precedence
    params = backend.resolve_engine_params("qwen3_custom", {"qwen3_mode": "voice_design"})
    assert params["qwen3_mode"] == "voice_design"