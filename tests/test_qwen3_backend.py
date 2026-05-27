import subprocess
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from tts_backends.base import BackendUnavailableError
from tts_backends import qwen3_backend as qb


def test_ensure_wav_ref_normalizes_with_ffmpeg(monkeypatch, tmp_path):
    calls = []

    def fake_run(cmd, check, capture_output, text):
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


def test_ensure_wav_ref_fallback_without_loudnorm(monkeypatch, tmp_path):
    calls = []

    def fake_run(cmd, check, capture_output, text):
        calls.append(cmd)
        if len(calls) == 1:
            raise subprocess.CalledProcessError(1, cmd)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(qb.shutil, "which", lambda _: "/usr/bin/ffmpeg")
    monkeypatch.setattr(qb.subprocess, "run", fake_run)

    qb._ensure_wav_ref(str(tmp_path / "voice.m4a"), tmp_path)
    assert len(calls) == 2
    assert "-af" in calls[0]
    assert "-af" not in calls[1]


def _write_mono_wav(path: Path, audio: np.ndarray, sr: int = 24000) -> None:
    sf.write(str(path), audio.astype(np.float32), sr, subtype="PCM_16")


def test_validate_ref_audio_accepts_clean_reference(tmp_path):
    sr = 24000
    t = np.linspace(0, 2.0, int(sr * 2.0), endpoint=False, dtype=np.float32)
    audio = 0.1 * np.sin(2 * np.pi * 220 * t)
    wav = tmp_path / "ref_ok.wav"
    _write_mono_wav(wav, audio, sr)

    metrics = qb._validate_ref_audio(str(wav))
    assert metrics["duration_s"] >= 1.9
    assert metrics["rms"] > 0.01


def test_validate_ref_audio_rejects_too_short(tmp_path):
    sr = 24000
    audio = np.ones(int(sr * 0.5), dtype=np.float32) * 0.1
    wav = tmp_path / "ref_short.wav"
    _write_mono_wav(wav, audio, sr)

    with pytest.raises(BackendUnavailableError, match="trop court"):
        qb._validate_ref_audio(str(wav))


def test_validate_ref_audio_rejects_too_silent(tmp_path):
    sr = 24000
    audio = np.zeros(int(sr * 2.0), dtype=np.float32)
    wav = tmp_path / "ref_silent.wav"
    _write_mono_wav(wav, audio, sr)

    with pytest.raises(BackendUnavailableError):
        qb._validate_ref_audio(str(wav))
