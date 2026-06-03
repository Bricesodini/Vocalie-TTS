"""Tests for Chatterbox backend subprocess runner via SubprocessBackendMixin."""

import json

from tts_backends import chatterbox_backend as cb
from tts_backends.base_runner import SubprocessBackendMixin


def test_chatterbox_backend_uses_mixin():
    assert issubclass(cb.ChatterboxBackend, SubprocessBackendMixin)


def test_chatterbox_runner_returns_ok_json(monkeypatch, tmp_path):
    py_path = tmp_path / "python"
    runner_path = tmp_path / "chatterbox_runner.py"
    py_path.write_text("", encoding="utf-8")
    runner_path.write_text("", encoding="utf-8")

    backend = cb.ChatterboxBackend()

    # Monkey-patch the mixin methods
    monkeypatch.setattr(backend, "_python_path", lambda: py_path)
    monkeypatch.setattr(backend, "_runner_path", lambda: runner_path)

    class DummyProc:
        returncode = 0
        stdout = json.dumps(
            {
                "ok": True,
                "out_path": "/tmp/out.wav",
                "duration_s": 1.23,
                "logs": ["ok"],
            }
        )
        stderr = ""

    import subprocess
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: DummyProc())

    result = backend._run_subprocess({"text": "hello", "out_wav_path": "/tmp/out.wav"})
    assert result["ok"] is True
    assert result["out_path"] == "/tmp/out.wav"


def test_chatterbox_runner_variants():
    variants = cb.ChatterboxBackend.engine_variants()
    ids = [v["id"] for v in variants]
    assert "chatterbox_native" in ids
    assert "chatterbox_finetune_fr" in ids


def test_chatterbox_runner_models():
    models = cb.ChatterboxBackend().list_models()
    assert len(models) >= 2
    ids = [m.id for m in models]
    assert "ResembleAI/chatterbox" in ids