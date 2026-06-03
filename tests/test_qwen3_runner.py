"""Tests for Qwen3 runner — now via BaseSubprocessRunner."""

from __future__ import annotations

import numpy as np

from tts_backends.base_runner import BaseSubprocessRunner
from tts_backends import qwen3_runner as qr


def test_coerce_bool_basic():
    from tts_backends.base import coerce_bool
    assert coerce_bool("1", False) is True
    assert coerce_bool("0", True) is False
    assert coerce_bool(None, True) is True
    assert coerce_bool(None, False) is False
    assert coerce_bool(True, False) is True
    assert coerce_bool(False, True) is False
    assert coerce_bool(1, False) is True
    assert coerce_bool(0, True) is False
    assert coerce_bool("yes", False) is True
    assert coerce_bool("no", True) is False
    assert coerce_bool("maybe", True) is True  # unknown → default


def test_resolve_dtype_none():
    assert BaseSubprocessRunner.resolve_dtype(None, None) is None
    assert BaseSubprocessRunner.resolve_dtype(None, "fp16") is None


def test_write_error():
    import json
    import io
    import sys
    captured = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = captured
    try:
        BaseSubprocessRunner.write_error("test_error", detail="test_detail")
    finally:
        sys.stdout = old_stdout
    output = captured.getvalue()
    data = json.loads(output)
    assert data["ok"] is False
    assert data["error"] == "test_error"
    assert data["detail"] == "test_detail"


def test_compute_audio_metrics_basic():
    sr = 24000
    t = np.linspace(0, 1.0, sr, endpoint=False, dtype=np.float32)
    audio = 0.2 * np.sin(2 * np.pi * 440 * t)

    # Audio metrics computation is now directly in this test
    # (the runner no longer includes _compute_audio_metrics since
    #  BaseSubprocessRunner handles the protocol, not the analysis)
    duration_s = float(len(audio) / sr)
    rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
    peak = float(np.max(np.abs(audio)))
    assert duration_s == 1.0
    assert rms > 0.1
    assert peak > 0.19


def test_qwen3_runner_is_base_subprocess_runner():
    assert issubclass(qr.Qwen3Runner, BaseSubprocessRunner)


def test_qwen3_runner_coerce_bool_delegates():
    """The Qwen3Runner.coerce_bool should delegate to base.coerce_bool."""
    assert qr.Qwen3Runner.coerce_bool(None, True) is True
    assert qr.Qwen3Runner.coerce_bool("0", True) is False