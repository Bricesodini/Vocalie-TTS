import numpy as np

from tts_backends import qwen3_runner as qr


def test_compute_audio_metrics_basic():
    sr = 24000
    t = np.linspace(0, 1.0, sr, endpoint=False, dtype=np.float32)
    audio = 0.2 * np.sin(2 * np.pi * 440 * t)

    metrics = qr._compute_audio_metrics(audio, sr)
    assert metrics["duration_s"] == 1.0
    assert metrics["rms"] > 0.1
    assert metrics["peak"] > 0.19
    assert metrics["silence_ratio"] < 0.2


def test_looks_suspect_output_for_short_and_low_rms():
    metrics_short = {
        "duration_s": 0.7,
        "rms": 0.1,
        "peak": 0.5,
        "silence_ratio": 0.1,
        "clipped_ratio": 0.0,
    }
    assert qr._looks_suspect_output("x" * 120, metrics_short) is True

    metrics_low_rms = {
        "duration_s": 3.0,
        "rms": 0.001,
        "peak": 0.03,
        "silence_ratio": 0.9,
        "clipped_ratio": 0.0,
    }
    assert qr._looks_suspect_output("Bonjour test", metrics_low_rms) is True


def test_candidate_score_prefers_non_suspect():
    suspect = {
        "duration_s": 0.8,
        "rms": 0.002,
        "peak": 0.03,
        "silence_ratio": 0.9,
        "clipped_ratio": 0.0,
    }
    healthy = {
        "duration_s": 2.2,
        "rms": 0.08,
        "peak": 0.6,
        "silence_ratio": 0.2,
        "clipped_ratio": 0.0,
    }
    assert qr._candidate_score("x" * 80, healthy) > qr._candidate_score("x" * 80, suspect)
