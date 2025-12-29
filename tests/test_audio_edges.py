import numpy as np

from tts_engine import (
    _find_zero_crossing_near,
    _apply_silence_edge_fades,
    _start_at_zero_and_fade_in,
    _trim_to_zero_and_fade_out,
)


def test_find_zero_crossing_near_picks_min_abs_sample():
    audio = np.array([0.5, -0.2, 0.0, 0.3], dtype=np.float32)
    idx = _find_zero_crossing_near(audio, center_idx=3, radius=3)
    assert idx == 2


def test_trim_to_zero_and_fade_out_reduces_length_and_fades():
    sr = 1000
    audio = np.concatenate(
        [np.ones(50, dtype=np.float32), np.zeros(5, dtype=np.float32)]
    )
    trimmed = _trim_to_zero_and_fade_out(audio, sr, radius_ms=10, fade_ms=5)
    assert len(trimmed) <= len(audio)
    assert abs(trimmed[-1]) < 1e-6


def test_start_at_zero_and_fade_in_trim_and_fades():
    sr = 1000
    audio = np.concatenate(
        [np.ones(5, dtype=np.float32), np.zeros(4, dtype=np.float32), np.ones(10, dtype=np.float32)]
    )
    started = _start_at_zero_and_fade_in(audio, sr, radius_ms=10, fade_ms=5)
    assert len(started) <= len(audio)
    assert abs(started[0]) < 1e-6


def test_apply_silence_edge_fades_softens_edges():
    sr = 1000
    audio = np.concatenate(
        [
            np.ones(20, dtype=np.float32),
            np.zeros(20, dtype=np.float32),
            np.ones(20, dtype=np.float32),
        ]
    )
    processed = _apply_silence_edge_fades(
        audio,
        sr,
        silence_threshold=0.001,
        silence_min_ms=10,
        fade_ms=10,
    )
    assert processed[19] < audio[19]
    assert processed[40] < audio[40]
