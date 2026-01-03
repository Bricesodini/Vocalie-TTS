import numpy as np

from tts_pipeline import _fade_in, _fade_out, _find_active_range, _snap_zero_crossing


def test_snap_zero_crossing_picks_nearest():
    audio = np.array([0.5, -0.2, 0.0, 0.3], dtype=np.float32)
    idx = _snap_zero_crossing(audio, idx=3, radius_samples=3)
    assert idx == 3


def test_fade_out_softens_tail():
    audio = np.ones(10, dtype=np.float32)
    processed = _fade_out(audio.copy(), fade_frames=5)
    assert processed[-1] == 0.0


def test_fade_in_softens_head():
    audio = np.ones(10, dtype=np.float32)
    processed = _fade_in(audio.copy(), fade_frames=5)
    assert processed[0] == 0.0


def test_find_active_range_trims_edges():
    audio = np.array([0.0, 0.0, 0.01, 0.02, 0.0, 0.0], dtype=np.float32)
    start, end = _find_active_range(audio, threshold=0.005, min_silence_frames=0)
    assert (start, end) == (2, 4)
