from pathlib import Path

import numpy as np
import soundfile as sf

from tts_pipeline import minimal_post_process


def test_minimal_post_process_writes_and_preserves_raw(tmp_path: Path):
    sr = 24000
    tone = np.sin(np.linspace(0, 2 * np.pi * 440, sr)).astype(np.float32) * 0.2
    audio = np.concatenate([np.zeros(int(0.1 * sr), dtype=np.float32), tone, np.zeros(int(0.1 * sr))])
    raw_path = tmp_path / "raw.wav"
    processed_path = tmp_path / "processed.wav"
    sf.write(raw_path, audio, sr)
    raw_bytes = raw_path.read_bytes()

    meta = minimal_post_process(raw_path, processed_path)

    assert raw_path.read_bytes() == raw_bytes
    assert processed_path.exists()
    assert meta["trim"]["start_sample"] >= 0
    assert meta["trim"]["end_sample"] > meta["trim"]["start_sample"]
