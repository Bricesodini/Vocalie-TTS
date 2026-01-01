from pathlib import Path
import datetime as dt

import numpy as np
import soundfile as sf

from session_manager import create_session_dir, get_processed_preview_path, get_take_path_global_raw, write_processed_meta
from tts_pipeline import minimal_post_process


def test_processed_preview_and_meta_written_in_session(tmp_path: Path):
    session_dir = create_session_dir(tmp_path, dt.datetime.now(), "minimal")
    raw_path = get_take_path_global_raw(session_dir, "v1")
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    sr = 24000
    audio = np.zeros(int(0.2 * sr), dtype=np.float32)
    audio[int(0.05 * sr):int(0.15 * sr)] = 0.1
    sf.write(raw_path, audio, sr)

    processed_preview = get_processed_preview_path(session_dir)
    processing_meta = minimal_post_process(raw_path, processed_preview)
    meta_path = write_processed_meta(
        session_dir,
        engine_id="chatterbox",
        engine_slug="chatterbox",
        source_take=raw_path.name,
        output_take=processed_preview.name,
        created_at=dt.datetime.now().isoformat(timespec="seconds"),
        processing_meta=processing_meta,
    )

    assert processed_preview.exists()
    assert meta_path.exists()
    assert session_dir in processed_preview.parents
    assert session_dir in meta_path.parents
    meta_payload = meta_path.read_text(encoding="utf-8")
    assert "\"mode\": \"minimal\"" in meta_payload
