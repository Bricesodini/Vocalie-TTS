from pathlib import Path

import numpy as np
import soundfile as sf

import app
import tts_pipeline
from text_tools import ChunkInfo, SpeechSegment


class DummyBackend:
    id = "dummy"

    def is_available(self):
        return True

    def unavailable_reason(self):
        return None

    def synthesize_chunk(self, text, *, voice_ref_path=None, lang=None, **params):
        sr = params.get("sr", 24000)
        length = params.get("length", sr)
        audio = np.ones(int(length), dtype=np.float32)
        return audio, sr, {}


def _make_chunks(count: int):
    chunks = []
    for idx in range(count):
        chunks.append(
            ChunkInfo(
                segments=[SpeechSegment("text", f"Chunk {idx}")],
                sentence_count=1,
                char_count=6,
                word_count=1,
                comma_count=0,
                estimated_duration=0.5,
                reason="manual_marker",
                boundary_kind="manual_marker",
                pivot=False,
                ends_with_suspended=False,
                oversize_sentence=False,
                warnings=[],
            )
        )
    return chunks


def _run_pipeline(monkeypatch, tmp_path: Path, *, engine_id: str, chunks: int, gap_ms: int):
    backend = DummyBackend()
    monkeypatch.setattr(tts_pipeline, "get_backend", lambda _backend_id: backend)
    out_path = tmp_path / f"{engine_id}_out.wav"
    request = {
        "tts_backend": engine_id,
        "script": "Hello",
        "chunks": _make_chunks(chunks),
        "out_path": str(out_path),
        "engine_params": {"length": 24000, "sr": 24000},
        "target_sr": 24000,
        "inter_chunk_gap_ms": gap_ms,
    }
    result = tts_pipeline.run_tts_pipeline(request)
    audio, sr = sf.read(result.out_path, dtype="float32")
    return audio, sr


def test_inter_chunk_gap_applies_for_chatterbox(monkeypatch, tmp_path):
    audio, sr = _run_pipeline(monkeypatch, tmp_path, engine_id="chatterbox", chunks=3, gap_ms=200)
    assert sr == 24000
    expected = (3 * sr) + int(0.4 * sr)
    assert abs(len(audio) - expected) < 100


def test_inter_chunk_gap_single_chunk_no_change(monkeypatch, tmp_path):
    audio, sr = _run_pipeline(monkeypatch, tmp_path, engine_id="chatterbox", chunks=1, gap_ms=200)
    assert sr == 24000
    assert abs(len(audio) - sr) < 10


def test_inter_chunk_gap_forced_zero_non_chatterbox(monkeypatch, tmp_path):
    audio, sr = _run_pipeline(monkeypatch, tmp_path, engine_id="xtts", chunks=3, gap_ms=200)
    assert sr == 24000
    assert abs(len(audio) - (3 * sr)) < 10


def test_inter_chunk_gap_slider_visibility():
    updates = app.handle_engine_change("xtts", "fr-FR", "fr_finetune")
    tail_index = len(app.all_param_keys()) + 2
    inter_chunk_update = updates[-tail_index]
    assert inter_chunk_update["visible"] is False
    assert inter_chunk_update["value"] == 0

    updates = app.handle_engine_change("chatterbox", "fr-FR", "fr_finetune")
    inter_chunk_update = updates[-tail_index]
    assert inter_chunk_update["visible"] is True
