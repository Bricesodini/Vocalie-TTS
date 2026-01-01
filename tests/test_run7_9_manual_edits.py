import pytest

pytest.skip("Manual edit workflow removed in V2.", allow_module_level=True)

import hashlib
import json
import queue
from pathlib import Path

import numpy as np
import soundfile as sf

import app
from tts_backends.chatterbox_backend import ChatterboxBackend


def _setup_session(monkeypatch, tmp_path, *, text: str):
    work_dir = tmp_path / "work"
    tmp_dir = work_dir / ".tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    def fake_worker(payload, result_queue):
        out_path = Path(payload["out_path"])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        audio = np.zeros(2400, dtype=np.float32)
        sf.write(out_path, audio, 24000)
        chunk_count = len(payload.get("chunks") or [])
        meta = {
            "chunks": chunk_count,
            "durations": [0.1 for _ in range(chunk_count)],
            "retries": [False for _ in range(chunk_count)],
            "boundary_kinds": [None for _ in range(chunk_count)],
            "boundary_pauses": [0 for _ in range(chunk_count)],
            "punct_fixes": [None for _ in range(chunk_count)],
            "pause_events": [[] for _ in range(chunk_count)],
            "total_duration": 0.1,
        }
        result_queue.put({"status": "ok", "meta": meta})

    class DummyProcess:
        def __init__(self, target, args):
            self._target = target
            self._args = args
            self._alive = False

        def start(self):
            self._alive = True
            self._target(*self._args)
            self._alive = False

        def is_alive(self):
            return self._alive

        def join(self, timeout=None):
            return None

    class DummyContext:
        def Queue(self):
            return queue.Queue()

        def Process(self, target, args):
            return DummyProcess(target, args)

    monkeypatch.setattr(app, "WORK_DIR", work_dir)
    monkeypatch.setattr(app, "TMP_DIR", tmp_dir)
    monkeypatch.setattr(app, "_generate_longform_worker", fake_worker)
    monkeypatch.setattr(app.mp, "get_context", lambda *_args, **_kwargs: DummyContext())
    monkeypatch.setattr(ChatterboxBackend, "is_available", classmethod(lambda cls: True))

    chunk_state = {"applied": False, "chunks": [], "signature": None}
    args = [
        text,
        text,
        False,
        None,
        str(tmp_path),
        "run79",
        False,
        False,
        True,
        True,
        "punctuation",
        True,
        True,
        "chatterbox",
        "fr-FR",
        200,
        350,
        300,
        300,
        250,
        300,
        1,
        10,
        10.0,
        False,
        False,
        50,
        10,
        0.002,
        20,
        chunk_state,
        None,
    ]
    return app.handle_generate(*args, *([None] * len(app.all_param_keys())))


def _write_raw(tmp_path: Path, samples: int = 24000) -> Path:
    raw_path = tmp_path / "raw.wav"
    sf.write(raw_path, np.zeros(samples, dtype=np.float32), 24000)
    return raw_path


def test_insert_silence_increases_duration(tmp_path):
    raw_path = _write_raw(tmp_path)
    out_path = tmp_path / "out.wav"
    render_edits(
        raw_path,
        [{"type": "INSERT_SILENCE", "at_s": 0.0, "duration_ms": 500, "snap": "off"}],
        out_path,
        fade_ms=0,
        zero_cross_radius_ms=10,
        silence_threshold=0.0,
        silence_min_ms=0,
    )
    raw_audio, _ = sf.read(raw_path)
    out_audio, _ = sf.read(out_path)
    assert len(out_audio) > len(raw_audio)


def test_cut_decreases_duration(tmp_path):
    raw_path = _write_raw(tmp_path)
    out_path = tmp_path / "out.wav"
    render_edits(
        raw_path,
        [{"type": "CUT", "start_s": 0.2, "end_s": 0.4, "snap": "off"}],
        out_path,
        fade_ms=0,
        zero_cross_radius_ms=10,
        silence_threshold=0.0,
        silence_min_ms=0,
    )
    raw_audio, _ = sf.read(raw_path)
    out_audio, _ = sf.read(out_path)
    assert len(out_audio) < len(raw_audio)


def test_mute_keeps_duration(tmp_path):
    raw_path = _write_raw(tmp_path)
    out_path = tmp_path / "out.wav"
    render_edits(
        raw_path,
        [{"type": "MUTE", "start_s": 0.2, "end_s": 0.4, "snap": "off"}],
        out_path,
        fade_ms=0,
        zero_cross_radius_ms=10,
        silence_threshold=0.0,
        silence_min_ms=0,
    )
    raw_audio, _ = sf.read(raw_path)
    out_audio, _ = sf.read(out_path)
    assert len(out_audio) == len(raw_audio)


def test_raw_hash_unchanged(tmp_path):
    raw_path = _write_raw(tmp_path)
    out_path = tmp_path / "out.wav"
    before = hashlib.sha1(raw_path.read_bytes()).hexdigest()
    render_edits(
        raw_path,
        [{"type": "INSERT_SILENCE", "at_s": 0.0, "duration_ms": 200, "snap": "off"}],
        out_path,
        fade_ms=0,
        zero_cross_radius_ms=10,
        silence_threshold=0.0,
        silence_min_ms=0,
    )
    after = hashlib.sha1(raw_path.read_bytes()).hexdigest()
    assert before == after


def test_toggle_off_no_processed_preview(monkeypatch, tmp_path):
    result = _setup_session(monkeypatch, tmp_path, text="Bonjour.")
    session_dir = Path(result[-1]["dir"])
    preview_path = session_dir / "preview" / "processed_preview.wav"
    assert not preview_path.exists()
    app.handle_render_processed_preview(
        result[-1],
        False,
        "raw",
        [],
        50,
        10,
        0.002,
        20,
        None,
    )
    assert not preview_path.exists()


def test_toggle_on_regenerates_preview(monkeypatch, tmp_path):
    result = _setup_session(monkeypatch, tmp_path, text="Bonjour.")
    session_dir = Path(result[-1]["dir"])
    edits = [{"type": "INSERT_SILENCE", "at_s": 0.0, "duration_ms": 200, "snap": "off"}]
    app.handle_render_processed_preview(
        result[-1],
        True,
        "processed",
        edits,
        50,
        10,
        0.002,
        20,
        None,
    )
    preview_path = session_dir / "preview" / "processed_preview.wav"
    first_audio, _ = sf.read(preview_path)
    edits[0]["duration_ms"] = 400
    app.handle_render_processed_preview(
        result[-1],
        True,
        "processed",
        edits,
        50,
        10,
        0.002,
        20,
        None,
    )
    second_audio, _ = sf.read(preview_path)
    assert len(second_audio) > len(first_audio)


def test_export_writes_version_and_edits(monkeypatch, tmp_path):
    result = _setup_session(monkeypatch, tmp_path, text="Bonjour.")
    session_dir = Path(result[-1]["dir"])
    edits = [{"type": "INSERT_SILENCE", "at_s": 0.0, "duration_ms": 200, "snap": "off"}]
    app.handle_render_processed_preview(
        result[-1],
        True,
        "processed",
        edits,
        50,
        10,
        0.002,
        20,
        None,
    )
    app.handle_apply_processed_version(
        result[-1],
        True,
        "processed",
        edits,
        50,
        10,
        0.002,
        20,
        None,
    )
    processed_v2 = session_dir / "takes" / "processed" / "processed_global_v2.wav"
    edits_v2 = session_dir / "meta" / "edits_v2.json"
    meta_v2 = session_dir / "meta" / "processed_global_v2.json"
    assert processed_v2.exists()
    assert edits_v2.exists()
    assert meta_v2.exists()
    session_data = json.loads((session_dir / "session.json").read_text(encoding="utf-8"))
    assert session_data.get("active_take", {}).get("processed") == "v2"
import pytest

pytest.skip("Manual edit workflow removed in V2.", allow_module_level=True)
