import hashlib
import json
import queue
from pathlib import Path

import numpy as np
import soundfile as sf

import app
from tts_backends.chatterbox_backend import ChatterboxBackend


def _run_generate(
    monkeypatch,
    tmp_path,
    *,
    text: str,
    post_processing_enabled: bool,
    markers_enabled: bool,
    pauses_enabled: bool = True,
    markers_strategy_non_xtts: str = "disabled",
    user_filename: str = "test",
):
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
        user_filename,
        False,
        False,
        pauses_enabled,
        markers_enabled,
        markers_strategy_non_xtts,
        True,
        post_processing_enabled,
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


def _hash_file(path: Path) -> str:
    return hashlib.sha1(path.read_bytes()).hexdigest()


def test_markers_off_no_file_and_no_duration_change(monkeypatch, tmp_path):
    result = _run_generate(
        monkeypatch,
        tmp_path,
        text="Bonjour.\nSalut.",
        post_processing_enabled=True,
        markers_enabled=False,
        pauses_enabled=True,
        user_filename="markers_off",
    )
    session_dir = Path(result[-1]["dir"])
    raw_path = session_dir / "takes" / "global" / "global_v1_raw.wav"
    processed_path = session_dir / "takes" / "processed" / "processed_global_v1.wav"
    markers_path = session_dir / "meta" / "markers_global_v1.json"
    assert processed_path.exists()
    assert not markers_path.exists()
    raw_audio, raw_sr = sf.read(raw_path)
    proc_audio, proc_sr = sf.read(processed_path)
    assert raw_sr == proc_sr
    assert len(raw_audio) == len(proc_audio)


def test_markers_on_no_punctuation_fallback(monkeypatch, tmp_path):
    result = _run_generate(
        monkeypatch,
        tmp_path,
        text="Bonjour.\nSalut.",
        post_processing_enabled=True,
        markers_enabled=True,
        pauses_enabled=True,
        user_filename="markers_on",
    )
    session_dir = Path(result[-1]["dir"])
    raw_path = session_dir / "takes" / "global" / "global_v1_raw.wav"
    processed_path = session_dir / "takes" / "processed" / "processed_global_v1.wav"
    markers_path = session_dir / "meta" / "markers_global_v1.json"
    assert markers_path.exists()
    markers = json.loads(markers_path.read_text(encoding="utf-8"))
    assert markers["disabled_reason"] == "disabled"
    raw_audio, _ = sf.read(raw_path)
    proc_audio, _ = sf.read(processed_path)
    assert len(proc_audio) == len(raw_audio)


def test_markers_on_punctuation_writes_meta(monkeypatch, tmp_path):
    result = _run_generate(
        monkeypatch,
        tmp_path,
        text="Bonjour, salut.",
        post_processing_enabled=True,
        markers_enabled=True,
        markers_strategy_non_xtts="punctuation",
        pauses_enabled=True,
        user_filename="markers_meta",
    )
    session_dir = Path(result[-1]["dir"])
    processed_meta_path = session_dir / "meta" / "processed_global_v1.json"
    assert processed_meta_path.exists()
    meta = json.loads(processed_meta_path.read_text(encoding="utf-8"))
    assert meta["processing"]["markers_enabled"] is True
    assert meta["processing"]["markers_strategy"] == "punctuation"
    assert meta["processing"]["markers_count"] > 0


def test_markers_raw_unchanged(monkeypatch, tmp_path):
    result = _run_generate(
        monkeypatch,
        tmp_path,
        text="Bonjour.\nSalut.",
        post_processing_enabled=True,
        markers_enabled=True,
        pauses_enabled=True,
        user_filename="markers_hash",
    )
    session_dir = Path(result[-1]["dir"])
    raw_path = session_dir / "takes" / "global" / "global_v1_raw.wav"
    before = _hash_file(raw_path)
    app.handle_listen_switch(result[-1], "processed", False, None)
    after = _hash_file(raw_path)
    assert before == after
import pytest

pytest.skip("Marker workflow removed in V2.", allow_module_level=True)
