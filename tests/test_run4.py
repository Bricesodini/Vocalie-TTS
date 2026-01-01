import hashlib
import queue
from pathlib import Path

import numpy as np
import soundfile as sf

import app
from tts_backends.chatterbox_backend import ChatterboxBackend


def _run_generate(monkeypatch, tmp_path, post_processing_enabled: bool, user_filename: str = "test"):
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
        "Bonjour",
        "Bonjour",
        False,
        None,
        str(tmp_path),
        user_filename,
        False,
        False,
        True,
        True,
        "disabled",
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
        2,
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


def test_run4_raw_written_in_session(monkeypatch, tmp_path):
    result = _run_generate(monkeypatch, tmp_path, post_processing_enabled=False, user_filename="raw")
    session_state = result[-1]
    session_dir = Path(session_state["dir"])
    raw_path = session_dir / "takes" / "global" / "global_v1_raw.wav"
    assert raw_path.exists()


def test_run4_processed_optional(monkeypatch, tmp_path):
    result = _run_generate(monkeypatch, tmp_path, post_processing_enabled=True, user_filename="proc")
    session_state = result[-1]
    session_dir = Path(session_state["dir"])
    processed_path = session_dir / "takes" / "processed" / "processed_global_v1.wav"
    processed_meta = session_dir / "meta" / "processed_global_v1.json"
    assert processed_path.exists()
    assert processed_meta.exists()

    result_off = _run_generate(monkeypatch, tmp_path, post_processing_enabled=False, user_filename="raw2")
    session_state_off = result_off[-1]
    session_dir_off = Path(session_state_off["dir"])
    processed_path_off = session_dir_off / "takes" / "processed" / "processed_global_v1.wav"
    assert not processed_path_off.exists()


def test_run4_ab_switch_fallback(monkeypatch, tmp_path):
    result = _run_generate(monkeypatch, tmp_path, post_processing_enabled=False, user_filename="ab")
    session_state = result[-1]
    session_preview = Path(session_state["dir"]) / "preview" / app.SESSION_PREVIEW_FILENAME
    audio_path, listen_update, _ = app.handle_listen_switch(
        session_state, "processed", False, None
    )
    assert audio_path == str(session_preview)
    assert listen_update["value"] == "raw"


def test_run4_raw_hash_unchanged(monkeypatch, tmp_path):
    result = _run_generate(monkeypatch, tmp_path, post_processing_enabled=True, user_filename="hash")
    session_state = result[-1]
    session_dir = Path(session_state["dir"])
    raw_path = session_dir / "takes" / "global" / "global_v1_raw.wav"
    before = _hash_file(raw_path)
    app.handle_listen_switch(session_state, "processed", False, None)
    after = _hash_file(raw_path)
    assert before == after
import pytest

pytest.skip("Legacy pause metadata removed in V2.", allow_module_level=True)
