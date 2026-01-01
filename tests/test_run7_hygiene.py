import hashlib
import queue
from pathlib import Path

import numpy as np
import soundfile as sf

import app
from tts_backends.chatterbox_backend import ChatterboxBackend


def _setup_session(monkeypatch, tmp_path, *, text: str, output_dir: Path, post_processing: bool):
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
        str(output_dir),
        "run7h",
        False,
        False,
        True,
        True,
        "punctuation",
        True,
        post_processing,
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


def test_no_output_pollution_on_generate(monkeypatch, tmp_path):
    output_dir = tmp_path / "output"
    _setup_session(
        monkeypatch,
        tmp_path,
        text="Bonjour, salut.",
        output_dir=output_dir,
        post_processing=True,
    )
    wavs = []
    for path in output_dir.rglob("*.wav"):
        if ".sessions" in path.parts:
            continue
        wavs.append(path)
    assert wavs == []


def test_exit_edit_mode_reverts_to_active_processed_take(monkeypatch, tmp_path):
    output_dir = tmp_path / "output"
    result = _setup_session(
        monkeypatch,
        tmp_path,
        text="Bonjour, salut.",
        output_dir=output_dir,
        post_processing=True,
    )
    session_dir = Path(result[-1]["dir"])
    raw_path = session_dir / "takes" / "global" / "global_v1_raw.wav"
    raw_hash = hashlib.sha1(raw_path.read_bytes()).hexdigest()
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
    preview_path, listen_update, _ = app.handle_listen_switch(
        result[-1], "processed", False, None
    )
    assert listen_update["value"] == "processed"
    current_path = Path(preview_path)
    assert current_path.exists()
    assert hashlib.sha1(current_path.read_bytes()).hexdigest() == hashlib.sha1(
        processed_v2.read_bytes()
    ).hexdigest()
    assert hashlib.sha1(raw_path.read_bytes()).hexdigest() == raw_hash


def test_listen_processed_fallback_to_raw(monkeypatch, tmp_path):
    output_dir = tmp_path / "output"
    result = _setup_session(
        monkeypatch,
        tmp_path,
        text="Bonjour.",
        output_dir=output_dir,
        post_processing=False,
    )
    session_dir = Path(result[-1]["dir"])
    raw_path = session_dir / "takes" / "global" / "global_v1_raw.wav"
    preview_path, listen_update, log_text = app.handle_listen_switch(
        result[-1], "processed", False, None
    )
    assert listen_update["value"] == "raw"
    assert preview_path == str(session_dir / "preview" / app.SESSION_PREVIEW_FILENAME)
    assert "Processed absent" in (log_text or "")
import pytest

pytest.skip("Edit hygiene checks removed in V2.", allow_module_level=True)
