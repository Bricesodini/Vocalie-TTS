from pathlib import Path
import queue

import pytest

import app
from tts_backends.chatterbox_backend import ChatterboxBackend
from text_tools import chunk_script as real_chunk_script


def test_handle_generate_uses_adjusted_text(monkeypatch, tmp_path):
    def fake_worker(payload, result_queue):
        out_path = Path(payload["out_path"])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(b"")
        chunk_count = len(payload.get("chunks") or [])
        meta = {
            "chunks": chunk_count,
            "durations": [0.5 for _ in range(chunk_count)],
            "retries": [False for _ in range(chunk_count)],
            "boundary_kinds": [None for _ in range(chunk_count)],
            "boundary_pauses": [0 for _ in range(chunk_count)],
            "punct_fixes": [None for _ in range(chunk_count)],
            "pause_events": [[] for _ in range(chunk_count)],
            "total_duration": 0.5,
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

    captured = {}

    def spy_chunk_script(script, **kwargs):
        captured["script"] = script
        return real_chunk_script(script, **kwargs)

    monkeypatch.setattr(app, "_generate_longform_worker", fake_worker)
    monkeypatch.setattr(app.mp, "get_context", lambda *_args, **_kwargs: DummyContext())
    monkeypatch.setattr(ChatterboxBackend, "is_available", classmethod(lambda cls: True))
    monkeypatch.setattr(app, "prepare_adjusted_text", lambda *_args, **_kwargs: ("Texte Ajuste", []))
    monkeypatch.setattr(app, "chunk_script", spy_chunk_script)

    chunk_state = {"applied": False, "chunks": [], "signature": None}
    args = [
        "RAW",
        "RAW",
        True,
        None,
        str(tmp_path),
        "test",
        False,
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
    app.handle_generate(*args, *([None] * len(app.all_param_keys())))
    assert captured.get("script") == "Texte Ajuste"
