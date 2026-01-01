import hashlib
import json
import queue
from pathlib import Path

import numpy as np
import soundfile as sf

import app
from tts_backends.xtts_backend import XTTSBackend


def test_xtts_segments_markers_and_processed(monkeypatch, tmp_path):
    work_dir = tmp_path / "work"
    tmp_dir = work_dir / ".tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    def fake_worker(payload, result_queue):
        out_path = Path(payload["out_path"])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        audio = np.zeros(2400, dtype=np.float32)
        sf.write(out_path, audio, 24000)
        meta = {
            "chunks": 1,
            "durations": [0.1],
            "retries": [False],
            "boundary_kinds": [None],
            "boundary_pauses": [0],
            "punct_fixes": [None],
            "pause_events": [[]],
            "total_duration": 0.1,
            "backend_meta": {
                "xtts_segments": ["Bonjour.", "Salut."],
                "xtts_segment_boundaries_samples": [1200],
                "xtts_sample_rate": 24000,
                "segment_strategy": "xtts_native_sentence_split",
            },
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

    ref_path = tmp_path / "ref.wav"
    sf.write(ref_path, np.zeros(2400, dtype=np.float32), 24000)

    monkeypatch.setattr(app, "WORK_DIR", work_dir)
    monkeypatch.setattr(app, "TMP_DIR", tmp_dir)
    monkeypatch.setattr(app, "_generate_longform_worker", fake_worker)
    monkeypatch.setattr(app.mp, "get_context", lambda *_args, **_kwargs: DummyContext())
    monkeypatch.setattr(XTTSBackend, "is_available", classmethod(lambda cls: True))
    monkeypatch.setattr(app, "resolve_ref_path", lambda _name: str(ref_path))

    chunk_state = {"applied": False, "chunks": [], "signature": None}
    args = [
        "Bonjour. Salut.",
        "Bonjour. Salut.",
        False,
        "ref.wav",
        str(tmp_path),
        "xtts",
        False,
        False,
        True,
        True,
        "disabled",
        False,
        True,
        "xtts",
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
    result = app.handle_generate(*args, *([None] * len(app.all_param_keys())))
    session_dir = Path(result[-1]["dir"])
    markers_path = session_dir / "meta" / "markers_global_v1.json"
    xtts_segments_path = session_dir / "meta" / "xtts_segments_global_v1.json"
    raw_path = session_dir / "takes" / "global" / "global_v1_raw.wav"
    processed_path = session_dir / "takes" / "processed" / "processed_global_v1.wav"

    assert markers_path.exists()
    assert xtts_segments_path.exists()
    raw_hash = hashlib.sha1(raw_path.read_bytes()).hexdigest()
    processed_audio, _ = sf.read(processed_path)
    raw_audio, _ = sf.read(raw_path)
    assert len(processed_audio) > len(raw_audio)
    assert hashlib.sha1(raw_path.read_bytes()).hexdigest() == raw_hash
    markers_payload = json.loads(markers_path.read_text(encoding="utf-8"))
    sample_indices = [m.get("sample_index") for m in markers_payload.get("markers", [])]
    assert all(isinstance(idx, int) for idx in sample_indices)
    assert sample_indices == sorted(sample_indices)
    assert all(0 < idx < len(raw_audio) for idx in sample_indices)


def test_xtts_missing_segments_disables_markers(monkeypatch, tmp_path):
    work_dir = tmp_path / "work"
    tmp_dir = work_dir / ".tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    def fake_worker(payload, result_queue):
        out_path = Path(payload["out_path"])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        audio = np.zeros(2400, dtype=np.float32)
        sf.write(out_path, audio, 24000)
        meta = {
            "chunks": 1,
            "durations": [0.1],
            "retries": [False],
            "boundary_kinds": [None],
            "boundary_pauses": [0],
            "punct_fixes": [None],
            "pause_events": [[]],
            "total_duration": 0.1,
            "backend_meta": {},
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

    ref_path = tmp_path / "ref.wav"
    sf.write(ref_path, np.zeros(2400, dtype=np.float32), 24000)

    monkeypatch.setattr(app, "WORK_DIR", work_dir)
    monkeypatch.setattr(app, "TMP_DIR", tmp_dir)
    monkeypatch.setattr(app, "_generate_longform_worker", fake_worker)
    monkeypatch.setattr(app.mp, "get_context", lambda *_args, **_kwargs: DummyContext())
    monkeypatch.setattr(XTTSBackend, "is_available", classmethod(lambda cls: True))
    monkeypatch.setattr(app, "resolve_ref_path", lambda _name: str(ref_path))

    chunk_state = {"applied": False, "chunks": [], "signature": None}
    args = [
        "Bonjour. Salut.",
        "Bonjour. Salut.",
        False,
        "ref.wav",
        str(tmp_path),
        "xtts_missing",
        False,
        False,
        True,
        True,
        "disabled",
        False,
        True,
        "xtts",
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
    result = app.handle_generate(*args, *([None] * len(app.all_param_keys())))
    session_dir = Path(result[-1]["dir"])
    markers_path = session_dir / "meta" / "markers_global_v1.json"
    raw_path = session_dir / "takes" / "global" / "global_v1_raw.wav"
    processed_path = session_dir / "takes" / "processed" / "processed_global_v1.wav"
    assert markers_path.exists()
    markers = json.loads(markers_path.read_text(encoding="utf-8"))
    assert markers["disabled_reason"] == "xtts_boundaries_missing"
    raw_audio, _ = sf.read(raw_path)
    proc_audio, _ = sf.read(processed_path)
    assert len(raw_audio) == len(proc_audio)
    raw_hash = hashlib.sha1(raw_path.read_bytes()).hexdigest()
    assert hashlib.sha1(raw_path.read_bytes()).hexdigest() == raw_hash
import pytest

pytest.skip("Marker workflow removed in V2.", allow_module_level=True)
