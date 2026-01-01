from pathlib import Path
import hashlib
import queue

import soundfile as sf

import app
from tts_backends.chatterbox_backend import ChatterboxBackend


def _dummy_context(monkeypatch, work_dir: Path, tmp_dir: Path):
    def fake_worker(payload, result_queue):
        out_path = Path(payload["out_path"])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(out_path, [0.0, 0.0, 0.0], 24000)
        result_queue.put({"status": "ok", "meta": {"chunks": len(payload.get("chunks") or [])}})

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


def _run_generate(monkeypatch, tmp_path: Path):
    work_dir = tmp_path / "work"
    tmp_dir = work_dir / ".tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    _dummy_context(monkeypatch, work_dir, tmp_dir)
    out_dir = tmp_path / "output"
    args = [
        "Bonjour",
        "Bonjour",
        False,
        None,
        str(out_dir),
        "test",
        False,
        False,
        "chatterbox",
        "fr-FR",
        120,
        False,
        False,
        "final",
        "",
        "",
        None,
    ]
    result = app.handle_generate(*args, *([None] * len(app.all_param_keys())))
    return result, out_dir


def test_generate_creates_no_edited_file(monkeypatch, tmp_path: Path):
    _result, out_dir = _run_generate(monkeypatch, tmp_path)
    assert list(out_dir.glob("*.wav")) == []


def test_generate_edited_audio_creates_file(monkeypatch, tmp_path: Path):
    result, out_dir = _run_generate(monkeypatch, tmp_path)
    session_state = result[-1]
    edited_audio, edited_path, _log = app.handle_generate_edited_audio(
        session_state,
        str(out_dir),
        "test",
        False,
        False,
        True,
        True,
        -1.0,
        None,
    )
    assert edited_audio == edited_path
    assert Path(edited_path).exists()
    assert Path(edited_path).parent == out_dir


def test_generate_edited_audio_collision_adds_suffix(monkeypatch, tmp_path: Path):
    result, out_dir = _run_generate(monkeypatch, tmp_path)
    session_state = result[-1]
    first_audio, first_path, _log = app.handle_generate_edited_audio(
        session_state,
        str(out_dir),
        "test",
        False,
        False,
        True,
        True,
        -1.0,
        None,
    )
    second_audio, second_path, _log = app.handle_generate_edited_audio(
        session_state,
        str(out_dir),
        "test",
        False,
        False,
        True,
        True,
        -1.0,
        None,
    )
    assert Path(first_path).exists()
    assert Path(second_path).exists()
    assert first_audio == first_path
    assert second_audio == second_path
    assert Path(first_path).name.endswith("_edit.wav")
    assert Path(second_path).name.endswith("_edit_01.wav")


def test_edit_does_not_change_raw(monkeypatch, tmp_path: Path):
    result, out_dir = _run_generate(monkeypatch, tmp_path)
    raw_path = Path(result[1])
    before = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    app.handle_generate_edited_audio(
        result[-1],
        str(out_dir),
        "test",
        False,
        False,
        True,
        True,
        -1.0,
        None,
    )
    after = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    assert before == after
