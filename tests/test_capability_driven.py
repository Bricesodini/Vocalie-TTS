from pathlib import Path
import queue

import pytest

import app
from tts_backends import list_backends
from tts_backends.base import VoiceInfo, validate_param_schema
from tts_backends.piper_backend import PiperBackend
from tts_backends.chatterbox_backend import ChatterboxBackend
import state_manager


def test_param_schema_contract():
    for backend in list_backends():
        errors = validate_param_schema(backend.params_schema())
        assert errors == []


def test_engine_switch_coercion(monkeypatch):
    state = {
        "engines": {
            "chatterbox": {
                "language": "fr-FR",
                "params": {
                    "chatterbox_mode": "multilang",
                    "cfg_weight": "0.7",
                    "temperature": "0.4",
                    "repetition_penalty": "1.2",
                },
            }
        }
    }
    monkeypatch.setattr(app, "load_state", lambda: state)
    updates = app.handle_engine_change(
        "chatterbox",
        "fr-FR",
        "multilang",
        {"applied": True, "chunks": ["x"], "signature": ("sig",)},
    )
    param_updates = updates[-len(app.all_param_keys()):]
    param_keys = app.all_param_keys()
    for key in ("cfg_weight", "temperature", "repetition_penalty"):
        idx = param_keys.index(key)
        assert isinstance(param_updates[idx]["value"], float)


def _dummy_context(monkeypatch, captured):
    def fake_worker(payload, result_queue):
        out_path = Path(payload["out_path"])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(b"")
        captured["payload"] = payload
        result_queue.put({"status": "ok", "meta": {}})

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

    monkeypatch.setattr(app, "_generate_longform_worker", fake_worker)
    monkeypatch.setattr(app.mp, "get_context", lambda *_args, **_kwargs: DummyContext())


def test_payload_filtering(monkeypatch, tmp_path):
    captured = {}
    _dummy_context(monkeypatch, captured)
    monkeypatch.setattr(PiperBackend, "is_available", classmethod(lambda cls: True))
    monkeypatch.setattr(app, "piper_voice_supports_length_scale", lambda _voice_id: False)
    monkeypatch.setattr(
        PiperBackend,
        "list_voices",
        lambda self: [VoiceInfo(id="voice", label="voice")],
    )
    args = [
        "Bonjour",
        "Bonjour",
        False,
        None,
        str(tmp_path),
        "test",
        False,
        "piper",
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
        {"applied": True, "chunks": [], "signature": None},
        None,
    ]
    param_values = [None] * len(app.all_param_keys())
    if "voice_id" in app.all_param_keys():
        param_values[app.all_param_keys().index("voice_id")] = "voice"
    app.handle_generate(*args, *param_values)
    engine_params = captured["payload"]["engine_params"]
    assert "cfg_weight" not in engine_params
    assert "chatterbox_mode" not in engine_params
    assert "voice" in engine_params

    captured.clear()
    _dummy_context(monkeypatch, captured)
    monkeypatch.setattr(ChatterboxBackend, "is_available", classmethod(lambda cls: True))
    args = [
        "Bonjour",
        "Bonjour",
        False,
        None,
        str(tmp_path),
        "test2",
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
        {"applied": True, "chunks": [], "signature": None},
        None,
    ]
    param_values = [None] * len(app.all_param_keys())
    if "voice_id" in app.all_param_keys():
        param_values[app.all_param_keys().index("voice_id")] = "voice"
    app.handle_generate(*args, *param_values)
    engine_params = captured["payload"]["engine_params"]
    assert "voice" not in engine_params


def test_preset_migration_engines(monkeypatch, tmp_path):
    monkeypatch.setattr(state_manager, "PRESET_DIR", tmp_path)
    preset_path = tmp_path / "legacy.json"
    preset_path.write_text(
        '{"tts_engine":"chatterbox","tts_model_mode":"fr_finetune","tts_language":""}',
        encoding="utf-8",
    )
    data = state_manager.load_preset("legacy")
    assert "engines" in data
    assert data["engines"]["chatterbox"]["language"] == "fr-FR"
