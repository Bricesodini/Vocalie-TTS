from pathlib import Path
import queue

import app
from backend_install.manifests import get_manifest
from tts_backends import get_backend
from tts_backends.xtts_backend import XTTSBackend, _extract_xtts_segments
from tts_backends import xtts_runner
import soundfile as sf


def test_xtts_manifest_exists():
    manifest = get_manifest("xtts")
    assert manifest is not None
    assert manifest.engine_id == "xtts"
    assert any(pkg.split("==")[0] == "TTS" for pkg in manifest.pip_packages)


def test_xtts_schema_exposed():
    backend = get_backend("xtts")
    assert backend is not None
    assert backend.supports_ref_audio is True
    assert backend.uses_internal_voices is False
    assert backend.supports_multilang is True
    schema = backend.params_schema()
    assert "speed" in schema
    assert schema["speed"].type == "float"


def test_xtts_default_language_prefers_fr():
    backend = get_backend("xtts")
    assert backend is not None
    assert backend.default_language() == "fr-FR"


def test_runner_command_build_xtts():
    cmd = XTTSBackend.build_command(
        py=Path("/tmp/py"),
        runner=Path("/tmp/runner"),
        text="bonjour",
        out_path="/tmp/out.wav",
        speaker_wav="/tmp/ref.wav",
        language="fr",
        model_id="model",
        speed=1.1,
        meta_json="/tmp/meta.json",
    )
    assert cmd[0] == "/tmp/py"
    assert "--speaker_wav" in cmd
    assert "--language" in cmd
    assert "--speed" in cmd


def test_xtts_runner_forces_cpu_on_macos_arm():
    assert xtts_runner.force_cpu_for_platform("Darwin", "arm64") is True
    assert xtts_runner.force_cpu_for_platform("Darwin", "aarch64") is True
    assert xtts_runner.force_cpu_for_platform("Linux", "x86_64") is False


def test_extract_xtts_segments_from_log():
    log_text = " > Using model: xtts\n > Text splitted to sentences.\n['Bonjour.', 'Salut.']\n"
    segments = _extract_xtts_segments(log_text)
    assert segments == ["Bonjour.", "Salut."]


def _dummy_context(monkeypatch, captured):
    def fake_worker(payload, result_queue):
        out_path = Path(payload["out_path"])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(out_path, [0.0, 0.0, 0.0], 24000)
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


def test_payload_filters_params_for_xtts(monkeypatch, tmp_path):
    captured = {}
    _dummy_context(monkeypatch, captured)
    monkeypatch.setattr(XTTSBackend, "is_available", classmethod(lambda cls: True))
    monkeypatch.setattr(app, "resolve_ref_path", lambda _name: str(tmp_path / "ref.wav"))

    args = [
        "Bonjour",
        "Bonjour",
        False,
        "ref.wav",
        str(tmp_path),
        "test",
        False,
        False,
        "xtts",
        "fr-FR",
        0,
        False,
        False,
        "final",
        "",
        "",
        None,
    ]
    param_values = [None] * len(app.all_param_keys())
    if "speed" in app.all_param_keys():
        param_values[app.all_param_keys().index("speed")] = 1.25
    app.handle_generate(*args, *param_values)
    engine_params = captured["payload"]["engine_params"]
    assert "speed" in engine_params
    assert "cfg_weight" not in engine_params
    assert "chatterbox_mode" not in engine_params
    assert "voice" not in engine_params
    assert captured["payload"]["tts_backend"] == "xtts"
