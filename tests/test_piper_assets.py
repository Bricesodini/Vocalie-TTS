from pathlib import Path

import app
import tts_backends.piper_assets as piper_assets
from tts_backends.base import VoiceInfo


def test_piper_assets_install_writes_two_files(monkeypatch, tmp_path):
    monkeypatch.setattr(piper_assets, "get_piper_voices_dir", lambda: tmp_path)

    def fake_retrieve(url, filename):
        if str(filename).endswith(".json.tmp"):
            Path(filename).write_text("{}", encoding="utf-8")
        else:
            Path(filename).write_bytes(b"data")

    monkeypatch.setattr(piper_assets.urllib.request, "urlretrieve", fake_retrieve)
    result = piper_assets.install_voice_from_catalog(
        "fr_FR-upmc-medium",
        "http://example.com/voice.onnx",
        "http://example.com/voice.onnx.json",
    )
    assert result.ok is True
    assert (tmp_path / "fr_FR-upmc-medium.onnx").exists()
    assert (tmp_path / "fr_FR-upmc-medium.onnx.json").exists()


def test_piper_assets_list_empty_triggers_default_install_mocked(monkeypatch):
    monkeypatch.setattr(piper_assets, "is_voice_installed", lambda _voice_id: False)
    called = {}

    def fake_install(voice_id, onnx_url, json_url):
        called["voice_id"] = voice_id
        return piper_assets.InstallResult(True, voice_id, "OK")

    monkeypatch.setattr(piper_assets, "install_voice_from_catalog", fake_install)
    result = piper_assets.ensure_default_voice_installed()
    assert result.ok is True
    assert called.get("voice_id") == piper_assets.DEFAULT_VOICE_ID


def test_refresh_piper_voices_returns_choices(monkeypatch):
    monkeypatch.setattr(
        app,
        "list_piper_voices",
        lambda: [
            VoiceInfo(id="voice1", label="Voice 1", lang_codes=["fr-FR"], installed=True),
            VoiceInfo(id="voice2", label="Voice 2", lang_codes=["fr-FR"], installed=True),
        ],
    )
    monkeypatch.setattr(app, "piper_voice_supports_length_scale", lambda _voice_id: False)
    updates = app.refresh_piper_voices()
    param_updates = updates[: len(app.all_param_keys())]
    idx = app.all_param_keys().index("voice_id")
    voice_update = param_updates[idx]
    assert "voice1" in [choice[1] for choice in voice_update["choices"]]


def test_piper_assets_list_subfolders(monkeypatch, tmp_path):
    monkeypatch.setattr(piper_assets, "get_piper_voices_dir", lambda: tmp_path)
    voice_dir = tmp_path / "fr_FR" / "upmc"
    voice_dir.mkdir(parents=True)
    (voice_dir / "medium.onnx").write_bytes(b"data")
    (voice_dir / "medium.onnx.json").write_text("{}", encoding="utf-8")
    voices = piper_assets.list_installed_voices()
    assert "fr_FR/upmc/medium" in voices
