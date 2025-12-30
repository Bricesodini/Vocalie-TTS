import json

import app
import state_manager


class DummyBackend:
    def __init__(self, supported_languages, default_language=None):
        self._supported_languages = supported_languages
        self._default_language = default_language
        self.supports_ref_audio = False
        self.uses_internal_voices = False

    @property
    def supports_multilang(self):
        return len(self._supported_languages) > 1

    def supported_languages(self):
        return self._supported_languages

    def default_language(self):
        if self._default_language:
            return self._default_language
        if self._supported_languages:
            return self._supported_languages[0]
        return "fr-FR"


def test_language_default_prefers_fr():
    backend = DummyBackend(["fr-FR", "en-US"], default_language="en-US")
    final, did_fallback, lang_update, locked_update = app.language_ui_updates(
        "xtts", backend, "multilang", None
    )
    assert final == "fr-FR"
    assert did_fallback is False
    assert lang_update["visible"] is True
    assert locked_update["visible"] is False


def test_language_fallback_to_fr_on_incompatible_request():
    backend = DummyBackend(["fr-FR", "en-US"])
    final, did_fallback, _, _ = app.language_ui_updates(
        "xtts", backend, "multilang", "de-DE"
    )
    assert final == "fr-FR"
    assert did_fallback is True


def test_language_fallback_without_fr():
    backend = DummyBackend(["en-US"])
    final, did_fallback, lang_update, locked_update = app.language_ui_updates(
        "xtts", backend, "multilang", "fr-FR"
    )
    assert final == "en-US"
    assert did_fallback is True
    assert lang_update["visible"] is False
    assert locked_update["visible"] is True


def test_load_preset_defaults_language(monkeypatch, tmp_path):
    monkeypatch.setattr(state_manager, "PRESET_DIR", tmp_path)
    preset_path = tmp_path / "legacy.json"
    preset_path.write_text(json.dumps({"tts_engine": "chatterbox"}), encoding="utf-8")
    data = state_manager.load_preset("legacy")
    assert data["engines"]["chatterbox"]["language"] == "fr-FR"

    preset_path.write_text(
        json.dumps({"tts_engine": "chatterbox", "tts_language": ""}), encoding="utf-8"
    )
    data = state_manager.load_preset("legacy")
    assert data["engines"]["chatterbox"]["language"] == "fr-FR"
