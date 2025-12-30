import app
from tts_backends import get_backend


def test_backend_capabilities_voice_modes():
    chatterbox = get_backend("chatterbox")
    piper = get_backend("piper")
    assert chatterbox.capabilities() == {
        "uses_voice_reference": True,
        "uses_internal_voices": False,
    }
    assert piper.capabilities() == {
        "uses_voice_reference": False,
        "uses_internal_voices": True,
    }


def test_piper_uses_internal_voices():
    piper = get_backend("piper")
    assert piper.uses_internal_voices is True
    assert piper.supports_ref_audio is False


def test_ref_audio_hidden_for_piper(monkeypatch):
    monkeypatch.setattr(app, "load_state", lambda: {"engines": {}})
    updates = app.handle_engine_change(
        "piper",
        "fr-FR",
        "fr_finetune",
        {"applied": True, "chunks": ["x"], "signature": ("sig",)},
    )
    ref_dropdown_update = updates[4]
    assert ref_dropdown_update["visible"] is False


def test_voice_selector_hidden_for_chatterbox(monkeypatch):
    monkeypatch.setattr(app, "load_state", lambda: {"engines": {}})
    updates = app.handle_engine_change(
        "chatterbox",
        "fr-FR",
        "fr_finetune",
        {"applied": True, "chunks": ["x"], "signature": ("sig",)},
    )
    param_updates = updates[-len(app.all_param_keys()):]
    param_keys = app.all_param_keys()
    if "voice_id" in param_keys:
        idx = param_keys.index("voice_id")
        assert param_updates[idx]["visible"] is False
