import app
import tts_backends.piper_assets as piper_assets


def test_piper_voice_supports_length_scale_cached(monkeypatch, tmp_path):
    monkeypatch.setattr(piper_assets, "_caps_path", lambda: tmp_path / "caps.json")
    piper_assets._PIPER_CAPS_CACHE.clear()
    piper_assets._PIPER_CAPS_LOADED = False
    calls = {"count": 0}

    def fake_probe(_voice_id):
        calls["count"] += 1
        return True

    monkeypatch.setattr(piper_assets, "_probe_length_scale", fake_probe)
    assert piper_assets.piper_voice_supports_length_scale("voice") is True
    assert piper_assets.piper_voice_supports_length_scale("voice") is True
    assert calls["count"] == 1


def test_piper_speed_param_visibility():
    param_keys = app.all_param_keys()
    assert "speed" in param_keys
    idx = param_keys.index("speed")
    param_updates = app.build_param_updates(
        "piper",
        {"speed": 1.0},
        {
            "uses_internal_voices": True,
            "voice_count": 1,
            "piper_supports_speed": True,
        },
    )
    assert param_updates[idx]["visible"] is True
    param_updates = app.build_param_updates(
        "piper",
        {"speed": 1.0},
        {
            "uses_internal_voices": True,
            "voice_count": 1,
            "piper_supports_speed": False,
        },
    )
    assert param_updates[idx]["visible"] is False
