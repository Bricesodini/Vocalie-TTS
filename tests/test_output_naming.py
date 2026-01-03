import json

import state_manager
from output_paths import get_engine_slug, make_output_filename


def test_output_name_includes_engine_slug_when_enabled():
    slug = get_engine_slug("chatterbox", {"chatterbox_mode": "multilang"})
    name = make_output_filename(
        text="Bonjour",
        ref_name="ref.wav",
        user_filename=None,
        add_timestamp=False,
        include_engine_slug=True,
        engine_slug=slug,
    )
    assert "__chatterbox-multilang" in name


def test_slugify_accepts_int():
    assert "123" in get_engine_slug(123, {})


def test_preset_roundtrip_includes_flag(monkeypatch, tmp_path):
    monkeypatch.setattr(state_manager, "PRESET_DIR", tmp_path)
    data = {"tts_engine": "chatterbox", "include_model_name": True}
    state_manager.save_preset("with-flag", data)
    loaded = state_manager.load_preset("with-flag")
    assert loaded.get("include_model_name") is True
