from state_manager import ensure_default_presets, load_preset


def test_default_preset_min_words():
    ensure_default_presets()
    data = load_preset("default")
    assert data.get("min_words_per_chunk") == 16
    assert data.get("disable_newline_chunking") is False
    assert data.get("tts_engine") == "chatterbox"
    assert data.get("tts_model_mode") == "fr_finetune"
    assert data.get("tts_language") == "fr-FR"
    assert data.get("multilang_cfg_weight") == 0.5


def test_load_preset_migrates_legacy_format(tmp_path):
    from state_manager import _preset_path
    legacy_name = "legacy-format"
    path = _preset_path(legacy_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        '{"tts_model_mode":"fr_finetune","min_words_per_chunk":12}',
        encoding="utf-8",
    )
    data = load_preset(legacy_name)
    assert data.get("tts_engine") == "chatterbox"
    assert data.get("engines", {}).get("chatterbox", {}).get("language") == "fr-FR"
