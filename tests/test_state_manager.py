from state_manager import ensure_default_presets, load_preset


def test_default_preset_defaults():
    ensure_default_presets()
    data = load_preset("default")
    assert data.get("tts_engine") == "chatterbox"
    engine_cfg = data.get("engines", {}).get("chatterbox", {})
    assert engine_cfg.get("language") == "fr-FR"
    params = engine_cfg.get("params", {})
    assert params.get("chatterbox_mode") == "fr_finetune"
    assert params.get("multilang_cfg_weight") == 0.5


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
