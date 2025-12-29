from state_manager import ensure_default_presets, load_preset


def test_default_preset_min_words():
    ensure_default_presets()
    data = load_preset("default")
    assert data.get("min_words_per_chunk") == 16
    assert data.get("tts_model_mode") == "fr_finetune"
    assert data.get("tts_language") == "fr-FR"
    assert data.get("multilang_cfg_weight") == 0.5
