import numpy as np
import pytest

pytest.importorskip("chatterbox")

import app
from text_tools import ChunkInfo, SpeechSegment, estimate_duration
from tts_engine import TTSEngine


def test_handle_engine_change_enables_language():
    updates = app.handle_engine_change(
        "chatterbox",
        "en-US",
        "multilang",
    )
    (
        lang_update,
        lang_locked_update,
        ref_dropdown_update,
        ref_note_update,
        status_update,
        install_btn_update,
        uninstall_btn_update,
        generate_btn_update,
        _voice_label_update,
        _piper_status_update,
        _piper_refresh_update,
        _piper_install_update,
        _piper_catalog_update,
        _piper_speed_note_update,
        warning_update,
        _xtts_segmentation_update,
        _inter_chunk_gap_update,
        _inter_chunk_gap_help,
        *param_updates,
    ) = updates
    param_keys = app.all_param_keys()
    mode_update = param_updates[param_keys.index("chatterbox_mode")]
    assert mode_update["value"] == "multilang"
    assert lang_update["value"] == "en-US"
    assert lang_update["visible"] is True
    assert lang_locked_update["visible"] is False
    assert warning_update["value"] == ""
    assert isinstance(ref_note_update, dict)
    assert isinstance(ref_dropdown_update, dict)
    assert isinstance(status_update, dict)
    assert isinstance(install_btn_update, dict)
    assert isinstance(uninstall_btn_update, dict)
    assert isinstance(generate_btn_update, dict)


def test_handle_engine_change_fr_keeps_language_visible():
    updates = app.handle_engine_change(
        "chatterbox",
        "fr-FR",
        "fr_finetune",
    )
    (
        lang_update,
        lang_locked_update,
        ref_dropdown_update,
        ref_note_update,
        status_update,
        install_btn_update,
        uninstall_btn_update,
        generate_btn_update,
        _voice_label_update,
        _piper_status_update,
        _piper_refresh_update,
        _piper_install_update,
        _piper_catalog_update,
        _piper_speed_note_update,
        warning_update,
        _xtts_segmentation_update,
        _inter_chunk_gap_update,
        _inter_chunk_gap_help,
        *param_updates,
    ) = updates
    param_keys = app.all_param_keys()
    mode_update = param_updates[param_keys.index("chatterbox_mode")]
    assert mode_update["value"] == "fr_finetune"
    assert lang_update["value"] == "fr-FR"
    assert lang_update["visible"] is False
    assert lang_locked_update["visible"] is True
    assert warning_update["value"] == ""
    assert isinstance(ref_note_update, dict)
    assert isinstance(ref_dropdown_update, dict)
    assert isinstance(status_update, dict)
    assert isinstance(install_btn_update, dict)
    assert isinstance(uninstall_btn_update, dict)
    assert isinstance(generate_btn_update, dict)


def test_engine_backend_selection():
    engine = TTSEngine()
    engine._tts_fr_finetune = "fr"
    engine._tts_multilang = "multi"
    assert engine._get_backend("fr_finetune") == "fr"
    assert engine._get_backend("multilang") == "multi"
    assert engine._resolve_language("fr_finetune", "en-US") == "fr-FR"
    assert engine._resolve_language("multilang", "en-US") == "en-US"
    assert engine._map_multilang_language("fr-FR") == "fr"


def test_multilang_backend_routing(monkeypatch):
    class DummyMultilang:
        pass

    engine = TTSEngine()
    monkeypatch.setattr(engine, "_load_multilang_backend", lambda: DummyMultilang())
    backend = engine._get_backend("multilang")
    assert isinstance(backend, DummyMultilang)


def test_multilang_meta_language_mapping(monkeypatch):
    class DummyBackend:
        sr = 24000

    engine = TTSEngine()
    monkeypatch.setattr(engine, "_get_backend", lambda mode: DummyBackend())
    monkeypatch.setattr(
        engine,
        "_build_audio_from_text",
        lambda *args, **kwargs: (np.zeros(0, dtype=np.float32), []),
    )
    chunk = ChunkInfo(
        segments=[SpeechSegment("text", "Bonjour")],
        sentence_count=1,
        char_count=len("Bonjour"),
        word_count=1,
        comma_count=0,
        estimated_duration=estimate_duration("Bonjour"),
        reason="end",
        boundary_kind=None,
        pivot=False,
        ends_with_suspended=False,
        oversize_sentence=False,
        warnings=[],
    )
    _, _, meta = engine.generate_longform(
        script="Bonjour",
        chunks=[chunk],
        audio_prompt_path=None,
        out_path="/tmp/out.wav",
        tts_model_mode="multilang",
        tts_language="fr-FR",
        multilang_cfg_weight=0.3,
    )
    assert meta["requested_language_bcp47"] == "fr-FR"
    assert meta["backend_language_id"] == "fr"
