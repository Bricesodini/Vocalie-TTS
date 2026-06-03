from tts_backends import get_backend, list_backends
from tts_backends.base import TTSBackend, ModelInfo, coerce_bool


def test_auto_registration():
    """Backends self-register via __init_subclass__."""
    assert "chatterbox" in TTSBackend._REGISTRY
    assert "qwen3" in TTSBackend._REGISTRY
    assert "bark" in TTSBackend._REGISTRY
    assert "piper" in TTSBackend._REGISTRY
    assert "xtts" in TTSBackend._REGISTRY


def test_backend_registry_has_chatterbox():
    backend = get_backend("chatterbox")
    assert backend is not None
    assert backend.id == "chatterbox"


def test_backend_availability_flags():
    backends = {backend.id: backend for backend in list_backends()}
    assert "chatterbox" in backends
    assert "qwen3" in backends
    assert backends["chatterbox"].is_available() in (True, False)
    assert backends["qwen3"].is_available() in (True, False)


def test_backend_language_mapping():
    backend = get_backend("chatterbox")
    assert backend is not None
    assert backend.map_language("fr-FR") == "fr"
    assert backend.map_language("en-US") == "en"


def test_backend_validate_config_returns_list():
    backend = get_backend("chatterbox")
    assert backend is not None
    warnings = backend.validate_config({"voice_ref": None})
    assert isinstance(warnings, list)


def test_get_backend_by_engine_variant():
    """get_backend resolves engine_id variants via supports_engine_id."""
    backend = get_backend("chatterbox_native")
    assert backend is not None
    assert backend.id == "chatterbox"

    backend = get_backend("qwen3_clone")
    assert backend is not None
    assert backend.id == "qwen3"


def test_list_models_qwen3():
    backend = get_backend("qwen3")
    models = backend.list_models()
    assert len(models) >= 3
    ids = [m.id for m in models]
    assert "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice" in ids
    assert "Qwen/Qwen3-TTS-12Hz-1.7B-Base" in ids


def test_list_models_chatterbox():
    backend = get_backend("chatterbox")
    models = backend.list_models()
    assert len(models) >= 2
    ids = [m.id for m in models]
    assert "ResembleAI/chatterbox" in ids
    assert "Thomcles/Chatterbox-TTS-French" in ids


def test_engine_variants():
    from tts_backends.chatterbox_backend import ChatterboxBackend
    variants = ChatterboxBackend.engine_variants()
    assert len(variants) == 2
    ids = [v["id"] for v in variants]
    assert "chatterbox_native" in ids
    assert "chatterbox_finetune_fr" in ids

    from tts_backends.qwen3_backend import Qwen3Backend
    variants = Qwen3Backend.engine_variants()
    assert len(variants) == 2
    ids = [v["id"] for v in variants]
    assert "qwen3_custom" in ids
    assert "qwen3_clone" in ids


def test_dynamic_catalog():
    from tts_backends.catalog import get_engine_catalog
    catalog = get_engine_catalog()
    assert len(catalog) >= 7  # bark + 2 chatterbox + piper + 2 qwen3 + xtts_v2
    ids = [e["id"] for e in catalog]
    assert "chatterbox_native" in ids
    assert "chatterbox_finetune_fr" in ids
    assert "qwen3_custom" in ids
    assert "qwen3_clone" in ids
    assert "bark" in ids  # bark has only 1 variant: id = backend_id


def test_coerce_bool_in_base():
    assert coerce_bool("1", False) is True
    assert coerce_bool("0", True) is False
    assert coerce_bool(None, True) is True
    assert coerce_bool(None, False) is False
    assert coerce_bool(True, False) is True
    assert coerce_bool(False, True) is False
    assert coerce_bool(1, False) is True
    assert coerce_bool(0, True) is False


def test_supports_ref_for_engine():
    backend = get_backend("qwen3")
    assert backend.supports_ref_for_engine("qwen3_custom") is False
    assert backend.supports_ref_for_engine("qwen3_clone") is True

    backend = get_backend("chatterbox")
    assert backend.supports_ref_for_engine("chatterbox_native") is True
    assert backend.supports_ref_for_engine("chatterbox_finetune_fr") is True


def test_resolve_engine_params():
    backend = get_backend("chatterbox")
    params = backend.resolve_engine_params("chatterbox_native", {})
    assert params["chatterbox_mode"] == "multilang"

    params = backend.resolve_engine_params("chatterbox_finetune_fr", {})
    assert params["chatterbox_mode"] == "fr_finetune"

    backend = get_backend("qwen3")
    params = backend.resolve_engine_params("qwen3_clone", {})
    assert params["qwen3_mode"] == "voice_clone"

    params = backend.resolve_engine_params("qwen3_custom", {})
    assert params["qwen3_mode"] == "custom_voice"


def test_catalog_backward_compat():
    from tts_backends.catalog import (
        canonical_engine_id,
        engine_meta,
        is_legacy_alias,
        CHATTERBOX_LANGUAGE_MAP,
        QWEN3_LANGUAGE_MAP,
    )
    assert canonical_engine_id("chatterbox") == "chatterbox_finetune_fr"
    assert canonical_engine_id("xtts") == "xtts_v2"
    assert is_legacy_alias("xtts") is True
    assert is_legacy_alias("chatterbox_native") is False
    meta = engine_meta("chatterbox_native")
    assert meta is not None
    assert meta["backend_id"] == "chatterbox"
    assert "fr-FR" in CHATTERBOX_LANGUAGE_MAP
    assert "fr-FR" in QWEN3_LANGUAGE_MAP


def test_model_info_dataclass():
    m = ModelInfo(id="test/model", label="Test Model", version="1.0", meta={"key": "val"})
    assert m.id == "test/model"
    assert m.label == "Test Model"
    assert m.version == "1.0"
    assert m.meta == {"key": "val"}