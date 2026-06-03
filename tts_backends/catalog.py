"""Canonical engine catalog and legacy alias map.

This module is the single source of truth for:
- Engine IDs and their metadata (label, backend_id).
- Legacy engine ID aliases (for backward-compatible preset migration).

``supports_ref`` is NOT stored here — it comes from the backend via
``backend.supports_ref_for_engine(engine_id)``.  This avoids a
dual-source-of-truth problem.

The ``ENGINE_CATALOG`` list is rebuilt dynamically via
``rebuild_engine_catalog()`` which is called from ``tts_backends.__init__``
after all backends have been imported and self-registered.

All consumers that need engine identity information should import from here.
No other module should define its own engine alias map.
"""

from __future__ import annotations

from typing import Dict, List


# ---------------------------------------------------------------------------
# Engine catalog — populated dynamically by rebuild_engine_catalog()
# ---------------------------------------------------------------------------

ENGINE_CATALOG: List[Dict[str, str]] = []

_CATALOG_INDEX: Dict[str, Dict[str, str]] = {}


def _build_catalog_index(catalog: List[Dict[str, str]]) -> Dict[str, Dict[str, str]]:
    """Build a dict index for O(1) lookups from the catalog list."""
    return {entry["id"]: entry for entry in catalog}


def rebuild_engine_catalog() -> None:
    """Rebuild ENGINE_CATALOG from all registered backends.

    Called once from ``tts_backends.__init__`` after backend modules
    have been imported (triggering ``__init_subclass__`` registration).
    """
    global ENGINE_CATALOG, _CATALOG_INDEX
    from .base import TTSBackend

    catalog: List[Dict[str, str]] = []
    for cls in TTSBackend._REGISTRY.values():
        for variant in cls.engine_variants():
            catalog.append({
                "id": variant["id"],
                "label": variant.get("label", cls.display_name),
                "backend_id": cls.id,
            })
    ENGINE_CATALOG = catalog
    _CATALOG_INDEX = _build_catalog_index(ENGINE_CATALOG)


def get_engine_catalog() -> List[Dict[str, str]]:
    """Return the current engine catalog (possibly dynamic)."""
    return ENGINE_CATALOG


# ---------------------------------------------------------------------------
# Legacy engine alias map — maps old IDs to canonical IDs
# ---------------------------------------------------------------------------

ENGINE_ALIAS_MAP: Dict[str, str] = {
    "chatterbox": "chatterbox_finetune_fr",
    "xtts": "xtts_v2",
}

# Backend IDs that cannot be uninstalled (core dependencies).
PROTECTED_BACKENDS: frozenset[str] = frozenset({"chatterbox"})


# ---------------------------------------------------------------------------
# Language maps — per-backend BCP47 → ISO/language mappings
# ---------------------------------------------------------------------------

CHATTERBOX_LANGUAGE_MAP: Dict[str, str] = {
    "fr-FR": "fr",
    "en-US": "en",
    "en-GB": "en",
    "es-ES": "es",
    "de-DE": "de",
    "it-IT": "it",
    "pt-PT": "pt",
    "nl-NL": "nl",
}

QWEN3_LANGUAGE_MAP: Dict[str, str] = {
    "zh-CN": "Chinese",
    "zh-TW": "Chinese",
    "en-US": "English",
    "en-GB": "English",
    "ja-JP": "Japanese",
    "ko-KR": "Korean",
    "de-DE": "German",
    "fr-FR": "French",
    "ru-RU": "Russian",
    "pt-PT": "Portuguese",
    "pt-BR": "Portuguese",
    "es-ES": "Spanish",
    "it-IT": "Italian",
}


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def canonical_engine_id(raw_id: str) -> str:
    """Resolve a potentially-legacy engine ID to its canonical form.

    If *raw_id* is already canonical, it is returned as-is.
    If *raw_id* is a known legacy alias, the canonical ID is returned.
    If *raw_id* is unknown, it is returned unchanged (graceful degradation).
    """
    return ENGINE_ALIAS_MAP.get(raw_id, raw_id)


def is_legacy_alias(engine_id: str) -> bool:
    """Return True if *engine_id* is a legacy alias (not canonical)."""
    return engine_id in ENGINE_ALIAS_MAP


def engine_meta(engine_id: str) -> Dict[str, str] | None:
    """Return catalog metadata for *engine_id*, or None if not found.

    Does NOT include ``supports_ref`` — resolve that from the backend.
    """
    entry = _CATALOG_INDEX.get(engine_id)
    return dict(entry) if entry else None