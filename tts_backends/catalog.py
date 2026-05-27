"""Canonical engine catalog and legacy alias map.

This module is the single source of truth for:
- Engine IDs and their metadata (label, backend_id, supports_ref).
- Legacy engine ID aliases (for backward-compatible preset migration).

All consumers that need engine identity information should import from here.
No other module should define its own engine alias map.
"""

from __future__ import annotations

from typing import Dict, List


# ---------------------------------------------------------------------------
# Engine catalog — canonical engine definitions
# ---------------------------------------------------------------------------

ENGINE_CATALOG: List[Dict[str, str | bool]] = [
    {
        "id": "chatterbox_native",
        "label": "Chatterbox (native multilang)",
        "backend_id": "chatterbox",
        "supports_ref": True,
    },
    {
        "id": "chatterbox_finetune_fr",
        "label": "Chatterbox (FR fine-tune)",
        "backend_id": "chatterbox",
        "supports_ref": True,
    },
    {
        "id": "xtts_v2",
        "label": "XTTS v2 (voice cloning)",
        "backend_id": "xtts",
        "supports_ref": True,
    },
    {
        "id": "piper",
        "label": "Piper",
        "backend_id": "piper",
        "supports_ref": False,
    },
    {
        "id": "bark",
        "label": "Bark",
        "backend_id": "bark",
        "supports_ref": False,
    },
    {
        "id": "qwen3_custom",
        "label": "Qwen3 (CustomVoice/Design)",
        "backend_id": "qwen3",
        "supports_ref": False,
    },
    {
        "id": "qwen3_clone",
        "label": "Qwen3 (Voice clone)",
        "backend_id": "qwen3",
        "supports_ref": True,
    },
]


# ---------------------------------------------------------------------------
# Legacy engine alias map — maps old IDs to canonical IDs
# ---------------------------------------------------------------------------

ENGINE_ALIAS_MAP: Dict[str, str] = {
    "chatterbox": "chatterbox_finetune_fr",
    "xtts": "xtts_v2",
}


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


def engine_meta(engine_id: str) -> Dict[str, str | bool] | None:
    """Return catalog metadata for *engine_id*, or None if not found."""
    for entry in ENGINE_CATALOG:
        if entry["id"] == engine_id:
            return dict(entry)
    return None