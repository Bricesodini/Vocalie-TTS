"""Text-tool constants used across the TTS preparation pipeline.

Single source of truth for the magic numbers and word lists that govern
how the chunker / normalizer / renderer behave. Kept separate from
behavior modules so the values can be tweaked or shared with the
frontend without dragging in any logic dependencies.
"""

from __future__ import annotations

import re
from typing import Dict


# ── speaking-speed estimate ─────────────────────────────────────────────
AVERAGE_WPS = 2.6  # empiric speaking speed for French promo VO


# ── chunking defaults ───────────────────────────────────────────────────
DEFAULT_MAX_CHARS_PER_CHUNK = 380
DEFAULT_MAX_PHRASES_PER_CHUNK = 3
DEFAULT_MIN_WORDS_PER_CHUNK = 16
DEFAULT_MAX_EST_SECONDS_PER_CHUNK = 10.0
DEFAULT_MAX_WORDS_WITHOUT_TERMINATOR = 35
FINAL_MERGE_EST_SECONDS = 3.5
MANUAL_CHUNK_MARKER = "[[CHUNK]]"


# ── punctuation taxonomy ────────────────────────────────────────────────
TERMINATOR_CHARS = (".", "!", "?")
FALLBACK_PUNCTUATION = (":", ";", "—", "-", ",")
PIVOT_WORDS = {
    "Cependant", "Pourtant", "Or", "Alors", "Néanmoins", "Toutefois",
}


# ── legacy SSML-style directives we silently strip ──────────────────────
LEGACY_TOKEN_PATTERN = re.compile(
    r"\{(?P<token>pause:\s*\d+|breath|beat)\}",
    re.IGNORECASE,
)


# ── French determiner set (used to avoid splitting on a leading article) ─
DETERMINERS: Dict[str, str] = {  # dict form keeps the constant O(1)-hashable
    "le", "la", "les",
    "un", "une", "des",
    "du", "de", "au", "aux",
    "ce", "cet", "cette", "ces",
    "mon", "ma", "mes",
    "ton", "ta", "tes",
    "son", "sa", "ses",
    "notre", "nos",
    "votre", "vos",
    "leur", "leurs",
}


# ── lexique cache (per-process, per-path) ───────────────────────────────
LEXIQUE_CACHE: Dict[str, Dict] = {}


__all__ = [
    "AVERAGE_WPS",
    "DEFAULT_MAX_CHARS_PER_CHUNK",
    "DEFAULT_MAX_PHRASES_PER_CHUNK",
    "DEFAULT_MIN_WORDS_PER_CHUNK",
    "DEFAULT_MAX_EST_SECONDS_PER_CHUNK",
    "DEFAULT_MAX_WORDS_WITHOUT_TERMINATOR",
    "FINAL_MERGE_EST_SECONDS",
    "MANUAL_CHUNK_MARKER",
    "TERMINATOR_CHARS",
    "FALLBACK_PUNCTUATION",
    "PIVOT_WORDS",
    "LEGACY_TOKEN_PATTERN",
    "DETERMINERS",
    "LEXIQUE_CACHE",
]
