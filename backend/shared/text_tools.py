"""Backwards-compatible shim for the text-preparation pipeline.

Historically the entire pipeline lived in this one module (884 lines
of constants, models, normalization, lexique, duration, chunking,
rendering).  It has since been split into focused submodules under
``backend/shared/`` so each concern lives in its own file:

  * text_constants — magic numbers, regexes, word lists
  * text_models    — DurationAdjustment, SpeechSegment, ChunkInfo
  * text_normalize — whitespace, count, paste-fr, normalize
  * text_lexique   — lexique JSON loading, chatterbox substitutions
  * text_duration  — estimate_duration, adjust_text_to_duration
  * text_chunk     — chunk_script, parse_manual_chunks, post-processing
  * text_render    — render_clean_text, render_clean_text_from_segments,
                     stitch_segments

This module re-exports every public name from those submodules so
existing ``from backend.shared.text_tools import X`` calls keep
working unchanged.
"""

from __future__ import annotations

from backend.shared.text_chunk import (
    _split_by_length,
    _split_by_word_count,
    _split_oversize_chunks,
    _split_text_by_punctuation,
    _apply_pivot_splits,
    _merge_short_chunks,
    chunk_script,
    parse_manual_chunks,
)
from backend.shared.text_constants import (
    AVERAGE_WPS,
    DEFAULT_MAX_CHARS_PER_CHUNK,
    DEFAULT_MAX_EST_SECONDS_PER_CHUNK,
    DEFAULT_MAX_PHRASES_PER_CHUNK,
    DEFAULT_MAX_WORDS_WITHOUT_TERMINATOR,
    DEFAULT_MIN_WORDS_PER_CHUNK,
    DETERMINERS,
    FALLBACK_PUNCTUATION,
    FINAL_MERGE_EST_SECONDS,
    LEGACY_TOKEN_PATTERN,
    LEXIQUE_CACHE,
    MANUAL_CHUNK_MARKER,
    PIVOT_WORDS,
    TERMINATOR_CHARS,
)
from backend.shared.text_duration import (
    adjust_text_to_duration,
    estimate_duration,
)
from backend.shared.text_lexique import (
    load_lexique_json,
    normalize_for_chatterbox,
    prepare_adjusted_text,
)
from backend.shared.text_models import (
    ChunkInfo,
    DurationAdjustment,
    SpeechSegment,
    TextUnit,
)
from backend.shared.text_normalize import (
    count_words,
    normalize_paste_fr,
    normalize_text,
    normalize_whitespace,
    strip_legacy_tokens,
)
from backend.shared.text_render import (
    render_clean_text,
    render_clean_text_from_segments,
    stitch_segments,
)


__all__ = [
    # constants
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
    # models
    "DurationAdjustment",
    "SpeechSegment",
    "TextUnit",
    "ChunkInfo",
    # normalize
    "normalize_whitespace",
    "count_words",
    "strip_legacy_tokens",
    "normalize_text",
    "normalize_paste_fr",
    # lexique
    "load_lexique_json",
    "normalize_for_chatterbox",
    "prepare_adjusted_text",
    # duration
    "estimate_duration",
    "adjust_text_to_duration",
    # chunk
    "chunk_script",
    "parse_manual_chunks",
    # render
    "render_clean_text",
    "render_clean_text_from_segments",
    "stitch_segments",
]
