"""Pure utility helpers extracted from the Gradio compatibility surface (app.py).

These functions have **no Gradio dependency** and are safe to import from
other modules without triggering the Gradio runtime.  The canonical backend
has equivalent functionality in ``backend/services/`` and ``backend/config.py``
for new code; this module exists to reduce the monolith size of ``app.py``
without behavior changes.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import logging
import os
from pathlib import Path

from state_manager import load_state, save_state
from text_tools import (
    ChunkInfo,
    SpeechSegment,
    estimate_duration,
    render_clean_text,
)

LOGGER = logging.getLogger("chatterbox_app")

# ---------------------------------------------------------------------------
# Path / directory constants (mirrors app.py module-level computation)
# ---------------------------------------------------------------------------

_BASE_DIR = Path(__file__).resolve().parent.parent  # repo root

_work_env = os.environ.get("VOCALIE_WORK_DIR")
WORK_DIR = Path(_work_env).expanduser() if _work_env else _BASE_DIR / "work"
WORK_DIR.mkdir(parents=True, exist_ok=True)

_output_env = os.environ.get("VOCALIE_OUTPUT_DIR") or os.environ.get("CHATTERBOX_OUT_DIR")
DEFAULT_OUTPUT_DIR = Path(_output_env).expanduser() if _output_env else _BASE_DIR / "output"
DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LEXIQUE_PATH = _BASE_DIR / "lexique_tts_fr.json"

# ---------------------------------------------------------------------------
# Chunk serialization
# ---------------------------------------------------------------------------


def serialize_chunks(chunks: list[ChunkInfo]) -> list[dict]:
    """Serialize a list of ChunkInfo to plain dicts."""
    return [dataclasses.asdict(chunk) for chunk in chunks]


def deserialize_chunks(chunks: list[dict]) -> list[ChunkInfo]:
    """Reconstruct ChunkInfo objects from plain dicts (e.g. from job payloads)."""
    rebuilt = []
    for chunk in chunks:
        segments = [SpeechSegment(**seg) for seg in chunk.get("segments", [])]
        rebuilt.append(
            ChunkInfo(
                segments=segments,
                sentence_count=chunk.get("sentence_count", 0),
                char_count=chunk.get("char_count", 0),
                word_count=chunk.get("word_count", 0),
                comma_count=chunk.get("comma_count", 0),
                estimated_duration=chunk.get("estimated_duration", 0.0),
                reason=chunk.get("reason"),
                boundary_kind=chunk.get("boundary_kind"),
                pivot=chunk.get("pivot", False),
                ends_with_suspended=chunk.get("ends_with_suspended", False),
                oversize_sentence=chunk.get("oversize_sentence", False),
                warnings=chunk.get("warnings", []),
            )
        )
    return rebuilt


# ---------------------------------------------------------------------------
# Coercion helpers
# ---------------------------------------------------------------------------


def coerce_float(value, default: float = 0.0) -> float:
    """Safely coerce *value* to float, returning *default* on failure."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def coerce_bool(value, default: bool = False) -> bool:
    """Return *value* if it is already a bool, otherwise *default*."""
    if isinstance(value, bool):
        return value
    return default


def coerce_int(value, default: int = 0) -> int:
    """Safely coerce *value* to int (via float), returning *default* on failure."""
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return int(default)


# ---------------------------------------------------------------------------
# Log formatting
# ---------------------------------------------------------------------------


def append_log(message: str, previous: str | None) -> str:
    """Append a timestamped log line after *previous* text."""
    stamp = dt.datetime.now().strftime("%H:%M:%S")
    new_line = f"[{stamp}] {message}"
    if previous:
        return f"{previous}\n{new_line}"
    return new_line


def append_ui_log(message: str, previous: str | None, verbose: bool = False, enabled: bool = True) -> str:
    """UI-aware log append: suppresses verbose messages when *enabled* is False."""
    if verbose and not enabled:
        return previous or ""
    return append_log(message, previous)


def format_adjustment_log(changes: list[str], enabled: bool) -> str:
    """Format a list of adjustment changes into a Markdown summary."""
    if not enabled:
        return ""
    if not changes:
        return "Aucune correction."
    ordered: list[str] = []
    ordered.extend([c for c in changes if c.startswith("paste_norm_applied:")])
    ordered.extend([c for c in changes if c.startswith("paste_norm_counts:")])
    ordered.extend([c for c in changes if c.startswith("sigle_undot:")])
    ordered.extend([c for c in changes if c.startswith("lexicon_hit:")])
    ordered.extend([c for c in changes if c.startswith("sigle_auto:")])
    lines = "\n".join(f"- {entry}" for entry in ordered)
    return f"**Corrections appliquées**\n{lines}"


def summarize_adjustment_changes(changes: list[str], log_text: str | None, verbose_logs: bool) -> str | None:
    """Append adjustment summaries to *log_text* respecting verbose settings."""
    if not changes:
        return log_text
    paste_entry = next((c for c in changes if c.startswith("paste_norm_applied:")), None)
    counts_entry = next((c for c in changes if c.startswith("paste_norm_counts:")), None)
    if paste_entry:
        log_text = append_ui_log(paste_entry.replace(": ", "="), log_text, verbose=True, enabled=verbose_logs)
    if counts_entry:
        log_text = append_ui_log(counts_entry.replace(": ", "="), log_text, verbose=True, enabled=verbose_logs)

    def _collect(prefix: str) -> list[str]:
        return [c[len(prefix) + 1 :].strip() for c in changes if c.startswith(f"{prefix}:")]

    for prefix in ("sigle_undot", "lexicon_hit", "sigle_auto"):
        items = _collect(prefix)
        if items:
            examples = "; ".join(items[:3])
            log_text = append_ui_log(
                f"{prefix}_count={len(items)} examples={examples}",
                log_text,
                verbose=True,
                enabled=verbose_logs,
            )
    return log_text


# ---------------------------------------------------------------------------
# Text preview / estimation
# ---------------------------------------------------------------------------


def update_clean_preview(text: str) -> str:
    """Render the clean-text preview for a given *text*."""
    return render_clean_text(text)


def update_estimated_duration(text: str) -> str:
    """Return a human-readable estimated duration string."""
    est = estimate_duration(text)
    return f"Durée estimée: {est:.1f}s"


# ---------------------------------------------------------------------------
# Path / safety utilities
# ---------------------------------------------------------------------------


def ensure_output_dir(path: str | None) -> str:
    """Return *path* as a string, creating the directory if needed.

    Falls back to the module-level ``DEFAULT_OUTPUT_DIR`` when *path* is None.
    """
    target = Path(path).expanduser() if path else DEFAULT_OUTPUT_DIR
    target.mkdir(parents=True, exist_ok=True)
    return str(target)


def is_under_dir(candidate: Path, root: Path) -> bool:
    """Return True if *candidate* is inside *root* (resolved comparison)."""
    try:
        candidate.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def cleanup_tmp(path: str | None) -> None:
    """Delete a temporary file at *path* (missing_ok, silent on error)."""
    if not path:
        return
    try:
        Path(path).unlink(missing_ok=True)
    except Exception:
        LOGGER.exception("tmp_cleanup_failed path=%s", path)


# ---------------------------------------------------------------------------
# State persistence
# ---------------------------------------------------------------------------


def persist_state(update: dict) -> None:
    """Merge *update* into the persisted state file."""
    state = load_state()
    state.update(update)
    save_state(state)


def persist_engine_state(
    engine_id: str,
    language: str | None = None,
    voice_id: str | None = None,
    params: dict | None = None,
) -> None:
    """Persist engine-specific configuration into state."""
    state = load_state()
    engines = state.get("engines")
    if not isinstance(engines, dict):
        engines = {}
    engine_cfg = engines.get(engine_id)
    if not isinstance(engine_cfg, dict):
        engine_cfg = {}
    if language is not None:
        engine_cfg["language"] = language
    if voice_id is not None:
        engine_cfg["voice_id"] = voice_id
    if params is not None:
        engine_cfg["params"] = params
    engines[engine_id] = engine_cfg
    state["engines"] = engines
    save_state(state)