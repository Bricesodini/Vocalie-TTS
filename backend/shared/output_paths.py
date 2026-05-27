"""Helpers for naming output files and handling preview/user copies."""

from __future__ import annotations

import datetime as dt
import re
import unicodedata
from pathlib import Path
from typing import Tuple


FORBIDDEN_CHARS = set('\0\n\r\t\\/:*?"<>|')
MAX_FILENAME_LENGTH = 80


def slugify(value: str, fallback: str = "voix") -> str:
    """ASCII-only slug used when no filename is provided."""

    value = "" if value is None else str(value)
    normalized = unicodedata.normalize("NFKD", value)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    ascii_text = ascii_text.lower()
    ascii_text = re.sub(r"[^a-z0-9]+", "-", ascii_text).strip("-")
    ascii_text = ascii_text[:MAX_FILENAME_LENGTH].strip("-")
    return ascii_text or fallback


def sanitize_filename(name: str | None, max_length: int = MAX_FILENAME_LENGTH) -> str:
    if not name:
        return ""
    normalized = unicodedata.normalize("NFKD", name)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    cleaned = []
    for ch in ascii_text:
        if ch in FORBIDDEN_CHARS:
            continue
        cleaned.append(ch)
    result = "".join(cleaned)
    result = re.sub(r"\s+", "-", result)
    result = re.sub(r"-+", "-", result).strip("-")
    return result[:max_length]


def get_engine_slug(engine_id: str, engine_meta: dict | None = None) -> str:
    engine_meta = engine_meta or {}
    engine_id = "" if engine_id is None else str(engine_id)
    if engine_id == "chatterbox":
        mode = engine_meta.get("chatterbox_mode")
        if mode:
            return slugify(f"chatterbox_{mode}", fallback="chatterbox")
        return "chatterbox"
    if engine_id == "xtts":
        return "xtts_v2"
    if engine_id == "piper":
        return "piper"
    if engine_id == "bark":
        return "bark"
    if engine_id == "qwen3":
        mode = engine_meta.get("qwen3_mode")
        if mode:
            return slugify(f"qwen3_{mode}", fallback="qwen3")
        return "qwen3"
    return slugify(engine_id, fallback="tts")


def make_output_filename(
    text: str,
    ref_name: str | None,
    user_filename: str | None,
    add_timestamp: bool = True,
    timestamp: str | None = None,
    ext: str = "wav",
    include_engine_slug: bool = False,
    engine_slug: str | None = None,
) -> str:
    """Return a sanitized file name based on user input or text/ref fallbacks."""

    ext = ext.lstrip(".") or "wav"
    timestamp = timestamp or dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    user_base = sanitize_filename(user_filename)
    if user_base:
        base = user_base
    else:
        base = f"{slugify(text)}__{slugify(ref_name or 'ref')}"

    if include_engine_slug and engine_slug:
        base = f"{base}__{sanitize_filename(engine_slug)}"

    if add_timestamp:
        base = f"{base}__{timestamp}"

    return f"{base}.{ext}"


def ensure_unique_path(directory: Path | str, filename: str) -> Path:
    """Return a path inside *directory* that does not yet exist."""

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    base = Path(filename).stem
    ext = Path(filename).suffix
    candidate = directory / f"{base}{ext}"
    counter = 1
    while candidate.exists():
        candidate = directory / f"{base}_{counter:02d}{ext}"
        counter += 1
    return candidate


def prepare_output_paths(
    preview_dir: Path | str,
    user_dir: Path | str,
    filename: str,
) -> Tuple[Path, Path]:
    """Return safe preview and user paths using the same base filename."""

    preview_dir = Path(preview_dir)
    user_dir = Path(user_dir)
    preview_path = ensure_unique_path(preview_dir, filename)
    user_dir.mkdir(parents=True, exist_ok=True)
    user_path = user_dir / preview_path.name
    if user_path.exists() and user_path.resolve() != preview_path.resolve():
        user_path = ensure_unique_path(user_dir, preview_path.name)
    return preview_path, user_path


__all__ = [
    "FORBIDDEN_CHARS",
    "MAX_FILENAME_LENGTH",
    "ensure_unique_path",
    "get_engine_slug",
    "make_output_filename",
    "prepare_output_paths",
    "sanitize_filename",
    "slugify",
]
