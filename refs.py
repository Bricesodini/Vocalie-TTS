"""Utilities for managing reference audio files for Chatterbox TTS."""

from __future__ import annotations

import os
import shutil
import time
from pathlib import Path
from typing import Iterable, List


DEFAULT_REF_DIR = Path("/Users/bricesodini/01_ai-stack/Chatterbox/Ref_audio")
ALLOWED_EXTENSIONS = {".wav", ".mp3", ".m4a", ".aiff", ".flac"}


def _get_ref_dir(directory: str | os.PathLike[str] | None = None) -> Path:
    """Return the resolved reference directory, creating it if needed."""

    ref_dir = Path(
        directory
        or os.environ.get("CHATTERBOX_REF_DIR")
        or DEFAULT_REF_DIR
    ).expanduser()
    ref_dir.mkdir(parents=True, exist_ok=True)
    return ref_dir


def list_refs(directory: str | os.PathLike[str] | None = None) -> List[str]:
    """Return a sorted list of reference filenames found in the directory."""

    ref_dir = _get_ref_dir(directory)
    files = []
    for entry in ref_dir.iterdir():
        if entry.is_file() and entry.suffix.lower() in ALLOWED_EXTENSIONS:
            files.append(entry.name)
    return sorted(files)


def _derive_name(target_dir: Path, original_name: str) -> str:
    """Return a filename that avoids collisions inside *target_dir*."""

    stem = Path(original_name).stem or "ref"
    suffix = Path(original_name).suffix
    if suffix.lower() not in ALLOWED_EXTENSIONS:
        suffix = ".wav"

    candidate = f"{stem}{suffix}"
    if not (target_dir / candidate).exists():
        return candidate

    counter = 1
    timestamp = int(time.time())
    while True:
        candidate = f"{stem}_{counter:02d}_{timestamp}{suffix}"
        if not (target_dir / candidate).exists():
            return candidate
        counter += 1


def import_refs(
    files: Iterable[os.PathLike[str] | str | object],
    directory: str | os.PathLike[str] | None = None,
) -> List[str]:
    """Copy uploaded files into the reference directory and return their names."""

    ref_dir = _get_ref_dir(directory)
    saved: List[str] = []

    for file_obj in files or []:
        if file_obj is None:
            continue

        if isinstance(file_obj, (str, os.PathLike)):
            source_path = Path(file_obj)
        elif hasattr(file_obj, "name"):
            source_path = Path(str(getattr(file_obj, "name")))
        else:
            continue

        if not source_path.exists():
            continue

        ext = source_path.suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            continue

        dest_name = _derive_name(ref_dir, source_path.name)
        dest_path = ref_dir / dest_name
        shutil.copy2(source_path, dest_path)
        saved.append(dest_name)

    return saved


def resolve_ref_path(
    filename: str,
    directory: str | os.PathLike[str] | None = None,
) -> str:
    """Return the absolute path to *filename* inside the reference directory."""

    ref_dir = _get_ref_dir(directory)
    candidate_name = Path(str(filename)).name
    if candidate_name != str(filename) or ".." in candidate_name:
        raise ValueError("invalid_reference_name")
    candidate = (ref_dir / candidate_name).resolve()
    try:
        candidate.relative_to(ref_dir.resolve())
    except ValueError as exc:
        raise ValueError("reference_path_not_allowed") from exc
    if not candidate.exists():
        raise FileNotFoundError(f"Reference file not found: {candidate}")
    return str(candidate)


__all__ = [
    "ALLOWED_EXTENSIONS",
    "DEFAULT_REF_DIR",
    "import_refs",
    "list_refs",
    "resolve_ref_path",
]
