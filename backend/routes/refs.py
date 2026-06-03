"""Voice reference audio endpoints — list files, configure directory."""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile

from backend.schemas.models import RefDirConfig, RefListResponse
from backend.shared.refs import ALLOWED_EXTENSIONS, _get_ref_dir, import_refs, list_refs


router = APIRouter(prefix="/v1")

# Module-level mutable state for the configured ref directory.
# Starts from env var CHATTERBOX_REF_DIR or the default (backend/shared/Ref_audio).
_current_ref_dir: str | None = os.environ.get("CHATTERBOX_REF_DIR", "")


def _ref_dir() -> Path:
    """Return the current ref directory path."""
    if _current_ref_dir:
        return _get_ref_dir(_current_ref_dir)
    return _get_ref_dir(None)


def get_current_ref_dir() -> Path:
    """Return the current ref directory (used by voice listing)."""
    return _ref_dir()


@router.get("/refs", response_model=RefListResponse)
def list_references() -> RefListResponse:
    """List voice reference files in the configured directory."""
    ref_dir = _ref_dir()
    files = list_refs(str(ref_dir))
    return RefListResponse(directory=str(ref_dir), files=files)


@router.put("/refs/dir", response_model=RefDirConfig)
def set_ref_directory(config: RefDirConfig) -> RefDirConfig:
    """Set the voice reference directory. Creates it if it doesn't exist."""
    global _current_ref_dir
    target = Path(config.directory).expanduser()
    if not target.is_absolute():
        target = Path.cwd() / target
    try:
        target.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise HTTPException(status_code=400, detail=f"Cannot create directory: {exc}") from exc
    _current_ref_dir = str(target)
    return RefDirConfig(directory=str(target))


@router.post("/refs/upload", response_model=RefListResponse)
async def upload_references(files: list[UploadFile]) -> RefListResponse:
    """Upload one or more audio files to the voice reference directory."""
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    ref_dir = _ref_dir()
    saved: list[str] = []
    for upload in files:
        if not upload.filename:
            continue
        ext = Path(upload.filename).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            continue
        content = await upload.read()
        if not content:
            continue
        # Write directly—avoid import_refs which expects file paths
        dest_dir = Path(ref_dir)
        name = upload.filename
        # Avoid overwrite by appending counter
        stem = Path(name).stem
        candidate = dest_dir / name
        counter = 1
        while candidate.exists():
            candidate = dest_dir / f"{stem}_{counter:02d}{ext}"
            counter += 1
        candidate.write_bytes(content)
        saved.append(candidate.name)

    all_files = list_refs(str(ref_dir))
    return RefListResponse(directory=str(ref_dir), files=all_files)


@router.delete("/refs/{filename}")
def delete_reference(filename: str) -> dict:
    """Delete a voice reference file by name."""
    from backend.security import safe_filename

    safe_name = safe_filename(filename)
    ref_dir = _ref_dir()
    target = ref_dir / safe_name
    if not target.exists():
        raise HTTPException(status_code=404, detail="File not found")
    target.unlink()
    return {"deleted": safe_name}