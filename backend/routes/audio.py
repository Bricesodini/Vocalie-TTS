from __future__ import annotations

import math
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException

from audio_defaults import SILENCE_MIN_MS, SILENCE_THRESHOLD
import backend.config as backend_config
from backend.schemas.models import AudioEditRequest, AudioEditResponse
from backend.services import asset_service
from backend.services.tts_service import _apply_minimal_edit, _audio_meta
from output_paths import ensure_unique_path


router = APIRouter(prefix="/v1")


def _resolve_safe_path(path_value: str) -> Path:
    candidate = Path(path_value).expanduser()
    resolved = candidate.resolve()
    for root in (backend_config.OUTPUT_DIR, backend_config.WORK_DIR):
        try:
            resolved.relative_to(root.resolve())
            return resolved
        except ValueError:
            continue
    raise HTTPException(status_code=400, detail="path_not_allowed")


def _peak_dbfs(peak: float) -> float:
    if peak <= 0.0:
        return float("-inf")
    return 20.0 * math.log10(peak)


@router.post("/audio/edit", response_model=AudioEditResponse)
def edit_audio(request: AudioEditRequest) -> AudioEditResponse:
    input_path = None
    if request.input_wav_path:
        input_path = _resolve_safe_path(request.input_wav_path)
    elif request.asset_id:
        meta = asset_service.get_asset_meta(request.asset_id)
        if not meta:
            raise HTTPException(status_code=404, detail="asset_not_found")
        resolved = asset_service.resolve_asset_path(meta)
        if not resolved:
            raise HTTPException(status_code=404, detail="asset_path_not_found")
        input_path = resolved
    if input_path is None or not input_path.exists():
        raise HTTPException(status_code=404, detail="input_audio_not_found")

    output_dir = backend_config.OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = ensure_unique_path(output_dir, f"{input_path.stem}_edit{input_path.suffix}")

    before_meta = _audio_meta(input_path)
    meta = _apply_minimal_edit(
        input_path,
        output_path,
        trim_enabled=bool(request.trim_enabled),
        normalize_enabled=bool(request.normalize_enabled),
        target_dbfs=float(request.target_dbfs),
        silence_threshold=float(SILENCE_THRESHOLD),
        silence_min_ms=int(SILENCE_MIN_MS),
    )
    after_meta = _audio_meta(output_path)
    trim_s = max(0.0, float(before_meta.get("duration_s", 0.0)) - float(after_meta.get("duration_s", 0.0)))

    metrics = {
        "trim_s": trim_s,
        "duration_before_s": before_meta.get("duration_s"),
        "duration_after_s": after_meta.get("duration_s"),
        "peak_dbfs_before": _peak_dbfs(float(meta.get("peak_before", 0.0))),
        "peak_dbfs_after": _peak_dbfs(float(meta.get("peak_after", 0.0))),
        "normalized": meta.get("normalized"),
        "trimmed": meta.get("trimmed"),
        "target_dbfs": meta.get("target_dbfs"),
    }
    rel_path = None
    try:
        rel_path = str(output_path.relative_to(backend_config.OUTPUT_DIR))
    except ValueError:
        rel_path = output_path.name
    asset_id = f"asset_{uuid.uuid4().hex}"
    asset_service.write_asset_meta(
        asset_id,
        {
            "file_name": output_path.name,
            "relative_path": rel_path,
            "size_bytes": int(after_meta.get("size_bytes") or output_path.stat().st_size),
            "duration_s": after_meta.get("duration_s"),
            "sample_rate": after_meta.get("sample_rate"),
            "engine": "edit",
            "voice": None,
            "model": None,
        },
    )
    return AudioEditResponse(edited_wav_path=str(output_path), asset_id=asset_id, metrics=metrics)
