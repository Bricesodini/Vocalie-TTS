from __future__ import annotations

import datetime as dt
import math
import uuid
from pathlib import Path

import subprocess
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

from audio_defaults import SILENCE_MIN_MS, SILENCE_THRESHOLD
import backend.config as backend_config
from backend.schemas.models import AudioEditRequest, AudioEditResponse, AudioEnhanceResponse
from backend.rate_limit import enforce_heavy
from backend.services import asset_service
from backend.services import audiosr_service
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


async def _save_upload(upload: UploadFile, *, max_bytes: int) -> Path:
    suffix = Path(upload.filename or "audio").suffix or ".bin"
    upload_dir = backend_config.WORK_DIR / "uploads" / "audiosr"
    upload_dir.mkdir(parents=True, exist_ok=True)
    path = upload_dir / f"audiosr_{uuid.uuid4().hex}{suffix}"
    written = 0
    try:
        with path.open("wb") as handle:
            while True:
                chunk = await upload.read(1024 * 1024)
                if not chunk:
                    break
                written += len(chunk)
                if written > max_bytes:
                    raise HTTPException(status_code=413, detail="file_too_large")
                handle.write(chunk)
    except Exception:
        path.unlink(missing_ok=True)
        raise
    return path


def _ensure_wav(path: Path) -> Path:
    if path.suffix.lower() == ".wav":
        return path
    converted = path.with_suffix(".wav")
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(path), "-acodec", "pcm_s16le", "-ar", "48000", str(converted)],
            check=True,
            capture_output=True,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail="ffmpeg_missing") from exc
    except subprocess.CalledProcessError as exc:
        raise HTTPException(status_code=500, detail="ffmpeg_failed") from exc
    return converted


@router.post("/audio/edit", response_model=AudioEditResponse)
def edit_audio(http_request: Request, request: AudioEditRequest) -> AudioEditResponse:
    enforce_heavy(http_request)
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


@router.post("/audio/enhance", response_model=AudioEnhanceResponse)
async def enhance_audio(
    http_request: Request,
    file: UploadFile = File(...),
    engine: str = Form("audiosr"),
    ddim_steps: int = Form(100),
    guidance_scale: float = Form(2.5),
    seed: int = Form(0),
    chunk_size: int = Form(32768),
    overlap: int = Form(1024),
    multiband_ensemble: bool = Form(False),
    input_cutoff: int = Form(8000),
) -> AudioEnhanceResponse:
    enforce_heavy(http_request)
    if engine != "audiosr":
        raise HTTPException(status_code=400, detail="engine_not_supported")

    if not backend_config.VOCALIE_ENABLE_AUDIOSR:
        raise HTTPException(status_code=409, detail="audiosr_disabled")
    if not audiosr_service.audiosr_is_available():
        raise HTTPException(status_code=501, detail="audiosr_not_installed")

    params = {
        "ddim_steps": max(20, min(int(ddim_steps), 250)),
        "guidance_scale": max(1.0, min(float(guidance_scale), 4.0)),
        "seed": max(0, int(seed)),
        "chunk_size": max(0, int(chunk_size)),
        "overlap": max(0, int(overlap)),
        "multiband_ensemble": bool(multiband_ensemble),
        "input_cutoff": max(0, int(input_cutoff)),
    }

    upload_path = await _save_upload(file, max_bytes=int(backend_config.VOCALIE_MAX_UPLOAD_BYTES))
    wav_path = None
    try:
        wav_path = _ensure_wav(upload_path)
        output_path, meta_path = audiosr_service.build_output_paths(wav_path.stem)
        result = audiosr_service.run_audiosr(str(wav_path), str(output_path), params)
    except audiosr_service.FeatureDisabledError as exc:
        status = 501 if str(exc) == "audiosr_not_installed" else 409
        raise HTTPException(status_code=status, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        detail = str(exc) or "audiosr_failed"
        if len(detail) > 300:
            detail = detail[:300] + "..."
        raise HTTPException(status_code=500, detail=f"AudioSR runner failed: {detail}") from exc
    finally:
        try:
            upload_path.unlink(missing_ok=True)
        except Exception:
            pass
        if wav_path and wav_path != upload_path:
            try:
                wav_path.unlink(missing_ok=True)
            except Exception:
                pass

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
            "size_bytes": int(output_path.stat().st_size),
            "duration_s": result.get("duration_s"),
            "sample_rate": result.get("sample_rate"),
            "engine": "audiosr",
            "voice": None,
            "model": None,
        },
    )

    audiosr_service.write_sidecar(
        meta_path,
        {
            "engine": "audiosr",
            "params": params,
            "created_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "output_file": output_path.name,
            "sample_rate": result.get("sample_rate"),
            "duration_s": result.get("duration_s"),
        },
    )

    return AudioEnhanceResponse(
        output_file=str(output_path),
        sample_rate=int(result.get("sample_rate") or 48000),
        duration_s=float(result.get("duration_s") or 0.0),
        asset_id=asset_id,
        engine="audiosr",
    )
