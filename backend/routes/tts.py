from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Query, Request

from backend.schemas.models import (
    EnginesResponse,
    EngineInfo,
    JobCreateResponse,
    ModelsResponse,
    ModelInfo,
    TTSJobRequest,
    VoicesResponse,
    VoiceInfo,
)
from backend.services.job_service import JOB_STORE
from refs import list_refs
from tts_backends import get_backend, list_backends


router = APIRouter(prefix="/v1")
LOGGER = logging.getLogger("chatterbox_api")


ENGINE_CATALOG = [
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
]


def _engine_meta(engine_id: str) -> dict | None:
    for entry in ENGINE_CATALOG:
        if entry["id"] == engine_id:
            return dict(entry)
    return None


def _list_reference_voices() -> list[VoiceInfo]:
    voices = []
    for name in list_refs():
        voices.append(VoiceInfo(id=name, label=name, meta={"source": "Ref_audio"}))
    return voices


@router.get("/tts/engines", response_model=EnginesResponse)
def list_engines() -> EnginesResponse:
    engines = []
    backend_availability = {backend.id: backend.is_available() for backend in list_backends()}
    for entry in ENGINE_CATALOG:
        available = backend_availability.get(entry["backend_id"], False)
        engines.append(
            EngineInfo(
                id=entry["id"],
                label=entry["label"],
                available=available,
                supports_ref=bool(entry["supports_ref"]),
            )
        )
    return EnginesResponse(engines=engines)


@router.get("/tts/voices", response_model=VoicesResponse)
def list_voices(request: Request, engine: str | None = Query(default=None)) -> VoicesResponse:
    if not engine:
        LOGGER.warning(
            "tts_voices_missing_engine url=%s ua=%s",
            request.url,
            request.headers.get("user-agent"),
        )
        LOGGER.warning(
            "tts_voices_missing_engine referer=%s",
            request.headers.get("referer"),
        )
        raise HTTPException(status_code=400, detail="engine_required")
    meta = _engine_meta(engine)
    if meta is None:
        raise HTTPException(status_code=404, detail="engine_not_found")
    voices = _list_reference_voices() if meta["supports_ref"] else []
    return VoicesResponse(engine=engine, voices=voices)


@router.get("/tts/models", response_model=ModelsResponse)
def list_models(engine: str = Query(...)) -> ModelsResponse:
    backend = get_backend(engine)
    if backend is None:
        raise HTTPException(status_code=404, detail="engine_not_found")
    models = []
    return ModelsResponse(engine=engine, models=models)


@router.post("/tts/jobs", response_model=JobCreateResponse)
def create_job(request: TTSJobRequest) -> JobCreateResponse:
    meta = _engine_meta(request.engine)
    if meta is None:
        raise HTTPException(status_code=404, detail="engine_not_found")
    export = {
        "format": "wav",
        "filename": None,
        "include_timestamp": True,
        "include_model": False,
    }
    if request.export:
        export.update(request.export.dict())
    if export.get("format") != "wav":
        raise HTTPException(status_code=400, detail="only_wav_supported")
    voice = request.voice or None
    if meta["supports_ref"]:
        refs = list_refs()
        if voice is None or str(voice).strip() == "":
            if refs:
                voice = refs[0]
                LOGGER.info("default_voice_applied engine=%s voice=%s", request.engine, voice)
            else:
                raise HTTPException(status_code=400, detail="no reference voice available")
        elif voice not in refs:
            raise HTTPException(status_code=400, detail="reference voice not found")
    else:
        voice = None

    options = dict(request.options or {})
    if request.engine == "chatterbox_native":
        options.setdefault("chatterbox_mode", "multilang")
    elif request.engine == "chatterbox_finetune_fr":
        options.setdefault("chatterbox_mode", "fr_finetune")

    payload = {
        "text": request.text,
        "engine": request.engine,
        "voice": voice,
        "model": request.model,
        "language": request.language,
        "direction_enabled": bool(request.direction.enabled) if request.direction else False,
        "direction_marker": request.direction.chunk_marker if request.direction else "[[CHUNK]]",
        "options": options,
        "export": export,
        "editing": request.editing.dict() if request.editing else {"enabled": False},
    }
    job = JOB_STORE.create_job(payload)
    return JobCreateResponse(job_id=job["job_id"], status=job["status"])
