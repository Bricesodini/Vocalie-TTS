from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Query, Request

from backend.config import MAX_TEXT_CHARS
from backend.rate_limit import enforce_heavy
from backend.schemas.models import (
    EngineSchemaField,
    EngineSchemaResponse,
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
from text_tools import MANUAL_CHUNK_MARKER


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


@router.get("/tts/engine_schema", response_model=EngineSchemaResponse)
def get_engine_schema(engine: str = Query(...)) -> EngineSchemaResponse:
    meta = _engine_meta(engine) or {"backend_id": engine, "supports_ref": False}
    backend = get_backend(meta.get("backend_id") or engine)
    if backend is None:
        raise HTTPException(status_code=404, detail="engine_not_found")
    schema = backend.params_schema()
    fields = []
    for key, spec in schema.items():
        fields.append(
            EngineSchemaField(
                key=key,
                type=spec.type,
                label=spec.label,
                help=spec.help,
                min=spec.min,
                max=spec.max,
                step=spec.step,
                default=spec.default,
                choices=spec.choices,
                visible_if=spec.visible_if,
                serialize_scope=spec.serialize_scope,
            )
        )
    if engine.startswith("chatterbox_") or engine.startswith("qwen3_"):
        fields.append(
            EngineSchemaField(
                key="chunk_gap_ms",
                type="slider",
                min=0,
                max=2000,
                step=10,
                default=0,
                label="Blanc entre chunks (ms)",
                help="Ajoute un silence entre les chunks (Chatterbox/Qwen3).",
                serialize_scope="post",
            )
        )
    capabilities = dict(backend.capabilities())
    capabilities["supports_ref"] = bool(meta.get("supports_ref") or backend.supports_ref_audio)
    constraints = {}
    if capabilities.get("supports_ref"):
        constraints["required"] = ["voice_id"]
    return EngineSchemaResponse(
        engine_id=engine,
        backend_id=meta.get("backend_id"),
        capabilities=capabilities,
        fields=fields,
        constraints=constraints,
    )


@router.get("/tts/models", response_model=ModelsResponse)
def list_models(engine: str = Query(...)) -> ModelsResponse:
    backend = get_backend(engine)
    if backend is None:
        raise HTTPException(status_code=404, detail="engine_not_found")
    models = []
    return ModelsResponse(engine=engine, models=models)


@router.post("/tts/jobs", response_model=JobCreateResponse)
def create_job(http_request: Request, request: TTSJobRequest) -> JobCreateResponse:
    enforce_heavy(http_request)
    engine_id = request.engine_id or request.engine
    if not engine_id:
        raise HTTPException(status_code=400, detail="engine_required")
    meta = _engine_meta(engine_id)
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
    voice = request.voice_id or request.voice or None
    if meta["supports_ref"]:
        refs = list_refs()
        if voice is None or str(voice).strip() == "":
            if refs:
                voice = refs[0]
                LOGGER.info("default_voice_applied engine=%s voice=%s", engine_id, voice)
            else:
                raise HTTPException(status_code=400, detail="no reference voice available")
        elif voice not in refs:
            raise HTTPException(status_code=400, detail="reference voice not found")
    else:
        voice = None

    options = dict(request.options or {})
    if request.engine_params:
        options.update(request.engine_params)
    post_params = dict(request.post_params or {})
    gap_ms = post_params.get("chunk_gap_ms")
    if gap_ms is None:
        gap_ms = post_params.get("chatterbox_gap_ms")
    if gap_ms is not None and (engine_id.startswith("chatterbox_") or engine_id.startswith("qwen3_")):
        options["inter_chunk_gap_ms"] = int(gap_ms)
    if request.engine == "chatterbox_native" or engine_id == "chatterbox_native":
        options.setdefault("chatterbox_mode", "multilang")
    elif request.engine == "chatterbox_finetune_fr" or engine_id == "chatterbox_finetune_fr":
        options.setdefault("chatterbox_mode", "fr_finetune")
    elif request.engine == "qwen3_custom" or engine_id == "qwen3_custom":
        requested_mode = options.get("qwen3_mode")
        if requested_mode in {"custom_voice", "voice_design"}:
            options["qwen3_mode"] = requested_mode
        else:
            options["qwen3_mode"] = "custom_voice"
    elif request.engine == "qwen3_clone" or engine_id == "qwen3_clone":
        options["qwen3_mode"] = "voice_clone"
    if request.voice_id and not meta["supports_ref"]:
        options.setdefault("voice_id", request.voice_id)

    text = request.text
    if text is None:
        if request.text_source == "raw":
            text = request.text_raw
        elif request.text_source == "adjusted":
            text = request.text_adjusted or request.text_raw
        elif request.text_source == "interpreted":
            text = request.text_interpreted or request.text_adjusted or request.text_raw
        elif request.text_source == "snapshot":
            text = request.text_snapshot or request.text_interpreted or request.text_adjusted or request.text_raw
        else:
            text = request.text_interpreted or request.text_adjusted or request.text_raw

    direction_enabled = bool(request.direction.enabled) if request.direction else False
    direction_marker = request.direction.chunk_marker if request.direction else MANUAL_CHUNK_MARKER
    if request.text_snapshot:
        snapshot_text = request.text_snapshot
        if request.chunk_markers:
            for pos in sorted(set(request.chunk_markers), reverse=True):
                pos = int(pos)
                pos = max(0, min(pos, len(snapshot_text)))
                snapshot_text = f"{snapshot_text[:pos]}\n{direction_marker}\n{snapshot_text[pos:]}"
        text = snapshot_text
        if direction_marker in snapshot_text:
            direction_enabled = True

    if len(text or "") > MAX_TEXT_CHARS:
        raise HTTPException(status_code=413, detail="text_too_large")

    editing_payload = request.editing.dict() if request.editing else {}
    if request.edit_params:
        editing_payload = dict(request.edit_params)
    if editing_payload:
        editing_payload.setdefault("enabled", True)

    payload = {
        "text": text or "",
        "engine": engine_id,
        "voice": voice,
        "model": request.model,
        "language": request.language,
        "direction_enabled": direction_enabled,
        "direction_marker": direction_marker,
        "options": options,
        "export": export,
        "editing": editing_payload or {"enabled": False},
    }
    job = JOB_STORE.create_job(payload)
    if job.get("status") == "rejected":
        raise HTTPException(status_code=429, detail=job.get("error") or "rate_limited")
    return JobCreateResponse(job_id=job["job_id"], status=job["status"])
