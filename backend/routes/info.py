from __future__ import annotations

import platform
from datetime import datetime, timezone

from fastapi import APIRouter

import backend.config as backend_config
from backend.config import OUTPUT_DIR, PRESETS_DIR, WORK_DIR
from backend.schemas.models import AudioSRStatus, CapabilitiesResponse, InfoResponse
from backend.services import audiosr_service
from tts_backends import list_backends


router = APIRouter(prefix="/v1")


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


@router.get("/info", response_model=InfoResponse)
def info() -> InfoResponse:
    expose = bool(backend_config.VOCALIE_EXPOSE_SYSTEM_INFO)
    return InfoResponse(
        name="chatterbox-tts-fr",
        version="0.1.0",
        commit=None,
        python=platform.python_version() if expose else "hidden",
        os=platform.platform() if expose else "hidden",
        work_dir=str(WORK_DIR) if expose else "hidden",
        output_dir=str(OUTPUT_DIR) if expose else "hidden",
        presets_dir=str(PRESETS_DIR) if expose else "hidden",
    )


@router.get("/capabilities", response_model=CapabilitiesResponse)
def capabilities() -> CapabilitiesResponse:
    engines = [backend.id for backend in list_backends()]
    features = {
        "direction_chunking": True,
        "editing_trim": True,
        "editing_normalize": True,
        "export_formats": ["wav"],
    }
    audiosr_status = AudioSRStatus(
        enabled=backend_config.VOCALIE_ENABLE_AUDIOSR,
        available=audiosr_service.audiosr_is_available(),
    )
    return CapabilitiesResponse(engines=engines, features=features, audiosr=audiosr_status)
