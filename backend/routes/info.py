from __future__ import annotations

import platform
from datetime import datetime, timezone

from fastapi import APIRouter

from backend.config import OUTPUT_DIR, PRESETS_DIR, WORK_DIR
from backend.schemas.models import CapabilitiesResponse, InfoResponse
from tts_backends import list_backends


router = APIRouter(prefix="/v1")


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


@router.get("/info", response_model=InfoResponse)
def info() -> InfoResponse:
    return InfoResponse(
        name="chatterbox-tts-fr",
        version="0.1.0",
        commit=None,
        python=platform.python_version(),
        os=platform.platform(),
        work_dir=str(WORK_DIR),
        output_dir=str(OUTPUT_DIR),
        presets_dir=str(PRESETS_DIR),
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
    return CapabilitiesResponse(engines=engines, features=features)
