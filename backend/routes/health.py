from __future__ import annotations

from fastapi import APIRouter

from backend.config import API_VERSION
from backend.schemas.models import HealthResponse
from backend.state import START_TIME
from backend.utils.time import utc_now


router = APIRouter(prefix="/v1")


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    now = utc_now()
    uptime = int((now - START_TIME).total_seconds())
    return HealthResponse(status="ok", api_version=API_VERSION, uptime_s=uptime, timestamp=now)
