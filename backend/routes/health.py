from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter

from backend.config import API_VERSION
from backend.schemas.models import HealthResponse
from backend.state import START_TIME


router = APIRouter(prefix="/v1")


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    now = _utc_now()
    uptime = int((now - START_TIME).total_seconds())
    return HealthResponse(status="ok", api_version=API_VERSION, uptime_s=uptime, timestamp=now)
