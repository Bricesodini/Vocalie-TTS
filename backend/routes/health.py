"""Health check and metrics endpoints."""

from __future__ import annotations

import logging

from fastapi import APIRouter

from backend.config import API_VERSION, OUTPUT_DIR, WORK_DIR
from backend.schemas.models import HealthResponse, MetricsResponse
from backend.state import START_TIME
from backend.utils.time import utc_now
from tts_backends import available_backend_ids

logger = logging.getLogger("vocalie_api")

router = APIRouter(prefix="/v1")


def _check_dir_writable(path) -> bool:
    """Check if a directory is writable by creating a temp file."""
    try:
        test_file = path / ".health_check"
        test_file.write_text("ok", encoding="utf-8")
        test_file.unlink(missing_ok=True)
        return True
    except Exception:
        return False


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    now = utc_now()
    uptime = int((now - START_TIME).total_seconds())
    work_writable = _check_dir_writable(WORK_DIR)
    output_writable = _check_dir_writable(OUTPUT_DIR)
    degraded = not work_writable or not output_writable
    return HealthResponse(
        status="degraded" if degraded else "ok",
        api_version=API_VERSION,
        uptime_s=uptime,
        timestamp=now,
        work_dir_writable=work_writable,
        output_dir_writable=output_writable,
        backends=available_backend_ids() if not degraded else None,
    )


@router.get("/metrics", response_model=MetricsResponse)
def metrics() -> MetricsResponse:
    from backend.services.job_service import JOB_STORE

    now = utc_now()
    uptime = int((now - START_TIME).total_seconds())
    jobs = list(JOB_STORE._jobs.values())
    total = len(jobs)
    completed = sum(1 for j in jobs if j.get("status") == "completed")
    failed = sum(1 for j in jobs if j.get("status") == "failed")
    pending = sum(1 for j in jobs if j.get("status") == "pending")

    return MetricsResponse(
        uptime_s=uptime,
        jobs_total=total,
        jobs_completed=completed,
        jobs_failed=failed,
        jobs_pending=pending,
        backends_available=available_backend_ids(),
        work_dir_writable=_check_dir_writable(WORK_DIR),
        output_dir_writable=_check_dir_writable(OUTPUT_DIR),
    )