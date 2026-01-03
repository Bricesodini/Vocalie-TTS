from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend.schemas.models import JobCancelResponse, JobStatusResponse
from backend.services.job_service import JOB_STORE


router = APIRouter(prefix="/v1")


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
def get_job(job_id: str) -> JobStatusResponse:
    job = JOB_STORE.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job_not_found")
    return JobStatusResponse(**job)


@router.delete("/jobs/{job_id}", response_model=JobCancelResponse)
def cancel_job(job_id: str) -> JobCancelResponse:
    job = JOB_STORE.cancel_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job_not_found")
    return JobCancelResponse(job_id=job_id, status=job["status"])
