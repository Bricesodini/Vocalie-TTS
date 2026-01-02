from __future__ import annotations

import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from backend.config import OUTPUT_DIR
from backend.services import asset_service
from backend.services.tts_service import run_tts_job


class JobStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: Dict[str, Dict[str, Any]] = {}

    def create_job(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        job_id = f"job_{uuid.uuid4().hex}"
        now = datetime.now(timezone.utc)
        job = {
            "job_id": job_id,
            "status": "queued",
            "progress": 0.0,
            "created_at": now,
            "started_at": None,
            "finished_at": None,
            "asset_id": None,
            "error": None,
            "cancel_requested": False,
        }
        with self._lock:
            self._jobs[job_id] = job

        thread = threading.Thread(target=self._run_job, args=(job_id, payload), daemon=True)
        thread.start()
        return dict(job)

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            job = self._jobs.get(job_id)
            return dict(job) if job else None

    def cancel_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return None
            job["cancel_requested"] = True
            if job["status"] in {"queued", "running"}:
                job["status"] = "canceled"
                job["finished_at"] = datetime.now(timezone.utc)
            return dict(job)

    def _update_job(self, job_id: str, **updates) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return
            job.update(updates)

    def _progress_cb(self, job_id: str, value: float) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job or job.get("status") == "canceled":
                return
            job["progress"] = float(max(0.0, min(1.0, value)))

    def _run_job(self, job_id: str, payload: Dict[str, Any]) -> None:
        job = self.get_job(job_id)
        if job and job.get("status") == "canceled":
            return
        self._update_job(job_id, status="running", started_at=datetime.now(timezone.utc))
        try:
            result = run_tts_job(
                job_id=job_id,
                text=payload["text"],
                engine=payload["engine"],
                voice=payload.get("voice"),
                model=payload.get("model"),
                language=payload.get("language"),
                direction_enabled=payload.get("direction_enabled", False),
                direction_marker=payload.get("direction_marker", "[[CHUNK]]"),
                options=payload.get("options") or {},
                export=payload.get("export") or {},
                editing=payload.get("editing") or {},
                progress_cb=lambda v: self._progress_cb(job_id, v),
            )

            if self.get_job(job_id).get("status") == "canceled":
                return

            output_path: Path = result["output_path"]
            rel_path = None
            try:
                rel_path = str(output_path.relative_to(OUTPUT_DIR))
            except ValueError:
                rel_path = output_path.name

            asset_id = f"asset_{uuid.uuid4().hex}"
            meta_payload = {
                "file_name": output_path.name,
                "relative_path": rel_path,
                "size_bytes": int(result.get("size_bytes") or output_path.stat().st_size),
                "duration_s": result.get("duration_s"),
                "sample_rate": result.get("sample_rate"),
                "engine": result.get("engine"),
                "voice": result.get("voice"),
                "model": result.get("model"),
                "created_at": result.get("created_at").isoformat(timespec="seconds"),
                "job_id": job_id,
            }
            asset_service.write_asset_meta(asset_id, meta_payload)

            self._update_job(
                job_id,
                status="done",
                progress=1.0,
                finished_at=datetime.now(timezone.utc),
                asset_id=asset_id,
            )
        except Exception as exc:
            self._update_job(
                job_id,
                status="error",
                finished_at=datetime.now(timezone.utc),
                error=str(exc),
            )


JOB_STORE = JobStore()
