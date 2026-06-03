"""Backend installation and management endpoints."""

from __future__ import annotations

import shutil
import logging

from fastapi import APIRouter, HTTPException

from backend_install.installer import run_install
from backend_install.status import backend_status
from backend_install.paths import venv_dir
from backend.schemas.models import BackendInstallResponse
from tts_backends.catalog import PROTECTED_BACKENDS

LOGGER = logging.getLogger("vocalie_api")

router = APIRouter(prefix="/v1/backends")


@router.post("/{engine_id}/install", response_model=BackendInstallResponse)
def install_backend(engine_id: str) -> BackendInstallResponse:
    """Install a TTS backend by downloading its dependencies into a venv."""
    status = backend_status(engine_id)
    if status.get("installed"):
        raise HTTPException(status_code=409, detail=f"Backend {engine_id} already installed")
    ok, logs = run_install(engine_id)
    if not ok:
        log_text = "\n".join(logs) if isinstance(logs, list) else str(logs)
        raise HTTPException(status_code=500, detail=f"Installation failed: {log_text}")
    log_text = "\n".join(logs) if isinstance(logs, list) else str(logs)
    return BackendInstallResponse(engine_id=engine_id, status="installed", logs=log_text)


@router.delete("/{engine_id}/uninstall", response_model=BackendInstallResponse)
def uninstall_backend(engine_id: str) -> BackendInstallResponse:
    """Uninstall a TTS backend by removing its venv."""
    if engine_id in PROTECTED_BACKENDS:
        raise HTTPException(status_code=403, detail=f"Cannot uninstall a default backend: {engine_id}")
    target = venv_dir(engine_id)
    if not target.exists():
        raise HTTPException(status_code=404, detail=f"Backend {engine_id} not found")
    try:
        shutil.rmtree(target)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Uninstall failed: {exc}") from exc
    return BackendInstallResponse(engine_id=engine_id, status="uninstalled", logs=f"Removed {target}")


@router.get("/{engine_id}/status")
def backend_status_endpoint(engine_id: str) -> dict:
    """Get installation and availability status for a TTS backend."""
    return backend_status(engine_id)