from __future__ import annotations

from contextlib import asynccontextmanager
import logging

from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware

from backend.config import (
    API_VERSION,
    VOCALIE_ALLOWED_HOSTS,
    VOCALIE_CORS_ORIGINS,
    VOCALIE_ENABLE_API_DOCS,
    WORK_DIR,
)
from backend.routes import assets, audio, chunks, health, info, jobs, prep, presets, tts
from backend.security import require_authorized
from backend.services.work_service import clean_work_dir


@asynccontextmanager
async def lifespan(_app: FastAPI):
    clean_work_dir(WORK_DIR)
    try:
        from backend.services import audiosr_service

        audiosr_service.log_audiosr_status()
    except Exception:
        pass
    yield


app = FastAPI(
    title="Chatterbox TTS API",
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/v1/docs" if VOCALIE_ENABLE_API_DOCS else None,
    redoc_url="/v1/redoc" if VOCALIE_ENABLE_API_DOCS else None,
    openapi_url="/v1/openapi.json" if VOCALIE_ENABLE_API_DOCS else None,
)

allowed_hosts = [host for host in VOCALIE_ALLOWED_HOSTS if host != "*"]
if "*" in VOCALIE_ALLOWED_HOSTS:
    logging.getLogger("chatterbox_api").warning("VOCALIE_ALLOWED_HOSTS wildcard is not supported; ignoring")
if allowed_hosts:
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=allowed_hosts)

cors_origins = [origin for origin in VOCALIE_CORS_ORIGINS if origin != "*"]
if "*" in VOCALIE_CORS_ORIGINS:
    logging.getLogger("chatterbox_api").warning("VOCALIE_CORS_ORIGINS wildcard is not supported; ignoring")

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE", "PUT", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-API-Key"],
)


@app.middleware("http")
async def add_version_header(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Vocalie-Version"] = API_VERSION
    return response


app.include_router(health.router)
app.include_router(info.router, dependencies=[Depends(require_authorized)])
app.include_router(tts.router, dependencies=[Depends(require_authorized)])
app.include_router(presets.router, dependencies=[Depends(require_authorized)])
app.include_router(jobs.router, dependencies=[Depends(require_authorized)])
app.include_router(assets.router, dependencies=[Depends(require_authorized)])
app.include_router(prep.router, dependencies=[Depends(require_authorized)])
app.include_router(chunks.router, dependencies=[Depends(require_authorized)])
app.include_router(audio.router, dependencies=[Depends(require_authorized)])
