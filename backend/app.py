from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from backend.config import API_VERSION, VOCALIE_CORS_ORIGINS, WORK_DIR
from backend.routes import assets, audio, chunks, health, info, jobs, prep, presets, tts
from backend.security import is_authorized
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


app = FastAPI(title="Chatterbox TTS API", version="0.1.0", lifespan=lifespan)

cors_origins = [origin for origin in VOCALIE_CORS_ORIGINS if origin != "*"]
if "*" in VOCALIE_CORS_ORIGINS:
    import logging

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
    if request.url.path.startswith("/v1") and not is_authorized(request):
        return JSONResponse(status_code=403, content={"detail": "forbidden"})
    response = await call_next(request)
    response.headers["X-Vocalie-Version"] = API_VERSION
    return response


app.include_router(health.router)
app.include_router(info.router)
app.include_router(tts.router)
app.include_router(presets.router)
app.include_router(jobs.router)
app.include_router(assets.router)
app.include_router(prep.router)
app.include_router(chunks.router)
app.include_router(audio.router)
