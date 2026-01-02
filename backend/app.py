from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from backend.config import WORK_DIR
from backend.routes import assets, health, info, jobs, presets, tts
from backend.services.work_service import clean_work_dir


@asynccontextmanager
async def lifespan(_app: FastAPI):
    clean_work_dir(WORK_DIR)
    yield


app = FastAPI(title="Chatterbox TTS API", version="0.1.0", lifespan=lifespan)


app.include_router(health.router)
app.include_router(info.router)
app.include_router(tts.router)
app.include_router(presets.router)
app.include_router(jobs.router)
app.include_router(assets.router)
