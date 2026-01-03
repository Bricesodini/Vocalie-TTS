from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter

from backend.schemas.models import (
    PrepAdjustRequest,
    PrepAdjustResponse,
    PrepInterpretRequest,
    PrepInterpretResponse,
)
from text_tools import prepare_adjusted_text


router = APIRouter(prefix="/v1")

BASE_DIR = Path(__file__).resolve().parents[2]
LEXIQUE_PATH = BASE_DIR / "lexique_tts_fr.json"


@router.post("/prep/adjust", response_model=PrepAdjustResponse)
def prep_adjust(request: PrepAdjustRequest) -> PrepAdjustResponse:
    text_raw = request.text_raw or ""
    adjusted_text, _changes = prepare_adjusted_text(text_raw, LEXIQUE_PATH)
    return PrepAdjustResponse(text_adjusted=adjusted_text)


@router.post("/prep/interpret", response_model=PrepInterpretResponse)
def prep_interpret(request: PrepInterpretRequest) -> PrepInterpretResponse:
    source = request.text_adjusted if request.text_adjusted is not None else request.text_raw
    text_interpreted = source or ""
    applied = [] if request.glossary_enabled else []
    return PrepInterpretResponse(text_interpreted=text_interpreted, applied_rules_summary=applied)
