"""Glossary management endpoints — add/remove/lookup TTS pronunciation overrides."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import List

from fastapi import APIRouter, Query

from backend.schemas.models import (
    GlossaryDeleteRequest,
    GlossaryEntry,
    GlossaryListResponse,
    GlossaryUpsertRequest,
)
from backend.shared.text_tools import LEXIQUE_CACHE


router = APIRouter(prefix="/v1")

BASE_DIR = Path(__file__).resolve().parents[2]
LEXIQUE_PATH = BASE_DIR / "lexique_tts_fr.json"

_write_lock = threading.Lock()


def _load_lexique() -> dict:
    try:
        with LEXIQUE_PATH.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError):
        return {"meta": {"lang": "fr", "engine": "chatterbox"}, "exceptions": {}, "letters": {}}


def _save_lexique(data: dict) -> None:
    with LEXIQUE_PATH.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    # Invalidate cache so prepare_adjusted_text picks up changes
    LEXIQUE_CACHE.pop(str(LEXIQUE_PATH), None)


@router.get("/glossary", response_model=GlossaryListResponse)
def list_glossary() -> GlossaryListResponse:
    """List all glossary exception entries (word → pronunciation)."""
    data = _load_lexique()
    exceptions = data.get("exceptions", {})
    entries = [GlossaryEntry(word=k, pronunciation=v) for k, v in sorted(exceptions.items())]
    return GlossaryListResponse(entries=entries)


@router.put("/glossary", response_model=GlossaryEntry)
def upsert_glossary(request: GlossaryUpsertRequest) -> GlossaryEntry:
    """Add or update a glossary entry."""
    with _write_lock:
        data = _load_lexique()
        if "exceptions" not in data:
            data["exceptions"] = {}
        data["exceptions"][request.word] = request.pronunciation
        _save_lexique(data)
    return GlossaryEntry(word=request.word, pronunciation=request.pronunciation)


@router.delete("/glossary", response_model=GlossaryEntry)
def delete_glossary(word: str = Query(..., description="Word to delete")) -> GlossaryEntry:
    """Remove a glossary entry."""
    with _write_lock:
        data = _load_lexique()
        exceptions = data.get("exceptions", {})
        pronunciation = exceptions.pop(word, None)
        if pronunciation is None:
            return GlossaryEntry(word=word, pronunciation="")
        _save_lexique(data)
    return GlossaryEntry(word=word, pronunciation=pronunciation)