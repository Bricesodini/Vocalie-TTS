from __future__ import annotations

import datetime as dt
import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
import soundfile as sf

from backend.config import OUTPUT_DIR, WORK_DIR
from backend.shared.output_paths import ensure_unique_path, get_engine_slug, make_output_filename
from backend.shared.session_manager import (
    build_session_payload,
    build_session_slug,
    create_session_dir,
    get_take_path_global_raw,
)
from backend.shared.refs import resolve_ref_path
from backend.shared.text_tools import (
    ChunkInfo,
    MANUAL_CHUNK_MARKER,
    SpeechSegment,
    count_words,
    estimate_duration,
    normalize_text,
    parse_manual_chunks,
    render_clean_text,
)
from tts_backends import get_backend
from tts_backends.base import BackendUnavailableError
from backend.shared.audio_defaults import SILENCE_MIN_MS, SILENCE_THRESHOLD
from backend.utils.time import utc_now
from backend.shared.audio_edit import apply_minimal_edit as _apply_minimal_edit, audio_meta as _audio_meta
from backend.shared.tts_pipeline import generate_raw_wav


def _single_chunk(text: str, *, reason: str) -> Optional[ChunkInfo]:
    clean = render_clean_text(text).strip()
    if not clean:
        return None
    sentence_count = len([ch for ch in clean if ch in ".!?"])
    return ChunkInfo(
        segments=[SpeechSegment("text", clean)],
        sentence_count=sentence_count,
        char_count=len(clean),
        word_count=count_words(clean),
        comma_count=clean.count(","),
        estimated_duration=estimate_duration(clean),
        reason=reason,
        boundary_kind=reason,
        pivot=False,
        ends_with_suspended=clean.rstrip().endswith((",", ";", ":")),
        oversize_sentence=False,
        warnings=[],
    )


def _build_chunks(text: str, direction_enabled: bool, marker: str) -> Tuple[list[ChunkInfo], str, dict | None]:
    if direction_enabled:
        chunks, marker_count = parse_manual_chunks(text, marker=marker)
        if marker_count > 0 and chunks:
            return chunks, "manual_marker", {"markers_count": marker_count}
        single = _single_chunk(text, reason="manual_single")
        return ([single] if single else []), "manual_single", {"markers_count": 0}
    single = _single_chunk(text, reason="single")
    return ([single] if single else []), "single", None


def run_tts_job(
    *,
    job_id: str,
    text: str,
    engine: str,
    voice: Optional[str],
    model: Optional[str],
    language: Optional[str],
    direction_enabled: bool,
    direction_marker: str,
    options: Optional[Dict[str, Any]],
    export: Dict[str, Any],
    editing: Dict[str, Any],
    progress_cb,
) -> Dict[str, Any]:
    backend = get_backend(engine)
    if backend is None:
        raise BackendUnavailableError(f"Backend introuvable: {engine}")
    if not backend.is_available():
        reason = backend.unavailable_reason() or "Dépendances manquantes."
        raise BackendUnavailableError(f"Backend indisponible: {engine}. {reason}")

    backend_id = backend.id
    if backend_id == "bark":
        direction_enabled = False
    if backend is None:
        raise BackendUnavailableError(f"Backend introuvable: {engine}")
    if not backend.is_available():
        reason = backend.unavailable_reason() or "Dépendances manquantes."
        raise BackendUnavailableError(f"Backend indisponible: {engine}. {reason}")

    progress_cb(0.05)

    normalized_text = normalize_text(text or "")
    if not normalized_text.strip():
        raise ValueError("Le texte est vide.")

    chunks, chunk_mode, direction_meta = _build_chunks(
        normalized_text,
        direction_enabled=direction_enabled,
        marker=direction_marker or MANUAL_CHUNK_MARKER,
    )
    if not chunks:
        raise ValueError("Aucun chunk généré.")

    progress_cb(0.20)

    now = utc_now()
    session_slug = build_session_slug(normalized_text, export.get("filename"))
    session_dir = create_session_dir(WORK_DIR, now, session_slug)
    raw_path = get_take_path_global_raw(session_dir, "v1")
    tmp_path = session_dir / "takes" / "global" / f"tmp_{uuid.uuid4().hex}.wav"

    engine_params = backend.resolve_engine_params(engine, dict(options or {}))

    voice_ref_path = None
    if voice:
        voice_ref_path = resolve_ref_path(voice)
    if voice:
        engine_params["voice"] = voice
    if model:
        engine_params.setdefault("model_id", model)

    payload = {
        "tts_backend": backend.id,
        "script": normalized_text,
        "chunks": chunks,
        "voice_ref_path": voice_ref_path,
        "lang": language,
        "engine_params": engine_params,
        "target_sr": 24000,
        "inter_chunk_gap_ms": int((options or {}).get("inter_chunk_gap_ms") or 0) if backend.supports_inter_chunk_gap else 0,
        "out_path": str(tmp_path),
    }

    progress_cb(0.30)
    result = generate_raw_wav(
        payload,
        progress_cb=lambda value: progress_cb(0.30 + (0.60 * float(value))),
    )
    progress_cb(0.90)

    raw_path.parent.mkdir(parents=True, exist_ok=True)
    os.replace(result.out_path, raw_path)

    engine_slug = get_engine_slug(backend_id, {"chatterbox_mode": chatterbox_mode} if chatterbox_mode else engine_params)
    artifacts = {
        "raw_global": str(Path("takes") / "global" / raw_path.name),
    }
    session_payload = build_session_payload(
        engine_id=engine,
        engine_slug=engine_slug,
        ref_name=None,
        text=normalized_text,
        editorial_text=text or "",
        tts_ready_text=normalized_text,
        prep_log_md="",
        created_at=now,
        chunks=chunks,
        chunk_mode=chunk_mode,
        direction_meta=direction_meta,
        artifacts=artifacts,
        artifacts_list=[raw_path],
        takes={"global": ["v1"], "processed": []},
        active_take={"global": "v1"},
        active_listen="raw",
    )
    session_json = session_dir / "session.json"
    session_json.write_text(
        json.dumps(session_payload, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )

    progress_cb(0.95)

    output_filename = make_output_filename(
        text=normalized_text,
        ref_name=None,
        user_filename=export.get("filename"),
        add_timestamp=bool(export.get("include_timestamp", True)),
        include_engine_slug=bool(export.get("include_model", False)),
        engine_slug=engine_slug,
        ext="wav",
    )
    output_path = ensure_unique_path(OUTPUT_DIR, output_filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(raw_path, output_path)

    edited_path = None
    if editing.get("enabled"):
        edit_filename = f"{output_path.stem}_edit{output_path.suffix}"
        edit_path = ensure_unique_path(OUTPUT_DIR, edit_filename)
        _apply_minimal_edit(
            raw_path,
            edit_path,
            trim_enabled=bool(editing.get("trim_silence", True)),
            normalize_enabled=bool(editing.get("normalize", True)),
            target_dbfs=float(editing.get("target_dbfs", -1.0)),
            silence_threshold=SILENCE_THRESHOLD,
            silence_min_ms=SILENCE_MIN_MS,
        )
        edited_path = edit_path

    progress_cb(1.0)

    audio_meta = _audio_meta(output_path)
    return {
        "output_path": output_path,
        "edited_path": edited_path,
        "session_dir": session_dir,
        "engine": engine,
        "voice": voice,
        "model": model,
        "duration_s": audio_meta.get("duration_s"),
        "sample_rate": audio_meta.get("sample_rate"),
        "size_bytes": audio_meta.get("size_bytes"),
        "created_at": now,
        "job_id": job_id,
    }
