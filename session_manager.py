"""Helpers for creating non-destructive session folders and metadata."""

from __future__ import annotations

import datetime as dt
import json
import shutil
from pathlib import Path
from typing import Iterable

from output_paths import ensure_unique_path, make_output_filename, slugify
from text_tools import ChunkInfo, render_clean_text_from_segments


def build_session_slug(text: str | None, user_filename: str | None) -> str:
    base = user_filename or text or ""
    return slugify(base, fallback="session")


def create_session_dir(root_dir: Path | str, created_at: dt.datetime, slug: str) -> Path:
    sessions_root = Path(root_dir) / ".sessions"
    sessions_root.mkdir(parents=True, exist_ok=True)
    stamp = created_at.strftime("%Y%m%d_%H%M%S")
    session_dir = sessions_root / f"{stamp}_{slug}"
    session_dir.mkdir(parents=True, exist_ok=True)
    _ensure_session_structure(session_dir)
    return session_dir


def _ensure_session_structure(session_dir: Path) -> None:
    for subdir in [
        session_dir / "takes" / "global",
        session_dir / "takes" / "chunks",
        session_dir / "takes" / "processed",
        session_dir / "meta",
        session_dir / "preview",
    ]:
        subdir.mkdir(parents=True, exist_ok=True)


def get_take_path_global(session_dir: Path | str, v: str = "v1") -> Path:
    session_dir = Path(session_dir)
    return session_dir / "takes" / "global" / f"global_{v}.wav"


def get_take_path_global_raw(session_dir: Path | str, v: str = "v1") -> Path:
    session_dir = Path(session_dir)
    return session_dir / "takes" / "global" / f"global_{v}_raw.wav"


def get_take_path_chunk(session_dir: Path | str, chunk_idx: int, v: str = "v1") -> Path:
    session_dir = Path(session_dir)
    chunk_dir = session_dir / "takes" / "chunks" / f"chunk_{int(chunk_idx):03d}"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    return chunk_dir / f"{v}.wav"


def get_take_path_processed_global(session_dir: Path | str, v: str = "v1") -> Path:
    session_dir = Path(session_dir)
    return session_dir / "takes" / "processed" / f"processed_global_{v}.wav"


def get_processed_preview_path(session_dir: Path | str) -> Path:
    session_dir = Path(session_dir)
    return session_dir / "preview" / "processed_preview.wav"


def write_xtts_segments(
    session_dir: Path | str,
    *,
    engine_slug: str,
    take_id: str,
    segments: list[str],
    created_at: str,
    segment_boundaries_samples: list[int] | None = None,
    sample_rate: int | None = None,
) -> Path:
    session_dir = Path(session_dir)
    meta_dir = session_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    path = meta_dir / "xtts_segments_global_v1.json"
    payload = {
        "engine_slug": str(engine_slug),
        "take_id": str(take_id),
        "segments": [str(seg) for seg in segments],
        "segment_boundaries_samples": segment_boundaries_samples or [],
        "sample_rate": int(sample_rate) if sample_rate else None,
        "created_at": str(created_at),
    }
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=True)
        fh.write("\n")
    return path


def write_processed_meta(
    session_dir: Path | str,
    *,
    engine_id: str,
    engine_slug: str,
    source_take: str,
    output_take: str,
    created_at: str,
    processing_meta: dict,
) -> Path:
    session_dir = Path(session_dir)
    meta_dir = session_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    meta_name = f"{Path(output_take).stem}.json"
    path = meta_dir / meta_name
    payload = {
        "kind": "processed",
        "source_take": str(source_take),
        "output_take": str(output_take),
        "engine_id": str(engine_id),
        "engine_slug": str(engine_slug),
        "created_at": str(created_at),
        "processing": {
            "post_processing_enabled": True,
            "mode": "minimal",
            "params": dict(processing_meta or {}),
        },
    }
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=True)
        fh.write("\n")
    return path


def next_version(existing_versions: Iterable[str]) -> str:
    max_ver = 0
    for version in existing_versions:
        if not isinstance(version, str) or not version.startswith("v"):
            continue
        suffix = version[1:]
        if suffix.isdigit():
            max_ver = max(max_ver, int(suffix))
    return f"v{max_ver + 1}"


def _serialize_chunks(chunks: Iterable[ChunkInfo]) -> list[dict]:
    payload = []
    word_cursor = 1
    for idx, chunk in enumerate(chunks, start=1):
        text = render_clean_text_from_segments(chunk.segments)
        payload.append(
            {
                "index": idx,
                "text": text,
                "start_word": int(word_cursor),
                "est_seconds": float(chunk.estimated_duration),
            }
        )
        word_cursor += max(int(chunk.word_count), 0)
    return payload


def build_session_payload(
    *,
    engine_id: str,
    engine_slug: str,
    ref_name: str | None,
    text: str,
    editorial_text: str,
    tts_ready_text: str,
    prep_log_md: str | None,
    created_at: dt.datetime,
    chunks: Iterable[ChunkInfo] | None = None,
    chunk_mode: str | None = None,
    direction_meta: dict | None = None,
    artifacts: dict | None = None,
    artifacts_list: Iterable[str | Path] | None = None,
    takes: dict | None = None,
    active_take: dict | None = None,
    active_listen: str | None = None,
) -> dict:
    payload: dict = {
        "engine_id": str(engine_id),
        "engine_slug": str(engine_slug),
        "ref_name": ref_name,
        "text": {
            "editorial": editorial_text,
            "tts_ready": tts_ready_text,
            "prep_log_md": prep_log_md or "",
        },
        "text_legacy": text,
        "created_at": created_at.isoformat(timespec="seconds"),
        "artifacts": {},
    }
    if chunks:
        payload["chunks"] = _serialize_chunks(chunks)
    if chunk_mode:
        payload["chunk_mode"] = str(chunk_mode)
    if direction_meta:
        payload["direction"] = dict(direction_meta)
    if artifacts:
        payload["artifacts"] = dict(artifacts)
    if artifacts_list:
        payload["artifacts_list"] = [str(Path(path)) for path in artifacts_list]
    if takes is not None:
        payload["takes"] = takes
    if active_take is not None:
        payload["active_take"] = active_take
    if active_listen is not None:
        payload["active_listen"] = active_listen
    return payload


def write_session_json(session_dir: Path | str, payload: dict) -> Path:
    session_dir = Path(session_dir)
    session_dir.mkdir(parents=True, exist_ok=True)
    path = session_dir / "session.json"
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=True)
        fh.write("\n")
    return path


def load_session_json(session_dir: Path | str) -> tuple[Path, dict]:
    session_dir = Path(session_dir)
    path = session_dir / "session.json"
    if not path.exists():
        raise FileNotFoundError(f"session.json introuvable: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    return path, data


def extract_session_texts(session_data: dict) -> tuple[str, str, str]:
    text_field = session_data.get("text")
    editorial = ""
    tts_ready = ""
    prep_log_md = ""
    if isinstance(text_field, dict):
        editorial = str(text_field.get("editorial") or "")
        tts_ready = str(text_field.get("tts_ready") or "")
        prep_log_md = str(text_field.get("prep_log_md") or "")
    elif isinstance(text_field, str):
        editorial = text_field
        tts_ready = text_field
    legacy = session_data.get("text_legacy") or session_data.get("input_text") or ""
    if not editorial:
        editorial = str(legacy)
    if not tts_ready:
        tts_ready = str(legacy or editorial)
    return editorial, tts_ready, prep_log_md


def stage_take_copy(session_dir: Path | str, source_path: Path | str, filename: str) -> Path:
    session_dir = Path(session_dir)
    session_dir.mkdir(parents=True, exist_ok=True)
    _ensure_session_structure(session_dir)
    source_path = Path(source_path)
    takes_dir = session_dir / "takes" / "global"
    takes_dir.mkdir(parents=True, exist_ok=True)
    target_path = takes_dir / filename
    if target_path.exists():
        target_path = ensure_unique_path(takes_dir, filename)
    shutil.copy2(source_path, target_path)
    return target_path


def stage_preview_copy(session_dir: Path | str, source_path: Path | str) -> Path:
    session_dir = Path(session_dir)
    session_dir.mkdir(parents=True, exist_ok=True)
    _ensure_session_structure(session_dir)
    source_path = Path(source_path)
    preview_path = session_dir / "preview" / "current.wav"
    shutil.copy2(source_path, preview_path)
    return preview_path


def update_session_artifacts(
    session_dir: Path | str,
    *,
    artifacts: dict | None = None,
    active_listen: str | None = None,
) -> dict:
    session_dir = Path(session_dir)
    session_path, session_data = load_session_json(session_dir)
    payload = dict(session_data)
    existing_artifacts = payload.get("artifacts")
    if not isinstance(existing_artifacts, dict):
        existing_artifacts = {}
    if artifacts:
        existing_artifacts.update(artifacts)
    payload["artifacts"] = existing_artifacts
    if active_listen is not None:
        payload["active_listen"] = active_listen
    with session_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=True)
        fh.write("\n")
    return payload


def deliver_take_to_output(
    *,
    session_dir: Path | str,
    output_dir: Path | str,
    user_filename: str | None,
    add_timestamp: bool,
    include_engine_slug: bool,
    cleanup_on_deliver: bool = False,
) -> tuple[Path, Path]:
    session_dir = Path(session_dir)
    session_path, session_data = load_session_json(session_dir)
    active_take_data = session_data.get("active_take")
    active_take = "v1"
    if isinstance(active_take_data, dict):
        active_take = active_take_data.get("global") or "v1"
    elif isinstance(active_take_data, str):
        active_take = active_take_data
    artifacts = session_data.get("artifacts")
    take_path = None
    if isinstance(artifacts, dict) and artifacts.get("raw_global"):
        candidate = session_dir / str(artifacts["raw_global"])
        if candidate.exists():
            take_path = candidate
    if take_path is None:
        legacy_raw = get_take_path_global_raw(session_dir, active_take)
        legacy_clean = get_take_path_global(session_dir, active_take)
        take_path = legacy_raw if legacy_raw.exists() else legacy_clean
    if not take_path.exists():
        raise FileNotFoundError(f"take introuvable: {take_path}")

    now = dt.datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    engine_id = session_data.get("engine_id") or "tts"
    engine_slug = session_data.get("engine_slug") or slugify(engine_id, fallback="tts")
    ref_name = session_data.get("ref_name")
    _editorial, tts_ready, _prep_log = extract_session_texts(session_data)
    filename = make_output_filename(
        text=tts_ready,
        ref_name=ref_name,
        user_filename=user_filename,
        add_timestamp=bool(add_timestamp),
        timestamp=timestamp,
        include_engine_slug=bool(include_engine_slug),
        engine_slug=engine_slug,
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    exported_path = ensure_unique_path(output_dir, filename)
    shutil.copy2(take_path, exported_path)

    delivery_info = {
        "created_at": now.isoformat(timespec="seconds"),
        "active_take": active_take,
        "src_take": str(take_path),
        "dest_path": str(exported_path),
        "engine_id": engine_id,
        "settings": {
            "include_engine_slug": bool(include_engine_slug),
            "add_timestamp": bool(add_timestamp),
            "user_filename": user_filename or "",
        },
    }
    deliveries = session_data.get("deliveries")
    if not isinstance(deliveries, list):
        deliveries = []
    deliveries.append(delivery_info)
    session_data["deliveries"] = deliveries
    with session_path.open("w", encoding="utf-8") as fh:
        json.dump(session_data, fh, indent=2, ensure_ascii=True)
        fh.write("\n")

    meta_name = f"final_{timestamp}.json"
    meta_dir = session_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    meta_path = ensure_unique_path(meta_dir, meta_name)
    with meta_path.open("w", encoding="utf-8") as fh:
        json.dump(delivery_info, fh, indent=2, ensure_ascii=True)
        fh.write("\n")
    if cleanup_on_deliver:
        shutil.rmtree(session_dir)
    return exported_path, meta_path


__all__ = [
    "build_session_payload",
    "build_session_slug",
    "create_session_dir",
    "deliver_take_to_output",
    "extract_session_texts",
    "get_take_path_chunk",
    "get_take_path_global",
    "get_take_path_global_raw",
    "get_take_path_processed_global",
    "get_processed_preview_path",
    "load_session_json",
    "next_version",
    "stage_take_copy",
    "stage_preview_copy",
    "write_processed_meta",
    "write_xtts_segments",
    "update_session_artifacts",
    "write_session_json",
]
