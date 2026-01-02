"""Gradio interface for the local Chatterbox French TTS tool."""

from __future__ import annotations

import datetime as dt
import dataclasses
import json
import logging
import multiprocessing as mp
import os
import platform
import queue
import re
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path
from threading import Lock
from typing import Optional

import gradio as gr
import numpy as np
import soundfile as sf
import math

from logging_utils import set_verbosity
from output_paths import ensure_unique_path, get_engine_slug, make_output_filename
from refs import (
    ALLOWED_EXTENSIONS,
    DEFAULT_REF_DIR,
    import_refs,
    list_refs,
    resolve_ref_path,
)
from state_manager import (
    delete_preset,
    ensure_default_presets,
    list_presets,
    load_preset,
    load_state,
    save_preset,
    save_state,
)
from text_tools import (
    DEFAULT_MAX_EST_SECONDS_PER_CHUNK,
    ChunkInfo,
    MANUAL_CHUNK_MARKER,
    SpeechSegment,
    adjust_text_to_duration,
    count_words,
    estimate_duration,
    normalize_text,
    parse_manual_chunks,
    prepare_adjusted_text,
    render_clean_text,
    render_clean_text_from_segments,
)
from backend_install.installer import run_install
from backend_install.status import backend_status
from tts_backends import get_backend, list_backends
from tts_backends.piper_assets import (
    ensure_default_voice_installed,
    list_piper_voices,
    piper_voice_supports_length_scale,
)
from tts_backends.xtts_backend import XTTS_ASSETS_DIR
from tts_backends.base import BackendUnavailableError, coerce_language, pick_default_language
from audio_defaults import SILENCE_MIN_MS, SILENCE_THRESHOLD
from tts_pipeline import _find_active_range, generate_raw_wav
from session_manager import (
    build_session_payload,
    build_session_slug,
    create_session_dir,
    extract_session_texts,
    get_take_path_global_raw,
    load_session_json,
    write_xtts_segments,
    write_session_json,
)


logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("chatterbox_app")

BASE_DIR = Path(__file__).resolve().parent
LEXIQUE_PATH = BASE_DIR / "lexique_tts_fr.json"


work_env = os.environ.get("VOCALIE_WORK_DIR")
WORK_DIR = Path(work_env).expanduser() if work_env else BASE_DIR / "work"
WORK_DIR.mkdir(parents=True, exist_ok=True)
TMP_DIR = WORK_DIR / ".tmp"
TMP_DIR.mkdir(parents=True, exist_ok=True)

output_env = os.environ.get("VOCALIE_OUTPUT_DIR") or os.environ.get("CHATTERBOX_OUT_DIR")
DEFAULT_OUTPUT_DIR = Path(output_env).expanduser() if output_env else BASE_DIR / "output"
DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_EDIT_TARGET_DBFS = -1.0

_LOAD_PRESET_OUTPUT_COUNT: int | None = None

_JOB_LOCK = Lock()
_JOB_STATE = {
    "current_proc": None,
    "current_tmp_path": None,
    "current_final_path": None,
    "job_running": False,
}
HANDLE_GENERATE_LOG_INDEX = 4
LANGUAGE_LABELS = {
    "fr-FR": "Français (fr-FR)",
    "en-US": "English US (en-US)",
    "en-GB": "English UK (en-GB)",
    "es-ES": "Español (es-ES)",
    "de-DE": "Deutsch (de-DE)",
    "it-IT": "Italiano (it-IT)",
    "pt-PT": "Português (pt-PT)",
    "nl-NL": "Nederlands (nl-NL)",
}


def backend_choices() -> list[tuple[str, str]]:
    choices: list[tuple[str, str]] = []
    for backend in list_backends():
        label = backend.display_name
        if not backend.is_available():
            label = f"{label} (indisponible)"
        choices.append((label, backend.id))
    return choices


def backend_ref_note(backend) -> str:
    if backend is None:
        return "Moteur indisponible."
    if backend.uses_internal_voices:
        return "Référence vocale désactivée (moteur à voix internes)."
    if not backend.supports_ref_audio:
        return "Référence vocale non supportée par ce moteur."
    return ""


def _param_schema_cache():
    engine_schemas = {}
    all_keys = []
    for backend in list_backends():
        schema = backend.params_schema()
        engine_schemas[backend.id] = schema
        for key in schema:
            if key not in all_keys:
                all_keys.append(key)
    return engine_schemas, all_keys


def _param_spec_catalog() -> dict:
    engine_schemas, _ = _param_schema_cache()
    catalog: dict = {}
    for schema in engine_schemas.values():
        for key, spec in schema.items():
            if key not in catalog:
                catalog[key] = spec
    return catalog


def _ordered_languages(supported: list[str]) -> list[str]:
    if not supported:
        return ["fr-FR"]
    if "fr-FR" in supported:
        return ["fr-FR"] + [lang for lang in supported if lang != "fr-FR"]
    return list(supported)


def language_choices(supported: list[str]) -> list[tuple[str, str]]:
    ordered = _ordered_languages(supported)
    return [(LANGUAGE_LABELS.get(code, code), code) for code in ordered]


def coerce_param_value(spec, value):
    if spec.type == "bool":
        return bool(value) if isinstance(value, bool) else bool(spec.default)
    if spec.type == "int":
        try:
            coerced = int(float(value))
        except (TypeError, ValueError):
            coerced = int(spec.default)
        if spec.min is not None:
            coerced = max(int(spec.min), coerced)
        if spec.max is not None:
            coerced = min(int(spec.max), coerced)
        return coerced
    if spec.type == "float":
        try:
            coerced = float(value)
        except (TypeError, ValueError):
            coerced = float(spec.default)
        if spec.min is not None:
            coerced = max(float(spec.min), coerced)
        if spec.max is not None:
            coerced = min(float(spec.max), coerced)
        return coerced
    if spec.type in {"choice", "select"}:
        choices = spec.choices or []
        values = []
        for item in choices:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                values.append(item[1])
            else:
                values.append(item)
        return value if value in values else spec.default
    if spec.type == "str":
        return str(value) if value is not None else str(spec.default or "")
    return value if value is not None else spec.default


def spec_visible(spec, context: dict) -> bool:
    if not spec.visible_if:
        return True
    for key, expected in spec.visible_if.items():
        if key == "voice_count_min":
            if context.get("voice_count", 0) < int(expected):
                return False
            continue
        if context.get(key) != expected:
            return False
    return True


def engine_param_schema(engine_id: str) -> dict:
    schemas, _ = _param_schema_cache()
    return schemas.get(engine_id, {})


def all_param_keys() -> list[str]:
    _, keys = _param_schema_cache()
    return list(keys)


def build_param_updates(engine_id: str, engine_params: dict, context: dict) -> list[dict]:
    schema = engine_param_schema(engine_id)
    updates: list[dict] = []
    for key in all_param_keys():
        spec = schema.get(key)
        if spec is None:
            updates.append(gr.update(visible=False))
            continue
        value = coerce_param_value(spec, engine_params.get(key, spec.default))
        visible = spec_visible(spec, context)
        if spec.type in {"choice", "select"}:
            updates.append(
                gr.update(
                    value=value,
                    visible=visible,
                    interactive=visible,
                    choices=spec.choices or [],
                )
            )
        else:
            updates.append(gr.update(value=value, visible=visible, interactive=visible))
    return updates


def build_param_visibility_updates(engine_id: str, context: dict) -> list[dict]:
    schema = engine_param_schema(engine_id)
    updates: list[dict] = []
    for key in all_param_keys():
        spec = schema.get(key)
        if spec is None:
            updates.append(gr.update(visible=False))
            continue
        visible = spec_visible(spec, context)
        updates.append(gr.update(visible=visible, interactive=visible))
    return updates


def collect_engine_params(engine_id: str, values: dict) -> dict:
    schema = engine_param_schema(engine_id)
    params: dict = {}
    for key, spec in schema.items():
        params[key] = coerce_param_value(spec, values.get(key, spec.default))
    return params


def create_param_widget(spec, value, visible):
    label = spec.label or spec.key
    info = spec.help or None
    if spec.type in {"float", "int"}:
        if spec.min is None or spec.max is None or spec.step is None:
            return gr.Number(label=label, value=value, visible=visible, interactive=visible)
        return gr.Slider(
            spec.min,
            spec.max,
            value=value,
            step=spec.step,
            label=label,
            info=info,
            visible=visible,
            interactive=visible,
        )
    if spec.type in {"choice", "select"}:
        return gr.Dropdown(
            label=label,
            choices=spec.choices or [],
            value=value,
            visible=visible,
            interactive=visible,
        )
    if spec.type == "bool":
        return gr.Checkbox(
            label=label,
            value=bool(value),
            visible=visible,
            interactive=visible,
        )
    return gr.Textbox(
        label=label,
        value=str(value) if value is not None else "",
        visible=visible,
        interactive=visible,
    )


def load_engine_config(container: dict, engine_id: str) -> tuple[str, str | None, dict]:
    engines = container.get("engines", {})
    engine_data = engines.get(engine_id, {}) if isinstance(engines, dict) else {}
    language = engine_data.get("language") or "fr-FR"
    voice_id = engine_data.get("voice_id")
    params = engine_data.get("params") or {}
    return language, voice_id, params


def engine_status_markdown(engine_id: str) -> str:
    status = backend_status(engine_id)
    if status.get("installed"):
        if engine_id == "xtts" and not status.get("model_downloaded", True):
            return "Statut moteur: ⚠️ Poids XTTS non préchargés (téléchargement au premier usage)"
        return f"Statut moteur: ✅ Installé ({status.get('reason')})"
    return f"Statut moteur: ❌ Non installé ({status.get('reason')})"


def supported_languages_for(engine_id: str, backend, chatterbox_mode: str) -> list[str]:
    if backend is None:
        return ["fr-FR"]
    if engine_id == "chatterbox":
        if chatterbox_mode == "fr_finetune":
            return ["fr-FR"]
        return backend.supported_languages() or ["fr-FR"]
    return backend.supported_languages() or ["fr-FR"]


def language_ui_updates(
    engine_id: str,
    backend,
    chatterbox_mode: str,
    requested_language: str | None,
    voice_id: str | None = None,
):
    supported = supported_languages_for(engine_id, backend, chatterbox_mode)
    locked_reason = ""
    if backend and backend.uses_internal_voices:
        voices = backend.list_voices()
        voice_langs = None
        for voice in voices:
            if voice.id == voice_id and voice.lang_codes:
                voice_langs = voice.lang_codes
                break
        if voice_langs:
            supported = voice_langs
            locked_reason = " (verrouillée par la voix sélectionnée)"
    final_lang, did_fallback = coerce_language(
        requested_language or "fr-FR",
        supported,
        backend.default_language() if backend else None,
    )
    supports_select = (
        len(supported) > 1
        and bool(backend and backend.supports_multilang)
        and not (backend and backend.uses_internal_voices)
    )
    lang_update = gr.update(
        value=final_lang,
        visible=supports_select,
        interactive=supports_select,
        choices=language_choices(supported) if supports_select else None,
    )
    locked_text = ""
    if not supports_select:
        locked_text = f"Langue : {final_lang} (verrouillée{locked_reason})"
    locked_update = gr.update(value=locked_text, visible=not supports_select)
    return final_lang, did_fallback, lang_update, locked_update


def build_voice_label_update(engine_id: str, backend, requested_voice: str | None):
    if backend is None or not backend.uses_internal_voices:
        return None, gr.update(value="", visible=False), ""
    voices = backend.list_voices()
    voice_ids = [voice.id for voice in voices]
    fallback_note = ""
    final_voice = requested_voice if requested_voice in voice_ids else (voice_ids[0] if voice_ids else None)
    if requested_voice and final_voice != requested_voice:
        fallback_note = f"tts_voice_fallback requested={requested_voice} -> {final_voice} (engine={engine_id})"
    if len(voices) == 0:
        return final_voice, gr.update(value="⚠️ Aucune voix Piper installée", visible=True), fallback_note
    return final_voice, gr.update(value="", visible=False), fallback_note


def piper_speed_note_update(engine_id: str, voice_id: str | None, supports_speed: bool) -> dict:
    if engine_id != "piper" or not voice_id or supports_speed:
        return gr.update(value="", visible=False)
    return gr.update(value="Vitesse non supportée par cette voix", visible=True)


def language_warning(engine_id: str, requested: str, final: str, did_fallback: bool) -> str:
    if not did_fallback:
        return ""
    stamp = dt.datetime.now().strftime("%H:%M:%S")
    return f"[{stamp}] tts_language_fallback requested={requested} -> {final} (engine={engine_id})"


def piper_voice_status_text(voices) -> str:
    if not voices:
        return "⚠️ Aucune voix Piper installée"
    return f"✅ {len(voices)} voix Piper installée(s)"




def refresh_piper_voices() -> tuple:
    logs = []
    voices = list_piper_voices()
    backend = get_backend("piper")
    engine_params = {"voice_id": voices[0].id if voices else None}
    supports_speed = bool(
        engine_params.get("voice_id") and piper_voice_supports_length_scale(engine_params.get("voice_id"))
    )
    context = {
        "chatterbox_mode": "fr_finetune",
        "supports_ref_audio": False,
        "uses_internal_voices": True,
        "voice_count": len(voices),
        "piper_supports_speed": supports_speed,
    }
    param_updates = build_param_updates("piper", engine_params, context)
    if "voice_id" in all_param_keys():
        idx = all_param_keys().index("voice_id")
        choices = [(voice.label, voice.id) for voice in voices]
        param_updates[idx] = gr.update(
            value=engine_params.get("voice_id"),
            visible=bool(len(choices) >= 1),
            interactive=bool(len(choices) >= 1),
            choices=choices,
        )
    _, voice_label_update, _ = build_voice_label_update("piper", backend, engine_params.get("voice_id"))
    status_update = gr.update(value=piper_voice_status_text(voices), visible=True)
    speed_note_update = piper_speed_note_update(
        "piper", engine_params.get("voice_id"), supports_speed
    )
    warning_update = gr.update(value="\n".join(logs), visible=bool(logs))
    logs_update = gr.update(value="\n".join(logs))
    return (
        *param_updates,
        voice_label_update,
        status_update,
        speed_note_update,
        warning_update,
        logs_update,
    )


def install_default_piper_voice() -> tuple:
    result = ensure_default_voice_installed()
    logs = []
    if result.ok:
        logs.append("✅ Voix FR installée.")
    elif result.message:
        logs.append(f"⚠️ Install voix FR: {result.message}")
    voices = list_piper_voices()
    backend = get_backend("piper")
    engine_params = {"voice_id": voices[0].id if voices else None}
    supports_speed = bool(
        engine_params.get("voice_id") and piper_voice_supports_length_scale(engine_params.get("voice_id"))
    )
    context = {
        "chatterbox_mode": "fr_finetune",
        "supports_ref_audio": False,
        "uses_internal_voices": True,
        "voice_count": len(voices),
        "piper_supports_speed": supports_speed,
    }
    param_updates = build_param_updates("piper", engine_params, context)
    if "voice_id" in all_param_keys():
        idx = all_param_keys().index("voice_id")
        choices = [(voice.label, voice.id) for voice in voices]
        param_updates[idx] = gr.update(
            value=engine_params.get("voice_id"),
            visible=bool(len(choices) >= 1),
            interactive=bool(len(choices) >= 1),
            choices=choices,
        )
    _, voice_label_update, _ = build_voice_label_update("piper", backend, engine_params.get("voice_id"))
    status_update = gr.update(value=piper_voice_status_text(voices), visible=True)
    speed_note_update = piper_speed_note_update(
        "piper", engine_params.get("voice_id"), supports_speed
    )
    warning_update = gr.update(value="\n".join(logs), visible=bool(logs))
    logs_update = gr.update(value="\n".join(logs))
    return (
        *param_updates,
        voice_label_update,
        status_update,
        speed_note_update,
        warning_update,
        logs_update,
    )


def engine_status_updates(engine_id: str):
    status = backend_status(engine_id)
    installed = bool(status.get("installed"))
    status_md = gr.update(value=engine_status_markdown(engine_id))
    install_btn = gr.update(visible=not installed, interactive=not installed)
    uninstall_btn = gr.update(visible=installed and engine_id != "chatterbox", interactive=installed)
    generate_btn = gr.update(interactive=installed)
    return status_md, install_btn, uninstall_btn, generate_btn


def _serialize_chunks(chunks: list[ChunkInfo]) -> list[dict]:
    return [dataclasses.asdict(chunk) for chunk in chunks]


def _deserialize_chunks(chunks: list[dict]) -> list[ChunkInfo]:
    rebuilt = []
    for chunk in chunks:
        segments = [SpeechSegment(**seg) for seg in chunk.get("segments", [])]
        rebuilt.append(
            ChunkInfo(
                segments=segments,
                sentence_count=chunk.get("sentence_count", 0),
                char_count=chunk.get("char_count", 0),
                word_count=chunk.get("word_count", 0),
                comma_count=chunk.get("comma_count", 0),
                estimated_duration=chunk.get("estimated_duration", 0.0),
                reason=chunk.get("reason"),
                boundary_kind=chunk.get("boundary_kind"),
                pivot=chunk.get("pivot", False),
                ends_with_suspended=chunk.get("ends_with_suspended", False),
                oversize_sentence=chunk.get("oversize_sentence", False),
                warnings=chunk.get("warnings", []),
            )
        )
    return rebuilt


def mac_choose_folder(initial_dir: str | None = None) -> str | None:
    if sys.platform != "darwin":
        return None

    script = 'POSIX path of (choose folder'
    if initial_dir:
        safe_dir = initial_dir.replace('"', '\\"')
        script += f' default location POSIX file "{safe_dir}"'
    script += ')'

    try:
        out = subprocess.check_output(["osascript", "-e", script], text=True).strip()
        return out if out else None
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _reset_job_state() -> dict:
    with _JOB_LOCK:
        _JOB_STATE.update(
            {
                "current_proc": None,
                "current_tmp_path": None,
                "current_final_path": None,
                "job_running": False,
            }
        )
        return dict(_JOB_STATE)


def _set_job_state(**updates) -> dict:
    with _JOB_LOCK:
        _JOB_STATE.update(updates)
        return dict(_JOB_STATE)


def _get_job_state() -> dict:
    with _JOB_LOCK:
        return dict(_JOB_STATE)


def _cleanup_tmp(path: str | None) -> None:
    if not path:
        return
    try:
        Path(path).unlink(missing_ok=True)
    except Exception:
        LOGGER.exception("tmp_cleanup_failed path=%s", path)


def _terminate_proc(proc: mp.Process | None, timeout: float = 0.8) -> None:
    if not proc:
        return
    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=timeout)


def _generate_longform_worker(payload: dict, result_queue: mp.Queue) -> None:
    try:
        backend_id = payload.pop("tts_backend")
        if "audio_prompt_path" in payload and "voice_ref_path" not in payload:
            payload["voice_ref_path"] = payload.pop("audio_prompt_path")
        if isinstance(payload.get("chunks"), list):
            if payload["chunks"] and isinstance(payload["chunks"][0], dict):
                payload["chunks"] = _deserialize_chunks(payload["chunks"])
        request = dict(payload)
        request["tts_backend"] = backend_id
        meta = generate_raw_wav(request).meta
        meta.setdefault("warnings", [])
        meta.setdefault("backend_id", backend_id)
        meta.setdefault("backend_lang", payload.get("lang"))
        meta.setdefault("chunks", len(payload.get("chunks") or []))
        meta.setdefault("durations", [])
        meta.setdefault("total_duration", 0.0)
        result_queue.put({"status": "ok", "meta": meta})
    except BackendUnavailableError as exc:
        result_queue.put({"status": "unavailable", "error": str(exc)})
    except Exception as exc:
        result_queue.put({"status": "error", "error": str(exc)})


def ensure_output_dir(path: str | None) -> str:
    target = Path(path).expanduser() if path else DEFAULT_OUTPUT_DIR
    target.mkdir(parents=True, exist_ok=True)
    return str(target)


def clean_work_dir(work_root: Path) -> int:
    if os.environ.get("VOCALIE_KEEP_WORK") == "1":
        LOGGER.info("work cleanup skipped (VOCALIE_KEEP_WORK=1)")
        return 0
    base_root = BASE_DIR.resolve()
    work_root = Path(work_root).expanduser().resolve()
    try:
        work_root.relative_to(base_root)
    except ValueError as exc:
        raise ValueError(f"work_root must be inside repo: {work_root}") from exc
    work_root.mkdir(parents=True, exist_ok=True)
    sessions_dir = work_root / ".sessions"
    tmp_dir = work_root / ".tmp"
    tmp_dir_alt = work_root / "tmp"
    removed_sessions = 0
    if sessions_dir.exists():
        for entry in sessions_dir.iterdir():
            if entry.is_dir():
                shutil.rmtree(entry, ignore_errors=True)
                removed_sessions += 1
            elif entry.is_file():
                entry.unlink(missing_ok=True)
                removed_sessions += 1
    for tmp_path in (tmp_dir, tmp_dir_alt):
        if tmp_path.exists():
            for entry in tmp_path.iterdir():
                if entry.is_dir():
                    shutil.rmtree(entry, ignore_errors=True)
                elif entry.is_file():
                    entry.unlink(missing_ok=True)
    LOGGER.info("work cleaned: %s sessions removed", removed_sessions)
    return removed_sessions


def _is_under_dir(candidate: Path, root: Path) -> bool:
    try:
        candidate.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _resolve_raw_take_path(session_dir: Path, session_data: dict) -> Path:
    artifacts = session_data.get("artifacts")
    if isinstance(artifacts, dict) and artifacts.get("raw_global"):
        candidate = session_dir / str(artifacts["raw_global"])
        if candidate.exists():
            return candidate
    active_take_data = session_data.get("active_take")
    active_take = "v1"
    if isinstance(active_take_data, dict):
        active_take = active_take_data.get("global") or "v1"
    elif isinstance(active_take_data, str):
        active_take = active_take_data
    legacy_raw = get_take_path_global_raw(session_dir, active_take)
    if legacy_raw.exists():
        return legacy_raw
    return get_take_path_global_raw(session_dir, "v1")


def _apply_minimal_edit(
    raw_path: Path,
    output_path: Path,
    *,
    trim_enabled: bool,
    normalize_enabled: bool,
    target_dbfs: float,
    silence_threshold: float,
    silence_min_ms: int,
) -> dict:
    audio, sr = sf.read(str(raw_path), always_2d=False)
    if not isinstance(sr, (int, float)):
        raise ValueError("Sample rate invalide pour l'édition.")
    sr = int(sr)
    audio = np.asarray(audio, dtype=np.float32)
    trimmed = False
    if trim_enabled:
        mono = np.mean(audio, axis=1) if audio.ndim > 1 else audio
        min_silence_frames = int(sr * (int(silence_min_ms) / 1000.0))
        start_idx, end_idx = _find_active_range(
            mono,
            threshold=float(silence_threshold),
            min_silence_frames=min_silence_frames,
        )
        if 0 <= start_idx < end_idx <= len(audio):
            audio = audio[start_idx:end_idx]
            trimmed = True
    normalized = False
    peak_before = float(np.max(np.abs(audio))) if audio.size else 0.0
    target_peak = 10 ** (float(target_dbfs) / 20.0)
    gain = 1.0
    if normalize_enabled and peak_before > 0.0 and target_peak > 0.0:
        gain = target_peak / peak_before
        audio = audio * gain
        normalized = True
    output_path.parent.mkdir(parents=True, exist_ok=True)
    audio = np.clip(audio, -1.0, 1.0)
    sf.write(str(output_path), audio, sr, subtype="PCM_16")
    return {
        "trimmed": trimmed,
        "normalized": normalized,
        "target_dbfs": float(target_dbfs),
        "peak_before": peak_before,
        "peak_after": float(np.max(np.abs(audio))) if audio.size else 0.0,
        "gain": gain,
    }


def update_edit_panel_state(
    edit_enabled: bool,
    trim_enabled: bool,
    normalize_enabled: bool,
):
    panel_update = gr.update(visible=bool(edit_enabled))
    button_update = gr.update(interactive=bool(edit_enabled and (trim_enabled or normalize_enabled)))
    slider_update = gr.update(interactive=bool(edit_enabled and normalize_enabled))
    return panel_update, button_update, slider_update


def _resolve_direction_source_text(
    source: str | None,
    *,
    original_text: str,
    adjusted_text: str,
    final_text: str,
) -> str:
    source = str(source or "final")
    if source == "original":
        return original_text
    if source == "adjusted":
        return adjusted_text
    return final_text


def handle_direction_load_snapshot(
    direction_source: str | None,
    original_text: str,
    adjusted_text: str,
    final_text: str,
    log_text: str | None,
):
    snapshot = _resolve_direction_source_text(
        direction_source,
        original_text=original_text or "",
        adjusted_text=adjusted_text or "",
        final_text=final_text or "",
    )
    log_text = append_ui_log(f"Snapshot chargé (source={direction_source})", log_text)
    return snapshot, log_text


def handle_direction_preview(
    direction_enabled: bool,
    direction_source: str | None,
    direction_snapshot_text: str | None,
    final_text: str,
    log_text: str | None,
):
    chunks, _chunk_mode, _direction_meta, log_text = _apply_direction_chunking(
        direction_enabled=bool(direction_enabled),
        direction_source=direction_source,
        direction_snapshot_text=direction_snapshot_text,
        tts_ready_text=final_text,
        log_text=log_text,
    )
    preview = _build_chunk_preview(chunks) if chunks else ""
    return preview, log_text


def update_direction_controls(direction_enabled: bool, snapshot_text: str | None):
    has_snapshot = bool(snapshot_text and snapshot_text.strip())
    insert_update = gr.update(interactive=bool(direction_enabled and has_snapshot))
    preview_update = gr.update(interactive=bool(direction_enabled and has_snapshot))
    return insert_update, preview_update


def append_chunk_marker(snapshot: str | None) -> str:
    return snapshot or ""


def handle_generate_edited_audio(
    session_state: dict | None,
    out_dir: str | None,
    user_filename: str | None,
    add_timestamp: bool,
    include_model_name: bool,
    trim_enabled: bool,
    normalize_enabled: bool,
    target_dbfs: float,
    log_text: str | None,
):
    state = dict(session_state or {})
    session_dir_value = state.get("dir")
    if not session_dir_value:
        return gr.update(), gr.update(), append_ui_log("Aucune session active.", log_text)
    if not trim_enabled and not normalize_enabled:
        return (
            gr.update(),
            gr.update(),
            append_ui_log("Édition minimale: aucune option sélectionnée.", log_text),
        )
    session_dir = Path(session_dir_value)
    try:
        _session_path, session_data = load_session_json(session_dir)
    except Exception as exc:
        return gr.update(), gr.update(), append_ui_log(f"Session illisible: {exc}", log_text)
    raw_path = _resolve_raw_take_path(session_dir, session_data)
    if not raw_path.exists():
        return gr.update(), gr.update(), append_ui_log("RAW introuvable.", log_text)
    output_dir = Path(ensure_output_dir(out_dir))
    _editorial, tts_ready, _prep_log = extract_session_texts(session_data)
    engine_slug = session_data.get("engine_slug") or get_engine_slug(session_data.get("engine_id") or "tts", {})
    filename = make_output_filename(
        text=tts_ready,
        ref_name=session_data.get("ref_name"),
        user_filename=user_filename,
        add_timestamp=bool(add_timestamp),
        include_engine_slug=bool(include_model_name),
        engine_slug=engine_slug,
    )
    edit_filename = f"{Path(filename).stem}_edit{Path(filename).suffix}"
    output_path = ensure_unique_path(output_dir, edit_filename)
    try:
        meta = _apply_minimal_edit(
            raw_path,
            output_path,
            trim_enabled=bool(trim_enabled),
            normalize_enabled=bool(normalize_enabled),
            target_dbfs=float(target_dbfs),
            silence_threshold=SILENCE_THRESHOLD,
            silence_min_ms=SILENCE_MIN_MS,
        )
    except Exception as exc:
        return gr.update(), gr.update(), append_ui_log(f"Édition impossible: {exc}", log_text)
    peak_before = float(meta.get("peak_before") or 0.0)
    peak_after = float(meta.get("peak_after") or 0.0)
    if peak_before > 0.0:
        peak_before_dbfs = 20.0 * math.log10(peak_before)
    else:
        peak_before_dbfs = float("-inf")
    if peak_after > 0.0:
        peak_after_dbfs = 20.0 * math.log10(peak_after)
    else:
        peak_after_dbfs = float("-inf")
    log_text = append_ui_log(
        (
            f"Édition minimale OK: {output_path} "
            f"(trim={meta['trimmed']} normalize={meta['normalized']} target={meta['target_dbfs']:.1f}dBFS "
            f"peak_before={peak_before_dbfs:.2f}dBFS peak_after={peak_after_dbfs:.2f}dBFS "
            f"gain={meta['gain']:.3f})"
        ),
        log_text,
    )
    return str(output_path), str(output_path), log_text


def handle_export_raw_to_output(
    session_state: dict | None,
    out_dir: str | None,
    user_filename: str | None,
    add_timestamp: bool,
    include_model_name: bool,
    log_text: str | None,
):
    state = dict(session_state or {})
    session_dir_value = state.get("dir")
    if not session_dir_value:
        return gr.update(), append_ui_log("Aucune session active.", log_text)
    session_dir = Path(session_dir_value)
    try:
        _session_path, session_data = load_session_json(session_dir)
    except Exception as exc:
        return gr.update(), append_ui_log(f"Session illisible: {exc}", log_text)
    raw_path = _resolve_raw_take_path(session_dir, session_data)
    if not raw_path.exists():
        return gr.update(), append_ui_log("RAW introuvable.", log_text)
    output_dir = Path(ensure_output_dir(out_dir))
    _editorial, tts_ready, _prep_log = extract_session_texts(session_data)
    engine_slug = session_data.get("engine_slug") or get_engine_slug(session_data.get("engine_id") or "tts", {})
    filename = make_output_filename(
        text=tts_ready,
        ref_name=session_data.get("ref_name"),
        user_filename=user_filename,
        add_timestamp=bool(add_timestamp),
        include_engine_slug=bool(include_model_name),
        engine_slug=engine_slug,
    )
    raw_filename = f"{Path(filename).stem}_raw{Path(filename).suffix}"
    output_path = ensure_unique_path(output_dir, raw_filename)
    try:
        shutil.copy2(raw_path, output_path)
    except Exception as exc:
        return gr.update(), append_ui_log(f"Export RAW impossible: {exc}", log_text)
    log_text = append_ui_log(f"RAW exporté: {output_path}", log_text)
    return str(output_path), log_text


def handle_open_output_dir(out_dir: str | None, log_text: str | None):
    output_dir = Path(ensure_output_dir(out_dir))
    try:
        if sys.platform.startswith("darwin"):
            subprocess.run(["open", str(output_dir)], check=False)
        elif os.name == "nt":
            os.startfile(str(output_dir))  # type: ignore[attr-defined]
        else:
            subprocess.run(["xdg-open", str(output_dir)], check=False)
    except Exception as exc:
        return append_ui_log(f"Ouverture dossier impossible: {exc}", log_text)
    return append_ui_log(f"Dossier output ouvert: {output_dir}", log_text)


def _coerce_float(value, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _coerce_bool(value, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    return default


def persist_state(update: dict) -> None:
    state = load_state()
    state.update(update)
    save_state(state)


def persist_engine_state(engine_id: str, language: str | None = None, voice_id: str | None = None, params: dict | None = None) -> None:
    state = load_state()
    engines = state.get("engines")
    if not isinstance(engines, dict):
        engines = {}
    engine_cfg = engines.get(engine_id)
    if not isinstance(engine_cfg, dict):
        engine_cfg = {}
    if language is not None:
        engine_cfg["language"] = language
    if voice_id is not None:
        engine_cfg["voice_id"] = voice_id
    if params is not None:
        engine_cfg["params"] = params
    engines[engine_id] = engine_cfg
    state["engines"] = engines
    save_state(state)


def append_log(message: str, previous: str | None) -> str:
    stamp = dt.datetime.now().strftime("%H:%M:%S")
    new_line = f"[{stamp}] {message}"
    if previous:
        return f"{previous}\n{new_line}"
    return new_line


def append_ui_log(message: str, previous: str | None, verbose: bool = False, enabled: bool = True) -> str:
    if verbose and not enabled:
        return previous or ""
    return append_log(message, previous)


def refresh_dropdown(current: str | None) -> gr.Dropdown:
    refs = list_refs()
    value = current if current in refs else (refs[0] if refs else None)
    return gr.update(choices=refs, value=value)


def handle_upload(files, current: str | None, log_text: str | None):
    saved = import_refs(files)
    refs = list_refs()
    value = current if current in refs else (refs[0] if refs else None)
    if saved:
        value = saved[-1]
        log_text = append_log(f"Import réussi: {', '.join(saved)}", log_text)
    else:
        log_text = append_log("Aucun fichier importé.", log_text)
    return gr.update(choices=refs, value=value), log_text


def handle_adjust(text: str, target_seconds: float | None, log_text: str | None):
    target = float(target_seconds) if target_seconds else 0.0
    result = adjust_text_to_duration(text, target)
    info = f"Durée estimée: {result.estimated_duration:.1f}s"
    if result.target_duration > 0:
        info += f" / cible {result.target_duration:.1f}s"
    if result.warning:
        info += f"\n⚠️ {result.warning}"
    log_text = append_ui_log("Suggestion de durée calculée.", log_text)
    return result.text, info, log_text


def apply_adjusted(preview_text: str) -> str:
    return preview_text


def update_clean_preview(text: str) -> str:
    clean = render_clean_text(text)
    return clean




def _coerce_float(val, default=0.0):
    try:
        return float(val)
    except (TypeError, ValueError):
        return float(default)


def _coerce_int(val, default=0):
    try:
        return int(float(val))
    except (TypeError, ValueError):
        return int(default)


def update_estimated_duration(text: str) -> str:
    est = estimate_duration(text)
    return f"Durée estimée: {est:.1f}s"


def format_adjustment_log(changes: list[str], enabled: bool) -> str:
    if not enabled:
        return ""
    if not changes:
        return "Aucune correction."
    ordered: list[str] = []
    ordered.extend([c for c in changes if c.startswith("paste_norm_applied:")])
    ordered.extend([c for c in changes if c.startswith("paste_norm_counts:")])
    ordered.extend([c for c in changes if c.startswith("sigle_undot:")])
    ordered.extend([c for c in changes if c.startswith("lexicon_hit:")])
    ordered.extend([c for c in changes if c.startswith("sigle_auto:")])
    lines = "\n".join(f"- {entry}" for entry in ordered)
    return f"**Corrections appliquées**\n{lines}"


def summarize_adjustment_changes(changes: list[str], log_text: str | None, verbose_logs: bool) -> str | None:
    if not changes:
        return log_text
    paste_entry = next((c for c in changes if c.startswith("paste_norm_applied:")), None)
    counts_entry = next((c for c in changes if c.startswith("paste_norm_counts:")), None)
    if paste_entry:
        log_text = append_ui_log(paste_entry.replace(": ", "="), log_text, verbose=True, enabled=verbose_logs)
    if counts_entry:
        log_text = append_ui_log(counts_entry.replace(": ", "="), log_text, verbose=True, enabled=verbose_logs)

    def _collect(prefix: str) -> list[str]:
        return [c[len(prefix) + 1 :].strip() for c in changes if c.startswith(f"{prefix}:")]

    for prefix in ("sigle_undot", "lexicon_hit", "sigle_auto"):
        items = _collect(prefix)
        if items:
            examples = "; ".join(items[:3])
            log_text = append_ui_log(
                f"{prefix}_count={len(items)} examples={examples}",
                log_text,
                verbose=True,
                enabled=verbose_logs,
            )
    return log_text


def handle_text_adjustment(
    text: str,
    auto_adjust: bool,
    show_adjust_log: bool,
):
    persist_state(
        {
            "last_auto_adjust": bool(auto_adjust),
            "last_show_adjust_log": bool(show_adjust_log),
        }
    )
    if auto_adjust:
        adjusted_text, changes = prepare_adjusted_text(text or "", LEXIQUE_PATH)
    else:
        adjusted_text, changes = text or "", []
    clean_preview = render_clean_text(adjusted_text)
    duration = update_estimated_duration(adjusted_text)
    log_md = format_adjustment_log(changes, show_adjust_log and auto_adjust)
    return adjusted_text, clean_preview, duration, log_md


def handle_load_preset(
    preset_name: str | None,
    log_text: str | None,
):
    output_count = _LOAD_PRESET_OUTPUT_COUNT or (25 + len(all_param_keys()))
    outputs = [gr.update() for _ in range(output_count)]
    name_update = gr.update()
    if not preset_name:
        log_text = append_log("Sélectionnez un preset à charger.", log_text)
        outputs[-2:] = [name_update, log_text]
        assert len(outputs) == output_count
        return tuple(outputs)

    data = load_preset(preset_name)
    if not data:
        log_text = append_log(f"Preset introuvable: {preset_name}", log_text)
        outputs[-2:] = [name_update, log_text]
        assert len(outputs) == output_count
        return tuple(outputs)

    refs = list_refs()
    ref_value = data.get("ref_name")
    if ref_value not in refs:
        ref_value = None

    engine_id = data.get("tts_engine") or "chatterbox"
    language, voice_id, engine_params = load_engine_config(data, engine_id)
    backend = get_backend(engine_id)
    if backend is None:
        engine_id = "chatterbox"
        backend = get_backend(engine_id)
    engine_params = collect_engine_params(engine_id, engine_params)
    voice_id = engine_params.get("voice_id") or voice_id
    chatterbox_mode = engine_params.get("chatterbox_mode", "fr_finetune")
    voices = list_piper_voices() if engine_id == "piper" else (backend.list_voices() if backend else [])
    if voices and engine_params.get("voice_id") is None:
        engine_params["voice_id"] = voices[0].id
    final_voice, voice_label_update, voice_fallback = build_voice_label_update(
        engine_id, backend, voice_id
    )
    final_lang, did_fallback, lang_update, lang_locked_update = language_ui_updates(
        engine_id, backend, chatterbox_mode, language, final_voice
    )
    supports_speed = bool(
        engine_id == "piper"
        and final_voice
        and piper_voice_supports_length_scale(final_voice)
    )
    context = {
        "chatterbox_mode": chatterbox_mode,
        "supports_ref_audio": backend.supports_ref_audio if backend else False,
        "uses_internal_voices": backend.uses_internal_voices if backend else False,
        "voice_count": len(voices),
        "piper_supports_speed": supports_speed,
    }
    ref_note_update = gr.update(value=backend_ref_note(backend))
    warning_md = language_warning(engine_id, language, final_lang, did_fallback)
    if voice_fallback:
        warning_md = "\n".join(filter(None, [warning_md, voice_fallback]))
    if engine_id == "xtts":
        xtts_status = backend_status("xtts")
        if xtts_status.get("installed") and not xtts_status.get("model_downloaded", True):
            warning_md = "\n".join(
                filter(None, [warning_md, "XTTS: poids non préchargés, téléchargement au premier usage."])
            )
    warning_update = gr.update(value=warning_md, visible=bool(warning_md))
    piper_status_update = gr.update(
        value=piper_voice_status_text(voices),
        visible=engine_id == "piper",
    )
    piper_refresh_update = gr.update(visible=engine_id == "piper")
    piper_install_update = gr.update(visible=engine_id == "piper")
    piper_catalog_update = gr.update(visible=engine_id == "piper")
    piper_speed_note = piper_speed_note_update(engine_id, final_voice, supports_speed)
    param_updates = build_param_updates(engine_id, engine_params, context)
    xtts_split_note_update = gr.update(
        value="Segmentation: XTTS native (recommandée)",
        visible=engine_id == "xtts",
    )
    inter_chunk_gap_ms = data.get("inter_chunk_gap_ms")
    if inter_chunk_gap_ms is None:
        inter_chunk_gap_ms = 120
    if engine_id == "chatterbox":
        inter_chunk_gap_update = gr.update(value=int(inter_chunk_gap_ms), visible=True)
        inter_chunk_gap_help = gr.update(visible=True)
    else:
        inter_chunk_gap_update = gr.update(value=0, visible=False)
        inter_chunk_gap_help = gr.update(visible=False)
    idx = 0

    def put(value):
        nonlocal idx
        if idx < output_count:
            outputs[idx] = value
        idx += 1

    put(gr.update(value=ref_value, visible=bool(backend and backend.supports_ref_audio)))
    put(ref_note_update)
    put(gr.update(value=data.get("out_dir") or str(DEFAULT_OUTPUT_DIR)))
    put(gr.update(value=data.get("user_filename", "")))
    put(gr.update(value=bool(data.get("add_timestamp", True))))
    put(gr.update(value=bool(data.get("include_model_name", False))))
    put(gr.update(value=engine_id))
    put(lang_update)
    put(lang_locked_update)
    put(gr.update(value=engine_status_markdown(engine_id)))
    put(gr.update(visible=not backend_status(engine_id).get("installed")))
    put(gr.update(visible=backend_status(engine_id).get("installed") and engine_id != "chatterbox"))
    put(gr.update(interactive=backend_status(engine_id).get("installed")))
    put(voice_label_update)
    put(piper_status_update)
    put(piper_refresh_update)
    put(piper_install_update)
    put(piper_catalog_update)
    put(piper_speed_note)
    put(warning_update)
    put(xtts_split_note_update)
    put(inter_chunk_gap_update)
    put(inter_chunk_gap_help)
    for update in param_updates:
        put(update)
    put(gr.update(value=bool(data.get("verbose_logs", False))))
    put(gr.update(value=preset_name))
    name_update = gr.update(value=preset_name)
    persist_state(
        {
            "last_preset": preset_name,
            "last_tts_engine": engine_id,
            "last_include_model_name": bool(data.get("include_model_name", False)),
            "inter_chunk_gap_ms": int(inter_chunk_gap_ms),
        }
    )
    persist_engine_state(engine_id, language=final_lang, voice_id=final_voice, params=engine_params)
    log_text = append_log(f"Preset chargé: {preset_name}", log_text)
    outputs[-2:] = [name_update, log_text]
    assert len(outputs) == output_count
    return tuple(outputs)


def handle_save_preset(
    preset_name: str | None,
    ref_name: str | None,
    out_dir: str | None,
    user_filename: str | None,
    add_timestamp: bool,
    include_model_name: bool,
    inter_chunk_gap_ms: float,
    tts_language: str | None,
    tts_engine: str,
    verbose_logs: bool,
    log_text: str | None,
    *param_values,
):
    if not preset_name:
        log_text = append_log("Nom de preset requis.", log_text)
        return gr.update(), gr.update(), log_text

    param_values_map = dict(zip(all_param_keys(), param_values))
    engine_params = collect_engine_params(tts_engine, param_values_map)
    voice_id = engine_params.get("voice_id")
    data = {
        "ref_name": ref_name,
        "out_dir": out_dir,
        "user_filename": user_filename or "",
        "add_timestamp": bool(add_timestamp),
        "tts_engine": str(tts_engine),
        "include_model_name": bool(include_model_name),
        "inter_chunk_gap_ms": int(inter_chunk_gap_ms),
        "engines": {
            str(tts_engine): {
                "language": str(tts_language or "fr-FR"),
                "voice_id": voice_id,
                "params": engine_params,
            }
        },
        "verbose_logs": bool(verbose_logs),
    }

    try:
        preset_slug = save_preset(preset_name, data)
    except ValueError as exc:
        log_text = append_log(f"Nom de preset invalide: {exc}", log_text)
        return gr.update(), gr.update(), log_text

    choices = list_presets()
    persist_state({"last_preset": preset_slug})
    log_text = append_log(f"Preset sauvegardé: {preset_slug}", log_text)
    return (
        gr.update(choices=choices, value=preset_slug),
        gr.update(value=preset_slug),
        log_text,
    )


def handle_delete_preset(
    preset_name: str | None,
    log_text: str | None,
):
    if not preset_name:
        log_text = append_log("Aucun preset à supprimer.", log_text)
        return gr.update(), gr.update(), log_text

    try:
        delete_preset(preset_name)
    except ValueError as exc:
        log_text = append_log(f"Suppression impossible: {exc}", log_text)
        return gr.update(), gr.update(), log_text

    choices = list_presets()
    log_text = append_log(f"Preset supprimé: {preset_name}", log_text)
    persist_state({"last_preset": None})
    return (
        gr.update(choices=choices, value=None),
        gr.update(value=""),
        log_text,
    )


def handle_choose_output(current_path: str | None, log_text: str | None):
    base = current_path or str(DEFAULT_OUTPUT_DIR)
    chosen = mac_choose_folder(base)
    if not chosen:
        log_text = append_ui_log("Sélection dossier annulée.", log_text)
        return gr.update(value=current_path), log_text
    persist_state({"last_out_dir": chosen})
    log_text = append_ui_log(f"Dossier sortie choisi: {chosen}", log_text)
    return gr.update(value=chosen), log_text


def handle_toggle_verbose_logs(verbose: bool, log_text: str | None):
    set_verbosity(bool(verbose))
    persist_state({"last_verbose_logs": bool(verbose)})
    log_text = append_ui_log("Verbosity terminal mise à jour.", log_text)
    return log_text


def handle_engine_change(
    engine_id: str,
    language: str | None,
    chatterbox_mode: str,
):
    backend = get_backend(engine_id)
    if backend is None:
        engine_id = "chatterbox"
        backend = get_backend(engine_id)
    state_data = load_state()
    state_language, state_voice, state_params = load_engine_config(state_data, engine_id)
    engine_params = collect_engine_params(engine_id, state_params)
    if chatterbox_mode:
        engine_params["chatterbox_mode"] = chatterbox_mode
    chatterbox_mode = engine_params.get("chatterbox_mode", "fr_finetune")
    voices = list_piper_voices() if engine_id == "piper" else (backend.list_voices() if backend else [])
    if voices and engine_params.get("voice_id") is None:
        engine_params["voice_id"] = voices[0].id
    state_voice = engine_params.get("voice_id") or state_voice
    coerced_language = language or state_language or "fr-FR"
    final_voice, voice_label_update, voice_fallback = build_voice_label_update(
        engine_id, backend, state_voice
    )
    final_lang, did_fallback, lang_update, lang_locked_update = language_ui_updates(
        engine_id, backend, chatterbox_mode, coerced_language, final_voice
    )
    ref_note_update = gr.update(value=backend_ref_note(backend))
    supports_speed = bool(
        engine_id == "piper"
        and final_voice
        and piper_voice_supports_length_scale(final_voice)
    )
    context = {
        "chatterbox_mode": chatterbox_mode,
        "supports_ref_audio": backend.supports_ref_audio if backend else False,
        "uses_internal_voices": backend.uses_internal_voices if backend else False,
        "voice_count": len(voices),
        "piper_supports_speed": supports_speed,
    }
    param_updates = build_param_updates(engine_id, engine_params, context)
    warning_md = language_warning(engine_id, coerced_language, final_lang, did_fallback)
    if voice_fallback:
        warning_md = "\n".join(filter(None, [warning_md, voice_fallback]))
    if engine_id == "piper" and not voices:
        warning_md = "\n".join(
            filter(None, [warning_md, "Aucune voix Piper installée. Installez une voix FR recommandée."])
        )
    if engine_id == "xtts":
        xtts_status = backend_status("xtts")
        warning_md = "\n".join(
            filter(
                None,
                [
                    warning_md,
                    "XTTS: usage non commercial (licence CPML).",
                    "XTTS: CPU forcé sur macOS (limitation torchaudio/MPS)."
                    if platform.system() == "Darwin" and platform.machine() in {"arm64", "aarch64"}
                    else "",
                    "XTTS: poids non préchargés, téléchargement au premier usage."
                    if xtts_status.get("installed") and not xtts_status.get("model_downloaded", True)
                    else "",
                ],
            )
        )
    inter_chunk_gap_ms = state_data.get("inter_chunk_gap_ms")
    if inter_chunk_gap_ms is None:
        inter_chunk_gap_ms = 120
    if engine_id == "chatterbox":
        inter_chunk_gap_update = gr.update(value=int(inter_chunk_gap_ms), visible=True)
        inter_chunk_gap_help = gr.update(visible=True)
    else:
        inter_chunk_gap_update = gr.update(value=0, visible=False)
        inter_chunk_gap_help = gr.update(visible=False)
    warning_update = gr.update(value=warning_md, visible=bool(warning_md))
    piper_status_update = gr.update(
        value=piper_voice_status_text(voices),
        visible=engine_id == "piper",
    )
    piper_refresh_update = gr.update(visible=engine_id == "piper")
    piper_install_update = gr.update(visible=engine_id == "piper")
    piper_catalog_update = gr.update(visible=engine_id == "piper")
    piper_speed_note = piper_speed_note_update(engine_id, final_voice, supports_speed)
    xtts_split_note_update = gr.update(
        value="Segmentation: XTTS native (recommandée)",
        visible=engine_id == "xtts",
    )
    persist_state(
        {
            "last_tts_engine": engine_id,
        }
    )
    persist_engine_state(engine_id, language=final_lang, voice_id=final_voice, params=engine_params)
    return (
        lang_update,
        lang_locked_update,
        gr.update(visible=bool(backend and backend.supports_ref_audio)),
        ref_note_update,
        *engine_status_updates(engine_id),
        voice_label_update,
        piper_status_update,
        piper_refresh_update,
        piper_install_update,
        piper_catalog_update,
        piper_speed_note,
        warning_update,
        xtts_split_note_update,
        inter_chunk_gap_update,
        inter_chunk_gap_help,
        *param_updates,
    )


def handle_voice_change(
    engine_id: str,
    voice_id: str | None,
    language: str | None,
    chatterbox_mode: str,
    *param_values,
):
    backend = get_backend(engine_id)
    if backend is None:
        engine_id = "chatterbox"
        backend = get_backend(engine_id)
    param_values_map = dict(zip(all_param_keys(), param_values))
    engine_params = collect_engine_params(engine_id, param_values_map)
    if voice_id:
        engine_params["voice_id"] = voice_id
    chatterbox_mode = engine_params.get("chatterbox_mode", chatterbox_mode or "fr_finetune")
    final_voice, voice_label_update, voice_fallback = build_voice_label_update(
        engine_id, backend, voice_id
    )
    coerced_language = language or "fr-FR"
    final_lang, did_fallback, lang_update, lang_locked_update = language_ui_updates(
        engine_id, backend, chatterbox_mode, coerced_language, final_voice
    )
    supports_speed = bool(
        engine_id == "piper"
        and final_voice
        and piper_voice_supports_length_scale(final_voice)
    )
    context = {
        "chatterbox_mode": chatterbox_mode,
        "supports_ref_audio": backend.supports_ref_audio if backend else False,
        "uses_internal_voices": backend.uses_internal_voices if backend else False,
        "voice_count": len(backend.list_voices()) if backend else 0,
        "piper_supports_speed": supports_speed,
    }
    param_updates = build_param_updates(engine_id, engine_params, context)
    warning_md = language_warning(engine_id, coerced_language, final_lang, did_fallback)
    if voice_fallback:
        warning_md = "\n".join(filter(None, [warning_md, voice_fallback]))
    warning_update = gr.update(value=warning_md, visible=bool(warning_md))
    speed_note_update = piper_speed_note_update(engine_id, final_voice, supports_speed)
    return (
        lang_update,
        lang_locked_update,
        voice_label_update,
        speed_note_update,
        warning_update,
        *param_updates,
    )


def handle_language_change(
    engine_id: str,
    language: str | None,
    chatterbox_mode: str,
):
    backend = get_backend(engine_id)
    if backend is None:
        engine_id = "chatterbox"
        backend = get_backend(engine_id)
    engine_params = collect_engine_params(engine_id, {"chatterbox_mode": chatterbox_mode})
    chatterbox_mode = engine_params.get("chatterbox_mode", "fr_finetune")
    final_lang, did_fallback, lang_update, lang_locked_update = language_ui_updates(
        engine_id, backend, chatterbox_mode, language or "fr-FR", None
    )
    param_updates = [gr.update() for _ in all_param_keys()]
    warning_md = language_warning(engine_id, language or "fr-FR", final_lang, did_fallback)
    warning_update = gr.update(value=warning_md, visible=bool(warning_md))
    persist_engine_state(engine_id, language=final_lang)
    return lang_update, lang_locked_update, warning_update, *param_updates


def handle_chatterbox_mode_change(
    chatterbox_mode: str,
    language: str | None,
):
    coerced_mode = chatterbox_mode or "fr_finetune"
    backend = get_backend("chatterbox")
    engine_params = collect_engine_params("chatterbox", {"chatterbox_mode": coerced_mode})
    final_lang, did_fallback, lang_update, lang_locked_update = language_ui_updates(
        "chatterbox", backend, coerced_mode, language or "fr-FR", None
    )
    context = {
        "chatterbox_mode": coerced_mode,
        "supports_ref_audio": backend.supports_ref_audio if backend else False,
        "uses_internal_voices": backend.uses_internal_voices if backend else False,
        "voice_count": len(backend.list_voices()) if backend else 0,
    }
    param_updates = build_param_visibility_updates("chatterbox", context)
    warning_md = language_warning("chatterbox", language or "fr-FR", final_lang, did_fallback)
    warning_update = gr.update(value=warning_md, visible=bool(warning_md))
    _, _, state_params = load_engine_config(load_state(), "chatterbox")
    merged_params = dict(state_params)
    merged_params.update(engine_params)
    persist_engine_state("chatterbox", language=final_lang, params=merged_params)
    return lang_update, lang_locked_update, warning_update, *param_updates


def handle_install_backend(engine_id: str):
    ok, logs = run_install(engine_id)
    status_md, install_btn, uninstall_btn, generate_btn = engine_status_updates(engine_id)
    log_text = "\n".join(logs)
    if not ok:
        status_md = gr.update(value=f"{engine_status_markdown(engine_id)}\n\n⚠️ Installation incomplète.")
    return status_md, install_btn, uninstall_btn, generate_btn, gr.update(value=log_text)


def handle_uninstall_backend(engine_id: str):
    from backend_install.paths import venv_dir
    import shutil

    target = venv_dir(engine_id)
    if target.exists():
        shutil.rmtree(target)
    status_md, install_btn, uninstall_btn, generate_btn = engine_status_updates(engine_id)
    log_text = f"Désinstallé: {engine_id}"
    return status_md, install_btn, uninstall_btn, generate_btn, gr.update(value=log_text)


def _build_chunk_preview(chunks) -> str:
    lines = []
    for idx, chunk_info in enumerate(chunks, start=1):
        warn = f" warnings={','.join(chunk_info.warnings)}" if chunk_info.warnings else ""
        lines.append(
            f"[{idx}] words={chunk_info.word_count} est={chunk_info.estimated_duration:.1f}s "
            f"reason={chunk_info.reason}{warn}"
        )
        chunk_text = render_clean_text_from_segments(chunk_info.segments).strip()
        if chunk_text:
            lines.append(f"text: {chunk_text}")
    return "\n".join(lines)


def _single_chunk(text: str, *, reason: str) -> ChunkInfo | None:
    clean = render_clean_text(text).strip()
    if not clean:
        return None
    sentence_count = len(re.findall(r"[.!?]", clean))
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


def _apply_direction_chunking(
    *,
    direction_enabled: bool,
    direction_source: str | None,
    direction_snapshot_text: str | None,
    tts_ready_text: str,
    log_text: str | None,
) -> tuple[list[ChunkInfo], str, dict | None, str | None]:
    direction_source = str(direction_source or "final")
    snapshot = direction_snapshot_text or ""
    if not direction_enabled:
        single = _single_chunk(tts_ready_text, reason="single")
        chunks = [single] if single else []
        return chunks, "single", None, log_text
    chunks, marker_count = parse_manual_chunks(snapshot, marker=MANUAL_CHUNK_MARKER)
    if marker_count > 0 and chunks:
        direction_meta = {
            "source": direction_source,
            "markers_count": marker_count,
        }
        return chunks, "manual_marker", direction_meta, log_text
    log_text = append_ui_log("No markers → single chunk", log_text)
    if not snapshot.strip():
        snapshot = tts_ready_text
    single = _single_chunk(snapshot, reason="manual_single")
    chunks = [single] if single else []
    direction_meta = {
        "source": direction_source,
        "markers_count": 0,
    }
    return chunks, "manual_single", direction_meta, log_text


def handle_generate(
    text: str,
    adjusted_text: str,
    auto_adjust: bool,
    ref_name: str | None,
    out_dir: str | None,
    user_filename: str | None,
    add_timestamp: bool,
    include_model_name: bool,
    tts_engine: str,
    tts_language: str | None,
    inter_chunk_gap_ms: float,
    verbose_logs: bool,
    direction_enabled: bool,
    direction_source: str | None,
    direction_snapshot_text: str | None,
    log_text: str | None,
    *param_values,
):
    session_state = {"dir": None, "json": None}

    def _with_session(*values):
        return (*values, session_state)

    def _early_generate_error(message: str):
        log = append_ui_log(message, log_text)
        return (
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            log,
            gr.update(),
            gr.update(),
            gr.update(),
            session_state,
        )

    if not isinstance(tts_engine, str):
        return _early_generate_error("Sélection moteur invalide")
    valid_engines = {backend.id for backend in list_backends()}
    if tts_engine not in valid_engines:
        return _early_generate_error("Sélection moteur invalide")
    if not text or not text.strip():
        return _with_session(
            "",
            None,
            None,
            "",
            append_log("Erreur: texte vide.", log_text),
            None,
            "",
            "",
        )

    log_text = append_ui_log("Initialisation de la génération...", log_text)
    if tts_engine == "xtts":
        xtts_log_path = str(XTTS_ASSETS_DIR / ".tmp" / "xtts_runner.log")
        log_text = append_ui_log(
            f"XTTS logs: {xtts_log_path} (tail -f pendant le download)",
            log_text,
        )
        if platform.system() == "Darwin" and platform.machine() in {"arm64", "aarch64"}:
            log_text = append_ui_log(
                "XTTS forcé en CPU sur macOS (limitation torchaudio/MPS ComplexFloat).",
                log_text,
            )

    if auto_adjust:
        adjusted_text, changes = prepare_adjusted_text(text or "", LEXIQUE_PATH)
        log_text = summarize_adjustment_changes(changes, log_text, verbose_logs)
    else:
        adjusted_text = adjusted_text or text or ""

    param_values_map = dict(zip(all_param_keys(), param_values))
    engine_params = collect_engine_params(tts_engine, param_values_map)
    voice_id = engine_params.get("voice_id")
    tts_language = tts_language or "fr-FR"
    requested_language = tts_language
    chatterbox_mode = engine_params.get("chatterbox_mode", "fr_finetune")
    normalized_text = normalize_text(adjusted_text)
    if normalized_text != adjusted_text and verbose_logs:
        before = len(re.findall(r"\bII\b", adjusted_text))
        after = len(re.findall(r"\bII\b", normalized_text))
        ii_fix = max(before - after, 0)
        detail = f"II->Il x{ii_fix}" if ii_fix else "whitespace/ponctuation"
        log_text = append_ui_log(f"Normalisation: {detail}", log_text, verbose=True, enabled=True)

    output_dir = ensure_output_dir(out_dir)
    now = dt.datetime.now()
    engine_slug = get_engine_slug(tts_engine, {"chatterbox_mode": chatterbox_mode})
    session_slug = build_session_slug(normalized_text, user_filename)
    clean_preview = render_clean_text(normalized_text)
    if clean_preview:
        log_text = append_ui_log("Texte interprété prêt.", log_text)

    estimate = estimate_duration(normalized_text)
    if estimate >= 35:
        log_text = append_ui_log("⚠️ Texte long détecté.", log_text)

    backend = get_backend(tts_engine)
    if backend is None:
        log_text = append_ui_log(f"Backend introuvable: {tts_engine}", log_text)
        return _with_session(adjusted_text, None, "", "", log_text, None, "", "")
    if not backend.is_available():
        reason = backend.unavailable_reason() or "Dépendances manquantes."
        log_text = append_ui_log(f"Backend indisponible: {tts_engine}. {reason}", log_text)
        return _with_session(adjusted_text, None, "", "", log_text, None, "", "")

    supported = supported_languages_for(tts_engine, backend, chatterbox_mode)
    tts_language, did_fallback = coerce_language(
        tts_language,
        supported,
        backend.default_language() if backend else None,
    )
    if did_fallback:
        log_text = append_ui_log(
            f"tts_language_fallback requested={requested_language} -> {tts_language} (engine={tts_engine})",
            log_text,
        )
    backend_language = backend.map_language(tts_language)
    log_text = append_ui_log(f"tts_backend={backend.id}", log_text, verbose=False, enabled=True)
    log_text = append_ui_log(
        f"backend_lang={backend_language}",
        log_text,
        verbose=False,
        enabled=True,
    )
    log_text = append_ui_log(
        f"supports_ref={backend.supports_ref_audio}",
        log_text,
        verbose=False,
        enabled=True,
    )

    audio_prompt = None
    if ref_name and backend.supports_ref_audio:
        try:
            audio_prompt = resolve_ref_path(ref_name)
        except FileNotFoundError:
            log_text = append_ui_log(f"Référence introuvable: {ref_name}", log_text)
            return _with_session(
                adjusted_text,
                None,
                None,
                "",
                log_text,
                None,
                "",
                "",
            )
    elif ref_name and not backend.supports_ref_audio:
        log_text = append_ui_log("Référence ignorée (backend sans voice ref).", log_text)
    if backend.id == "xtts" and not audio_prompt:
        log_text = append_ui_log("XTTS nécessite une référence vocale.", log_text)
        return _with_session(adjusted_text, None, "", "", log_text, None, "", "")

    warnings = backend.validate_config(
        {
            "language": tts_language,
            "voice_ref": audio_prompt,
        }
    )
    for warning in warnings:
        log_text = append_ui_log(f"backend_warning={warning}", log_text)

    voices = backend.list_voices()
    voice_ids = [voice.id for voice in voices]
    if voice_id and voice_ids and voice_id not in voice_ids:
        log_text = append_ui_log(
            f"tts_voice_fallback requested={voice_id} -> {voice_ids[0]} (engine={tts_engine})",
            log_text,
        )
        voice_id = voice_ids[0]
    if backend.uses_internal_voices and not voice_id:
        if voices:
            voice_id = voices[0].id
        else:
            log_text = append_ui_log(
                "Aucune voix Piper installée. Installez une voix FR recommandée.",
                log_text,
            )
            return _with_session(adjusted_text, None, "", "", log_text, None, "", "")
    if backend.uses_internal_voices and voice_id:
        for voice in voices:
            if voice.id == voice_id and voice.lang_codes:
                tts_language = voice.lang_codes[0]
                break
    supports_speed = bool(
        backend.id == "piper"
        and voice_id
        and piper_voice_supports_length_scale(voice_id)
    )

    tts_model_mode = chatterbox_mode if tts_engine == "chatterbox" else "fr_finetune"
    if tts_engine == "chatterbox" and tts_model_mode == "fr_finetune":
        tts_language = "fr-FR"
    multilang_cfg_weight = engine_params.get("multilang_cfg_weight", 0.5)
    cfg_weight = engine_params.get("cfg_weight", 0.6)
    effective_cfg = cfg_weight if tts_model_mode == "fr_finetune" else float(multilang_cfg_weight)
    log_text = append_ui_log(f"voice_mode={tts_model_mode}", log_text, verbose=False, enabled=True)
    log_text = append_ui_log(f"cfg_weight={effective_cfg}", log_text, verbose=False, enabled=True)

    chunks, chunk_mode, direction_meta, log_text = _apply_direction_chunking(
        direction_enabled=bool(direction_enabled),
        direction_source=direction_source,
        direction_snapshot_text=direction_snapshot_text,
        tts_ready_text=normalized_text,
        log_text=log_text,
    )
    chunk_preview_text = _build_chunk_preview(chunks)
    if chunks:
        for chunk_info in chunks:
            if chunk_info.estimated_duration > DEFAULT_MAX_EST_SECONDS_PER_CHUNK:
                log_text = append_ui_log(
                    f"⚠️ Chunk long détecté: {chunk_info.estimated_duration:.1f}s (max_est={DEFAULT_MAX_EST_SECONDS_PER_CHUNK}s)",
                    log_text,
                )

    if not chunks:
        return _with_session(
            adjusted_text,
            None,
            "",
            chunk_preview_text,
            append_ui_log("Aucun chunk généré.", log_text),
            None,
            "",
            "",
        )

    job_state = _get_job_state()
    if job_state.get("job_running") and job_state.get("current_proc"):
        _terminate_proc(job_state.get("current_proc"))
        _cleanup_tmp(job_state.get("current_tmp_path"))
        _reset_job_state()
        log_text = append_ui_log("Job précédent interrompu.", log_text)

    tmp_path = TMP_DIR / f"{uuid.uuid4().hex}.tmp.wav"
    payload = {
        "tts_backend": backend.id,
        "script": normalized_text,
        "chunks": _serialize_chunks(chunks),
        "voice_ref_path": audio_prompt,
        "out_path": str(tmp_path),
        "lang": str(backend_language or tts_language or "fr-FR"),
        "engine_params": {},
        "target_sr": 24000,
    }
    effective_gap_ms = int(inter_chunk_gap_ms or 0)
    if tts_engine != "chatterbox":
        effective_gap_ms = 0
    payload["inter_chunk_gap_ms"] = effective_gap_ms
    engine_params_payload = dict(engine_params)
    if tts_engine == "chatterbox":
        engine_params_payload.update(
            {
                "tts_model_mode": str(tts_model_mode),
                "exaggeration": engine_params.get("exaggeration", 0.5),
                "cfg_weight": cfg_weight,
                "temperature": engine_params.get("temperature", 0.5),
                "repetition_penalty": engine_params.get("repetition_penalty", 1.35),
            }
        )
    if backend.id == "piper" and not supports_speed:
        engine_params_payload.pop("speed", None)
        if voice_id:
            log_text = append_ui_log(
                "Vitesse non supportée par cette voix (paramètre ignoré).",
                log_text,
            )
    if backend.uses_internal_voices and voice_id:
        engine_params_payload["voice"] = voice_id
    payload["engine_params"] = engine_params_payload
    log_text = append_ui_log(f"params_effectifs={engine_params_payload}", log_text, verbose=True, enabled=verbose_logs)
    ctx = mp.get_context("spawn")
    result_queue: mp.Queue = ctx.Queue()
    proc = ctx.Process(target=_generate_longform_worker, args=(payload, result_queue))
    log_text = append_log("Synthèse en cours...", log_text)
    proc.start()
    _set_job_state(
        current_proc=proc,
        current_tmp_path=str(tmp_path),
        current_final_path=None,
        job_running=True,
    )

    result = None
    while proc.is_alive():
        try:
            result = result_queue.get_nowait()
            break
        except queue.Empty:
            time.sleep(0.1)
    proc.join()
    if result is None:
        try:
            result = result_queue.get_nowait()
        except queue.Empty:
            result = None

    if not result or result.get("status") != "ok":
        _cleanup_tmp(str(tmp_path))
        _reset_job_state()
        if result and result.get("status") == "unavailable":
            log_text = append_ui_log(f"Backend indisponible: {result.get('error')}", log_text)
        elif result and result.get("status") == "error":
            log_text = append_ui_log(f"Erreur TTS: {result.get('error')}", log_text)
        else:
            log_text = append_ui_log("Annulé.", log_text)
        return _with_session(
            adjusted_text,
            None,
            "",
            chunk_preview_text,
            log_text,
            None,
            "",
            "",
        )

    meta = result.get("meta", {})
    backend_meta = meta.get("backend_meta") or {}
    gap_ms = meta.get("inter_chunk_gap_ms")
    if gap_ms is not None:
        applied = 1 if meta.get("inter_chunk_gap_applied") else 0
        gap_engine = meta.get("inter_chunk_gap_engine")
        gap_chunks = meta.get("inter_chunk_gap_chunks")
        log_text = append_ui_log(
            f"Montage inter-chunk: inter_chunk_gap_ms={gap_ms} applied={applied} "
            f"engine={gap_engine} chunks={gap_chunks}",
            log_text,
        )
        if gap_chunks == 1 and int(gap_ms) > 0:
            log_text = append_ui_log("single chunk → no gap applied", log_text)
    if meta.get("piper_voice"):
        log_text = append_ui_log(f"piper_voice={meta.get('piper_voice')}", log_text)
    if meta.get("piper_model_path"):
        log_text = append_ui_log(f"piper_model_path={meta.get('piper_model_path')}", log_text)
    if tts_engine == "xtts":
        if backend_meta.get("device"):
            log_text = append_ui_log(f"xtts_device={backend_meta.get('device')}", log_text)
        if backend_meta.get("model_id"):
            log_text = append_ui_log(f"xtts_model_id={backend_meta.get('model_id')}", log_text)
        if backend_meta.get("forced_cpu"):
            log_text = append_ui_log(
                "XTTS forced to CPU on macOS due to torchaudio complex dtype on MPS.",
                log_text,
            )
        if verbose_logs and backend_meta.get("xtts_segments"):
            segments = " | ".join(backend_meta.get("xtts_segments") or [])
            log_text = append_ui_log(f"xtts_segments={segments}", log_text, verbose=True, enabled=True)
        backend_logs = meta.get("backend_logs") or []
        if backend_logs:
            logs_text = "\n".join(backend_logs)
            if len(logs_text) > 2000:
                logs_text = f"{logs_text[:2000]}...\n(truncated)"
            log_text = append_ui_log(f"xtts_logs:\n{logs_text}", log_text)
        if backend_meta.get("log_path"):
            log_text = append_ui_log(f"xtts_log_path={backend_meta.get('log_path')}", log_text)
    if verbose_logs and meta.get("backend_cmd"):
        log_text = append_ui_log(f"backend_cmd={meta.get('backend_cmd')}", log_text, verbose=True, enabled=True)

    for idx, duration in enumerate(meta.get("durations", []), start=1):
        retry_flag = meta.get("retries", [])[idx - 1] if meta.get("retries") else False
        chunk_info = chunks[idx - 1] if idx - 1 < len(chunks) else None
        reason = chunk_info.reason if chunk_info else "n/a"
        est = chunk_info.estimated_duration if chunk_info else 0.0
        retry_note = " retry" if retry_flag else ""
        log_text = append_ui_log(
            f"Chunk {idx}/{meta.get('chunks', len(meta.get('durations', [])))} "
            f"reason={reason} est={est:.1f}s measured={duration:.2f}s{retry_note}",
            log_text,
            verbose=True,
            enabled=verbose_logs,
        )
    total_duration = meta.get("total_duration")
    if total_duration is not None:
        log_text = append_ui_log(f"Durée finale: {total_duration:.2f}s", log_text)
    if meta.get("segments_count_total") is not None:
        log_text = append_ui_log(
            f"segments_count_total={meta.get('segments_count_total')}",
            log_text,
            verbose=True,
            enabled=verbose_logs,
        )
    if meta.get("sr"):
        log_text = append_ui_log(
            f"sr_final={meta.get('sr')}",
            log_text,
            verbose=True,
            enabled=verbose_logs,
        )

    raw_path: Path | None = None
    try:
        session_dir = create_session_dir(WORK_DIR, now, session_slug)
        raw_path = get_take_path_global_raw(session_dir, "v1")
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        if tmp_path.exists():
            os.replace(tmp_path, raw_path)
        else:
            _reset_job_state()
            log_text = append_ui_log("Annulé.", log_text)
            return _with_session(
                adjusted_text,
                None,
                "",
                chunk_preview_text,
                log_text,
                None,
                "",
                "",
            )
        if tts_engine == "xtts" and backend_meta.get("xtts_segments"):
            segments = [str(seg) for seg in (backend_meta.get("xtts_segments") or [])]
            if segments:
                write_xtts_segments(
                    session_dir,
                    engine_slug=engine_slug,
                    take_id=raw_path.name,
                    segments=segments,
                    created_at=now.isoformat(timespec="seconds"),
                    segment_boundaries_samples=backend_meta.get("xtts_segment_boundaries_samples"),
                    sample_rate=backend_meta.get("xtts_sample_rate"),
                )
        artifacts = {
            "raw_global": str(Path("takes") / "global" / raw_path.name),
        }
        artifacts_list = [str(raw_path)]
        session_payload = build_session_payload(
            engine_id=tts_engine,
            engine_slug=engine_slug,
            ref_name=ref_name,
            text=normalized_text,
            editorial_text=text or "",
            tts_ready_text=normalized_text,
            prep_log_md="",
            created_at=now,
            chunks=chunks,
            chunk_mode=chunk_mode,
            direction_meta=direction_meta,
            artifacts=artifacts,
            artifacts_list=artifacts_list,
            takes={"global": ["v1"], "processed": []},
            active_take={"global": "v1"},
            active_listen="raw",
        )
        session_path = write_session_json(session_dir, session_payload)
        log_text = append_ui_log(f"Session créée: {session_path}", log_text, verbose=False, enabled=True)
        session_state = {"dir": str(session_dir), "json": str(session_path)}
    except Exception as exc:
        log_text = append_ui_log(f"Session non écrite: {exc}", log_text)
        _cleanup_tmp(str(tmp_path))
    _reset_job_state()
    persist_state(
        {
            "last_ref": ref_name,
            "last_out_dir": output_dir,
            "last_user_filename": user_filename or "",
            "last_add_timestamp": bool(add_timestamp),
            "last_include_model_name": bool(include_model_name),
            "last_verbose_logs": bool(verbose_logs),
            "last_tts_engine": str(tts_engine),
            "inter_chunk_gap_ms": int(inter_chunk_gap_ms or 0),
        }
    )
    persist_engine_state(tts_engine, language=tts_language, voice_id=voice_id, params=engine_params)
    if raw_path:
        log_text = append_ui_log(f"Fichier RAW: {raw_path}", log_text)
    return _with_session(
        adjusted_text,
        str(raw_path) if raw_path else "",
        str(raw_path) if raw_path else "",
        chunk_preview_text,
        log_text,
        None,
        "",
        "",
    )


def handle_stop(log_text: str | None):
    job_state = _get_job_state()
    proc = job_state.get("current_proc")
    if proc and proc.is_alive():
        _terminate_proc(proc)
    _cleanup_tmp(job_state.get("current_tmp_path"))
    _reset_job_state()
    if job_state.get("job_running"):
        log_text = append_ui_log("Annulé.", log_text)
    else:
        log_text = append_ui_log("Aucune génération en cours.", log_text)
    return None, "", log_text


def handle_session_texts(session_state: dict | None):
    state = dict(session_state or {})
    session_dir = state.get("dir")
    if not session_dir:
        return "", "", "", gr.update(visible=False)
    try:
        _session_path, session_data = load_session_json(session_dir)
        editorial, tts_ready, prep_log = extract_session_texts(session_data)
    except Exception:
        return "", "", "", gr.update(visible=False)
    return (
        editorial,
        tts_ready,
        prep_log,
        gr.update(visible=True),
    )


def _confirm_action(action: str, confirm_state: dict | None, log_text: str | None):
    now = time.time()
    state = dict(confirm_state or {"pending": None, "ts": 0.0})
    pending = state.get("pending")
    ts = float(state.get("ts") or 0.0)
    save_label = "Sauver"
    delete_label = "Supprimer"
    stop_label = "STOP"
    save_variant = "secondary"
    delete_variant = "secondary"
    stop_variant = "secondary"
    if pending == "sauvegarde":
        save_label = "Confirmer Sauver"
        save_variant = "stop"
    elif pending == "suppression":
        delete_label = "Confirmer Supprimer"
        delete_variant = "stop"
    elif pending == "arrêt":
        stop_label = "Confirmer STOP"
        stop_variant = "stop"
    if pending == action and (now - ts) <= 10.0:
        state = {"pending": None, "ts": 0.0}
        return (
            True,
            state,
            log_text,
            gr.update(value="Sauver", variant="secondary"),
            gr.update(value="Supprimer", variant="secondary"),
            gr.update(value="STOP", variant="secondary"),
        )
    state = {"pending": action, "ts": now}
    log_text = append_ui_log(f"Confirmez l'action ({action}) : cliquez à nouveau.", log_text)
    # Keep confirmation visible via button label + color.
    if action == "sauvegarde":
        save_label = "Confirmer Sauver"
        save_variant = "stop"
    elif action == "suppression":
        delete_label = "Confirmer Supprimer"
        delete_variant = "stop"
    elif action == "arrêt":
        stop_label = "Confirmer STOP"
        stop_variant = "stop"
    return (
        False,
        state,
        log_text,
        gr.update(value=save_label, variant=save_variant),
        gr.update(value=delete_label, variant=delete_variant),
        gr.update(value=stop_label, variant=stop_variant),
    )


def handle_save_preset_confirm(
    preset_name: str | None,
    ref_name: str | None,
    out_dir: str | None,
    user_filename: str | None,
    add_timestamp: bool,
    include_model_name: bool,
    inter_chunk_gap_ms: float,
    tts_language: str | None,
    tts_engine: str,
    verbose_logs: bool,
    confirm_state: dict | None,
    log_text: str | None,
    *param_values,
):
    (
        confirmed,
        confirm_state,
        log_text,
        save_update,
        delete_update,
        stop_update,
    ) = _confirm_action(
        "sauvegarde", confirm_state, log_text
    )
    if not confirmed:
        return (
            gr.update(),
            gr.update(),
            log_text,
            confirm_state,
            save_update,
            delete_update,
            stop_update,
        )
    dropdown_update, name_update, log_text = handle_save_preset(
        preset_name,
        ref_name,
        out_dir,
        user_filename,
        add_timestamp,
        include_model_name,
        inter_chunk_gap_ms,
        tts_language,
        tts_engine,
        verbose_logs,
        log_text,
        *param_values,
    )
    return (
        dropdown_update,
        name_update,
        log_text,
        confirm_state,
        save_update,
        delete_update,
        stop_update,
    )


def handle_delete_preset_confirm(
    preset_name: str | None,
    confirm_state: dict | None,
    log_text: str | None,
):
    (
        confirmed,
        confirm_state,
        log_text,
        save_update,
        delete_update,
        stop_update,
    ) = _confirm_action(
        "suppression", confirm_state, log_text
    )
    if not confirmed:
        return (
            gr.update(),
            gr.update(),
            log_text,
            confirm_state,
            save_update,
            delete_update,
            stop_update,
        )
    dropdown_update, name_update, log_text = handle_delete_preset(preset_name, log_text)
    return (
        dropdown_update,
        name_update,
        log_text,
        confirm_state,
        save_update,
        delete_update,
        stop_update,
    )


def handle_stop_confirm(
    confirm_state: dict | None,
    log_text: str | None,
):
    (
        confirmed,
        confirm_state,
        log_text,
        save_update,
        delete_update,
        stop_update,
    ) = _confirm_action(
        "arrêt", confirm_state, log_text
    )
    if not confirmed:
        return (
            gr.update(),
            gr.update(),
            log_text,
            confirm_state,
            save_update,
            delete_update,
            stop_update,
            gr.update(),
            gr.update(),
            gr.update(),
        )
    audio, path, log_text = handle_stop(log_text)
    return (
        audio,
        path,
        log_text,
        confirm_state,
        save_update,
        delete_update,
        stop_update,
        gr.update(),
        gr.update(),
        gr.update(),
    )


def handle_cancel_confirm(log_text: str | None):
    confirm_state = {"pending": None, "ts": 0.0}
    return (
        log_text,
        confirm_state,
        gr.update(value="Sauver", variant="secondary"),
        gr.update(value="Supprimer", variant="secondary"),
        gr.update(value="STOP", variant="secondary"),
    )


def build_ui() -> gr.Blocks:
    clean_work_dir(WORK_DIR)
    initial_refs = list_refs()
    state_data = load_state()
    ensure_default_presets()
    base_preset = load_preset("default")
    default_ref = state_data.get("last_ref") or base_preset.get("ref_name")
    if default_ref not in initial_refs:
        default_ref = initial_refs[0] if initial_refs else None
    def _state_or_preset(key: str, fallback):
        value = state_data.get(key)
        if value is not None:
            return value
        return base_preset.get(key, fallback)

    default_tts_engine = state_data.get("last_tts_engine") or _state_or_preset(
        "tts_engine", "chatterbox"
    )
    default_out_dir_value = str(DEFAULT_OUTPUT_DIR)
    default_user_filename = state_data.get("last_user_filename", "")
    default_add_timestamp = _coerce_bool(state_data.get("last_add_timestamp"), True)
    default_include_model_name = _coerce_bool(
        state_data.get("last_include_model_name")
        if "last_include_model_name" in state_data
        else base_preset.get("include_model_name"),
        False,
    )
    default_auto_adjust = _coerce_bool(state_data.get("last_auto_adjust"), True)
    default_show_adjust_log = _coerce_bool(state_data.get("last_show_adjust_log"), False)
    default_direction_enabled = _coerce_bool(
        state_data.get("direction_enabled")
        if "direction_enabled" in state_data
        else base_preset.get("direction_enabled"),
        True,
    )
    default_direction_source = str(
        state_data.get("direction_source")
        if "direction_source" in state_data
        else base_preset.get("direction_source") or "final"
    )
    default_verbose_logs = _coerce_bool(state_data.get("last_verbose_logs"), False)
    default_inter_chunk_gap_ms = state_data.get("inter_chunk_gap_ms")
    if default_inter_chunk_gap_ms is None:
        default_inter_chunk_gap_ms = base_preset.get("inter_chunk_gap_ms", 120)
    default_inter_chunk_gap_ms = int(default_inter_chunk_gap_ms)
    if default_tts_engine != "chatterbox":
        default_inter_chunk_gap_ms = 0
    default_backend_check = get_backend(default_tts_engine)
    if not default_backend_check or not default_backend_check.is_available():
        default_tts_engine = "chatterbox"
    default_backend = get_backend(default_tts_engine)
    state_language, state_voice, state_params = load_engine_config(state_data, default_tts_engine)
    preset_language, preset_voice, preset_params = load_engine_config(base_preset, default_tts_engine)
    merged_params = {}
    merged_params.update(preset_params)
    merged_params.update(state_params)
    engine_params = collect_engine_params(default_tts_engine, merged_params)
    default_chatterbox_mode = engine_params.get("chatterbox_mode", "fr_finetune")
    default_tts_language = state_language or preset_language or "fr-FR"
    if not default_tts_language:
        default_tts_language = "fr-FR"
    default_supported_languages = supported_languages_for(
        default_tts_engine, default_backend, default_chatterbox_mode
    )
    default_tts_language, _ = coerce_language(
        default_tts_language,
        default_supported_languages,
        default_backend.default_language() if default_backend else None,
    )
    show_lang_default = len(default_supported_languages) > 1 and bool(
        default_backend and default_backend.supports_multilang
    )
    default_lang_locked_text = ""
    if not show_lang_default:
        default_lang_locked_text = f"Langue : {default_tts_language} (verrouillée)"
    voices = default_backend.list_voices() if default_backend else []
    voice_ids = [voice.id for voice in voices]
    default_voice_value = state_voice or preset_voice
    if default_voice_value not in voice_ids:
        default_voice_value = voice_ids[0] if voice_ids else None
    default_voice_label_text = ""
    default_voice_label_visible = False
    if default_backend:
        if len(voice_ids) == 0 and default_tts_engine == "piper":
            default_voice_label_text = "Voix : (aucune)"
            default_voice_label_visible = True
    preset_choices = list_presets()
    default_preset = state_data.get("last_preset")
    if default_preset not in preset_choices:
        default_preset = "default"

    with gr.Blocks(
        title="Vocalie-TTS",
        css="""
        .gradio-container .prose :last-child {
          margin-bottom: 5px;
          margin-top: 5px;
          margin-left: 9px;
        }

        .gradio-container .prose h2 {
          margin-top: 6px;
        }

        .gradio-container {
          --block-radius: 14px;
          --block-border-width: 5px;
          --layout-gap: 0px;
          --form-gap-width: 7px;
          --button-border-width: 0px;
          --button-large-radius: 30px;
          --button-small-radius: 0px;
        }
        """,
    ) as demo:
        set_verbosity(default_verbose_logs)
        gr.Markdown("# 🎙️ Chatterbox TTS FR\nInterface locale pour générer des voix off expressives en français.")
        with gr.Group(elem_id="v-presets"):
            gr.Markdown("## Presets")
            with gr.Row():
                preset_dropdown = gr.Dropdown(
                    label=None,
                    choices=preset_choices,
                    value=default_preset,
                    show_label=False,
                )
                preset_name_box = gr.Textbox(
                    label="Nom preset",
                    value=default_preset or "",
                    placeholder="ex: pub-dynamique",
                )
            with gr.Row():
                load_preset_btn = gr.Button("Charger")
                save_preset_btn = gr.Button("Sauver", variant="secondary")
                delete_preset_btn = gr.Button("Supprimer", variant="secondary")
                cancel_confirm_btn = gr.Button("Annuler", size="sm")

        with gr.Group(elem_id="v-prep"):
            gr.Markdown("## Préparation")
            with gr.Group(elem_id="v-prep-text"):
                gr.Markdown("## Texte", elem_id="prep-text-title")
                text_input = gr.Textbox(
                    label=None,
                    lines=4,
                    max_lines=20,
                    placeholder="Collez votre script ici...",
                    show_label=False,
                )
                with gr.Row():
                    auto_adjust_toggle = gr.Checkbox(
                        label="Auto-ajustement",
                        value=default_auto_adjust,
                    )
                    show_adjust_log_toggle = gr.Checkbox(
                        label="Afficher log",
                        value=default_show_adjust_log,
                    )
                with gr.Row():
                    target_duration = gr.Number(
                        label="Durée cible (s)", value=None, precision=1, show_label=False
                    )
                    adjust_btn = gr.Button("Ajuster le texte")
                    apply_btn = gr.Button("Utiliser la suggestion")
                adjusted_preview = gr.Textbox(
                    label="Suggestion texte",
                    lines=3,
                    max_lines=16,
                    interactive=False,
                    placeholder="Le texte ajusté apparaîtra ici...",
                )
                adjusted_text_box = gr.Textbox(
                    label="Texte ajusté",
                    lines=3,
                    max_lines=16,
                    interactive=False,
                    placeholder="Le texte ajusté pour le TTS apparaîtra ici...",
                )
                adjust_info = gr.Markdown("Durée estimée: --")
                adjust_log_box = gr.Markdown("")
            gr.Markdown("### Voir le texte final")
            clean_text_box = gr.Textbox(
                label="Texte interprété (envoyé au TTS)",
                lines=3,
                max_lines=16,
                interactive=False,
                placeholder="Le texte sans balises apparaîtra ici...",
            )
            gr.Markdown(
                "⚠️ La ponctuation finale peut être renforcée à la synthèse, sans modifier cet aperçu."
            )
            with gr.Group(elem_id="v-prep-direction"):
                gr.Markdown("### Direction de lecture")
                direction_enabled_toggle = gr.Checkbox(
                    label="Découpage manuel (recommandé)",
                    value=default_direction_enabled,
                )
                gr.Markdown(
                    "Insérez `[[CHUNK]]` pour définir les limites envoyées au TTS.",
                )
                with gr.Row():
                    direction_source_dropdown = gr.Dropdown(
                        label="Source snapshot",
                        choices=[
                            ("original", "original"),
                            ("adjusted", "adjusted"),
                            ("final", "final"),
                        ],
                        value=default_direction_source,
                    )
                    direction_load_btn = gr.Button("Charger snapshot", size="sm")
                direction_snapshot_box = gr.Textbox(
                    label="Snapshot (modifiable)",
                    lines=6,
                    max_lines=20,
                    elem_id="direction_snapshot",
                )
                with gr.Row():
                    insert_chunk_btn = gr.Button("Insérer CHUNK", size="sm", interactive=False)
                    preview_chunks_btn = gr.Button("Voir chunks", size="sm", interactive=False)
                chunk_preview_box = gr.Textbox(
                    label="Aperçu des chunks",
                    lines=4,
                    max_lines=16,
                    interactive=False,
                    placeholder="Aperçu des chunks manuels.",
                )
            session_text_group = gr.Group(visible=False)
            with session_text_group:
                gr.Markdown("### 📝 Textes session (read-only)")
                session_editorial_box = gr.Textbox(
                    label="Texte éditorial",
                    lines=3,
                    max_lines=8,
                    interactive=False,
                )
                session_tts_ready_box = gr.Textbox(
                    label="Texte TTS-ready (envoyé au moteur)",
                    lines=3,
                    max_lines=8,
                    interactive=False,
                )
                session_prep_log_md = gr.Markdown("")
        with gr.Group(elem_id="v-gen"):
            gr.Markdown("## Génération / Régénération")
            with gr.Group(elem_id="v-gen-refs"):
                gr.Markdown("## Références vocales", elem_id="generation-refs-title")
                with gr.Row():
                    ref_dropdown = gr.Dropdown(
                        label=None,
                        choices=initial_refs,
                        value=default_ref,
                        interactive=True,
                        show_label=False,
                        visible=bool(default_backend and default_backend.supports_ref_audio),
                    )
                    refresh_btn = gr.Button("Refresh", size="sm")
                ref_support_note = gr.Markdown(
                    backend_ref_note(default_backend),
                )
                with gr.Accordion("Importer des fichiers audio", open=False):
                    upload = gr.Files(
                        label="Drag & drop",
                        file_types=list(ALLOWED_EXTENSIONS),
                        file_count="multiple",
                    )
            with gr.Group(elem_id="v-gen-model"):
                gr.Markdown("### Modèle / Langue")
                with gr.Row():
                    engine_dropdown = gr.Dropdown(
                        label="Moteur",
                        choices=backend_choices(),
                        value=default_tts_engine,
                    )
                    param_specs = _param_spec_catalog()
                    param_widgets = {}
                    default_context = {
                        "chatterbox_mode": default_chatterbox_mode,
                        "supports_ref_audio": default_backend.supports_ref_audio if default_backend else False,
                        "uses_internal_voices": default_backend.uses_internal_voices if default_backend else False,
                        "voice_count": len(voices),
                        "piper_supports_speed": bool(
                            default_tts_engine == "piper"
                            and default_voice_value
                            and piper_voice_supports_length_scale(default_voice_value)
                        ),
                    }
                    chatterbox_mode_spec = param_specs.get("chatterbox_mode")
                    chatterbox_mode_dropdown = create_param_widget(
                        chatterbox_mode_spec,
                        coerce_param_value(
                            chatterbox_mode_spec,
                            engine_params.get("chatterbox_mode", chatterbox_mode_spec.default),
                        ),
                        default_tts_engine == "chatterbox",
                    )
                    speed_spec = param_specs.get("speed")
                    speed_value = None
                    speed_visible = False
                    if speed_spec:
                        speed_value = coerce_param_value(
                            speed_spec, engine_params.get("speed", speed_spec.default)
                        )
                        speed_visible = "speed" in engine_param_schema(default_tts_engine) and spec_visible(
                            speed_spec, default_context
                        )
                    language_dropdown = gr.Dropdown(
                        label="Langue",
                        choices=language_choices(default_supported_languages),
                        value=default_tts_language,
                        visible=show_lang_default,
                        interactive=show_lang_default,
                    )
                lang_locked_md = gr.Markdown(
                    default_lang_locked_text,
                    visible=not show_lang_default,
                )
                voice_label_md = gr.Markdown(
                    default_voice_label_text,
                    visible=default_voice_label_visible,
                )
                piper_voice_status_md = gr.Markdown(
                    piper_voice_status_text(voices) if default_tts_engine == "piper" else "",
                    visible=default_tts_engine == "piper",
                )
                with gr.Row():
                    piper_refresh_button = gr.Button(
                        "Refresh voix",
                        visible=default_tts_engine == "piper",
                    )
                    piper_install_voice_button = gr.Button(
                        "Installer une voix FR recommandée",
                        visible=default_tts_engine == "piper",
                    )
                piper_catalog_md = gr.Markdown(
                    "Catalogue voix Piper : https://huggingface.co/rhasspy/piper-voices/tree/main",
                    visible=default_tts_engine == "piper",
                )
                default_supports_speed = bool(
                    default_tts_engine == "piper"
                    and default_voice_value
                    and piper_voice_supports_length_scale(default_voice_value)
                )
                piper_speed_note_md = gr.Markdown(
                    "Vitesse non supportée par cette voix",
                    visible=default_tts_engine == "piper"
                    and bool(default_voice_value)
                    and not default_supports_speed,
                )
                if speed_spec:
                    speed_widget = create_param_widget(speed_spec, speed_value, speed_visible)
                    param_widgets["speed"] = speed_widget
                warnings_md = gr.Markdown("", visible=False)
                xtts_segmentation_md = gr.Markdown(
                    "Segmentation: XTTS native (recommandée)",
                    visible=default_tts_engine == "xtts",
                )
                gr.Markdown("### Paramètres moteur")
                param_keys = [key for key in all_param_keys() if key not in ("chatterbox_mode", "speed")]
                for key in param_keys:
                    spec = param_specs.get(key)
                    if spec is None:
                        continue
                    if key in engine_params:
                        value = coerce_param_value(spec, engine_params.get(key, spec.default))
                    else:
                        value = coerce_param_value(spec, spec.default)
                    visible = key in engine_param_schema(default_tts_engine) and spec_visible(
                        spec, default_context
                    )
                    param_widgets[key] = create_param_widget(spec, value, visible)
                param_widgets["chatterbox_mode"] = chatterbox_mode_dropdown
                inter_chunk_gap_slider = gr.Slider(
                    label="Blanc entre chunks (ms)",
                    minimum=0,
                    maximum=600,
                    step=10,
                    value=default_inter_chunk_gap_ms,
                    visible=default_tts_engine == "chatterbox",
                )
                inter_chunk_gap_help = gr.Markdown(
                    "Ajoute un silence au montage entre les chunks (respiration). "
                    "N’affecte pas le texte ni la génération.",
                    visible=default_tts_engine == "chatterbox",
                )
                engine_status_md = gr.Markdown(
                    engine_status_markdown(default_tts_engine),
                )
                install_backend_btn = gr.Button(
                    "Installer",
                    variant="primary",
                    visible=not backend_status(default_tts_engine).get("installed"),
                )
                uninstall_backend_btn = gr.Button(
                    "Désinstaller",
                    variant="secondary",
                    visible=backend_status(default_tts_engine).get("installed")
                    and default_tts_engine != "chatterbox",
                )
            with gr.Group(elem_id="v-gen-output"):
                gr.Markdown("## Sortie", elem_id="generation-output-title")
                with gr.Row():
                    output_dir_box = gr.Textbox(
                        label="Dossier de sortie",
                        value=default_out_dir_value,
                        lines=1,
                    )
                    choose_btn = gr.Button("Choisir…", size="sm")
                with gr.Row():
                    filename_box = gr.Textbox(
                        label="Nom de fichier (optionnel)",
                        value=default_user_filename,
                        placeholder="Ex: spot_festival_voix_finale",
                    )
                    timestamp_toggle = gr.Checkbox(
                        label="Ajouter timestamp",
                        value=default_add_timestamp,
                    )
                    include_model_toggle = gr.Checkbox(
                        label="Inclure nom du modèle",
                        value=default_include_model_name,
                    )
                with gr.Row():
                    generate_btn = gr.Button(
                        "Générer",
                        variant="primary",
                        interactive=backend_status(default_tts_engine).get("installed", True),
                    )
                    stop_btn = gr.Button("STOP", variant="secondary")
                output_path_box = gr.Textbox(label="Fichier généré", interactive=False)
                open_output_btn = gr.Button("Ouvrir le dossier output", size="sm")
                gr.Markdown("_Régénération par chunk arrive en RUN 9._")

            raw_audio = gr.Audio(
                label="RAW (référence)",
                type="filepath",
                autoplay=False,
            )
            gr.Markdown("RAW immuable (référence).")
            with gr.Row():
                export_raw_btn = gr.Button("Exporter RAW", size="sm")
                raw_export_path_box = gr.Textbox(label="RAW exporté", interactive=False)
            with gr.Group(elem_id="v-edit"):
                gr.Markdown("## Édition")
                edit_enabled_toggle = gr.Checkbox(
                    label="Activer l’édition (minimal)",
                    value=False,
                )
                edit_panel = gr.Column(visible=False)
                with edit_panel:
                    gr.Markdown("Édition minimale (trim + normalisation).")
                    trim_silence_toggle = gr.Checkbox(
                        label="Couper blancs début/fin",
                        value=True,
                    )
                    normalize_toggle = gr.Checkbox(
                        label="Normaliser",
                        value=True,
                    )
                    target_dbfs_slider = gr.Slider(
                        label="Niveau cible (dBFS)",
                        minimum=-12.0,
                        maximum=-0.1,
                        value=DEFAULT_EDIT_TARGET_DBFS,
                        step=0.1,
                        interactive=False,
                    )
                    generate_edit_btn = gr.Button("Générer audio édité", size="sm", interactive=False)
                    edited_audio = gr.Audio(
                        label="Édité",
                        type="filepath",
                        autoplay=False,
                    )
                    edited_path_box = gr.Textbox(label="Fichier édité", interactive=False)
            with gr.Group(elem_id="v-logs"):
                gr.Markdown("## Logs")
                with gr.Row():
                    verbose_logs_toggle = gr.Checkbox(
                        label="Logs détaillés",
                        value=default_verbose_logs,
                    )
                    copy_logs_btn = gr.Button("📋 Copier les logs", size="sm")
                logs_box = gr.Textbox(label="Logs", lines=4, max_lines=16, interactive=False)
                install_logs_box = gr.Textbox(
                    label="Logs installation",
                    lines=4,
                    max_lines=16,
                    interactive=False,
                )

        session_state = gr.State({"dir": None, "json": None})
        confirm_state = gr.State({"pending": None, "ts": 0.0})

        refresh_btn.click(
            refresh_dropdown,
            inputs=ref_dropdown,
            outputs=ref_dropdown,
            api_name=False,
        )
        upload.upload(
            fn=handle_upload,
            inputs=[upload, ref_dropdown, logs_box],
            outputs=[ref_dropdown, logs_box],
            api_name=False,
        )
        adjust_btn.click(
            fn=handle_adjust,
            inputs=[text_input, target_duration, logs_box],
            outputs=[adjusted_preview, adjust_info, logs_box],
            api_name=False,
        )
        apply_btn.click(
            fn=apply_adjusted,
            inputs=adjusted_preview,
            outputs=text_input,
            api_name=False,
        )
        text_input.change(
            fn=handle_text_adjustment,
            inputs=[
                text_input,
                auto_adjust_toggle,
                show_adjust_log_toggle,
            ],
            outputs=[
                adjusted_text_box,
                clean_text_box,
                adjust_info,
                adjust_log_box,
            ],
            api_name=False,
        )
        auto_adjust_toggle.change(
            fn=handle_text_adjustment,
            inputs=[
                text_input,
                auto_adjust_toggle,
                show_adjust_log_toggle,
            ],
            outputs=[
                adjusted_text_box,
                clean_text_box,
                adjust_info,
                adjust_log_box,
            ],
            api_name=False,
        )
        show_adjust_log_toggle.change(
            fn=handle_text_adjustment,
            inputs=[
                text_input,
                auto_adjust_toggle,
                show_adjust_log_toggle,
            ],
            outputs=[
                adjusted_text_box,
                clean_text_box,
                adjust_info,
                adjust_log_box,
            ],
            api_name=False,
        )
        direction_load_btn.click(
            fn=handle_direction_load_snapshot,
            inputs=[
                direction_source_dropdown,
                text_input,
                adjusted_text_box,
                clean_text_box,
                logs_box,
            ],
            outputs=[direction_snapshot_box, logs_box],
            api_name=False,
        )
        direction_snapshot_box.change(
            fn=update_direction_controls,
            inputs=[direction_enabled_toggle, direction_snapshot_box],
            outputs=[insert_chunk_btn, preview_chunks_btn],
            api_name=False,
        )
        direction_enabled_toggle.change(
            fn=update_direction_controls,
            inputs=[direction_enabled_toggle, direction_snapshot_box],
            outputs=[insert_chunk_btn, preview_chunks_btn],
            api_name=False,
        )
        preview_chunks_btn.click(
            fn=handle_direction_preview,
            inputs=[
                direction_enabled_toggle,
                direction_source_dropdown,
                direction_snapshot_box,
                clean_text_box,
                logs_box,
            ],
            outputs=[chunk_preview_box, logs_box],
            api_name=False,
        )
        insert_chunk_btn.click(
            fn=append_chunk_marker,
            inputs=[direction_snapshot_box],
            outputs=[direction_snapshot_box],
            js="""
            (text) => {
              const marker = "[[CHUNK]]";
              const el = document.querySelector("#direction_snapshot textarea");
              const current = text || "";
              if (!el) {
                return current + (current.endsWith("\\n") || !current ? "" : "\\n") + marker;
              }
              const start = el.selectionStart || 0;
              const end = el.selectionEnd || 0;
              const before = current.slice(0, start);
              const after = current.slice(end);
              const next = before + marker + after;
              const pos = start + marker.length;
              setTimeout(() => {
                el.selectionStart = pos;
                el.selectionEnd = pos;
              }, 0);
              return next;
            }
            """,
            api_name=False,
        )
        edit_enabled_toggle.change(
            fn=update_edit_panel_state,
            inputs=[edit_enabled_toggle, trim_silence_toggle, normalize_toggle],
            outputs=[edit_panel, generate_edit_btn, target_dbfs_slider],
            api_name=False,
        )
        trim_silence_toggle.change(
            fn=update_edit_panel_state,
            inputs=[edit_enabled_toggle, trim_silence_toggle, normalize_toggle],
            outputs=[edit_panel, generate_edit_btn, target_dbfs_slider],
            api_name=False,
        )
        normalize_toggle.change(
            fn=update_edit_panel_state,
            inputs=[edit_enabled_toggle, trim_silence_toggle, normalize_toggle],
            outputs=[edit_panel, generate_edit_btn, target_dbfs_slider],
            api_name=False,
        )
        generate_edit_btn.click(
            fn=handle_generate_edited_audio,
            inputs=[
                session_state,
                output_dir_box,
                filename_box,
                timestamp_toggle,
                include_model_toggle,
                trim_silence_toggle,
                normalize_toggle,
                target_dbfs_slider,
                logs_box,
            ],
            outputs=[edited_audio, edited_path_box, logs_box],
            api_name=False,
        )
        export_raw_btn.click(
            fn=handle_export_raw_to_output,
            inputs=[
                session_state,
                output_dir_box,
                filename_box,
                timestamp_toggle,
                include_model_toggle,
                logs_box,
            ],
            outputs=[raw_export_path_box, logs_box],
            api_name=False,
        )
        open_output_btn.click(
            fn=handle_open_output_dir,
            inputs=[output_dir_box, logs_box],
            outputs=[logs_box],
            api_name=False,
        )
        param_widget_list = [param_widgets[key] for key in all_param_keys()]
        engine_dropdown.change(
            fn=handle_engine_change,
            inputs=[
                engine_dropdown,
                language_dropdown,
                chatterbox_mode_dropdown,
            ],
            outputs=[
                language_dropdown,
                lang_locked_md,
                ref_dropdown,
                ref_support_note,
                engine_status_md,
                install_backend_btn,
                uninstall_backend_btn,
                generate_btn,
                voice_label_md,
                piper_voice_status_md,
                piper_refresh_button,
                piper_install_voice_button,
                piper_catalog_md,
                piper_speed_note_md,
                warnings_md,
                xtts_segmentation_md,
                inter_chunk_gap_slider,
                inter_chunk_gap_help,
                *param_widget_list,
            ],
            api_name=False,
        )
        language_dropdown.change(
            fn=handle_language_change,
            inputs=[
                engine_dropdown,
                language_dropdown,
                chatterbox_mode_dropdown,
            ],
            outputs=[
                language_dropdown,
                lang_locked_md,
                warnings_md,
                *param_widget_list,
            ],
            api_name=False,
        )
        chatterbox_mode_dropdown.change(
            fn=handle_chatterbox_mode_change,
            inputs=[
                chatterbox_mode_dropdown,
                language_dropdown,
            ],
            outputs=[
                language_dropdown,
                lang_locked_md,
                warnings_md,
                *param_widget_list,
            ],
            api_name=False,
        )
        install_backend_btn.click(
            fn=handle_install_backend,
            inputs=[engine_dropdown],
            outputs=[
                engine_status_md,
                install_backend_btn,
                uninstall_backend_btn,
                generate_btn,
                install_logs_box,
            ],
            api_name=False,
        )
        piper_refresh_button.click(
            fn=refresh_piper_voices,
            inputs=[],
            outputs=[
                *param_widget_list,
                voice_label_md,
                piper_voice_status_md,
                piper_speed_note_md,
                warnings_md,
                install_logs_box,
            ],
            api_name=False,
        )
        piper_install_voice_button.click(
            fn=install_default_piper_voice,
            inputs=[],
            outputs=[
                *param_widget_list,
                voice_label_md,
                piper_voice_status_md,
                piper_speed_note_md,
                warnings_md,
                install_logs_box,
            ],
            api_name=False,
        )
        if "voice_id" in param_widgets:
            param_widgets["voice_id"].change(
                fn=handle_voice_change,
                inputs=[
                    engine_dropdown,
                    param_widgets["voice_id"],
                    language_dropdown,
                    chatterbox_mode_dropdown,
                    *param_widget_list,
                ],
                outputs=[
                    language_dropdown,
                    lang_locked_md,
                    voice_label_md,
                    piper_speed_note_md,
                    warnings_md,
                    *param_widget_list,
                ],
                api_name=False,
            )
        uninstall_backend_btn.click(
            fn=handle_uninstall_backend,
            inputs=[engine_dropdown],
            outputs=[
                engine_status_md,
                install_backend_btn,
                uninstall_backend_btn,
                generate_btn,
                install_logs_box,
            ],
            api_name=False,
        )
        choose_btn.click(
            fn=handle_choose_output,
            inputs=[output_dir_box, logs_box],
            outputs=[output_dir_box, logs_box],
            api_name=False,
        )
        verbose_logs_toggle.change(
            fn=handle_toggle_verbose_logs,
            inputs=[verbose_logs_toggle, logs_box],
            outputs=[logs_box],
            api_name=False,
        )
        copy_logs_btn.click(
            fn=None,
            inputs=[logs_box],
            outputs=[],
            js="""
            (text) => {
              if (navigator && navigator.clipboard) {
                navigator.clipboard.writeText(text || "");
              }
              return [];
            }
            """,
            api_name=False,
        )
        load_preset_outputs = [
            ref_dropdown,
            ref_support_note,
            output_dir_box,
            filename_box,
            timestamp_toggle,
            include_model_toggle,
            engine_dropdown,
            language_dropdown,
            lang_locked_md,
            engine_status_md,
            install_backend_btn,
            uninstall_backend_btn,
            generate_btn,
            voice_label_md,
            piper_voice_status_md,
            piper_refresh_button,
            piper_install_voice_button,
            piper_catalog_md,
            piper_speed_note_md,
            warnings_md,
            xtts_segmentation_md,
            inter_chunk_gap_slider,
            inter_chunk_gap_help,
            *param_widget_list,
            verbose_logs_toggle,
            preset_dropdown,
            preset_name_box,
            logs_box,
        ]
        global _LOAD_PRESET_OUTPUT_COUNT
        _LOAD_PRESET_OUTPUT_COUNT = len(load_preset_outputs)
        load_preset_btn.click(
            fn=handle_load_preset,
            inputs=[preset_dropdown, logs_box],
            outputs=load_preset_outputs,
            api_name=False,
        )
        save_preset_btn.click(
            fn=handle_save_preset_confirm,
            inputs=[
                preset_name_box,
                ref_dropdown,
                output_dir_box,
                filename_box,
                timestamp_toggle,
                include_model_toggle,
                inter_chunk_gap_slider,
                language_dropdown,
                engine_dropdown,
                verbose_logs_toggle,
                confirm_state,
                logs_box,
                *param_widget_list,
            ],
            outputs=[
                preset_dropdown,
                preset_name_box,
                logs_box,
                confirm_state,
                save_preset_btn,
                delete_preset_btn,
                stop_btn,
            ],
            api_name=False,
        )
        delete_preset_btn.click(
            fn=handle_delete_preset_confirm,
            inputs=[preset_dropdown, confirm_state, logs_box],
            outputs=[
                preset_dropdown,
                preset_name_box,
                logs_box,
                confirm_state,
                save_preset_btn,
                delete_preset_btn,
                stop_btn,
            ],
            api_name=False,
        )
        cancel_confirm_btn.click(
            fn=handle_cancel_confirm,
            inputs=[logs_box],
            outputs=[
                logs_box,
                confirm_state,
                save_preset_btn,
                delete_preset_btn,
                stop_btn,
            ],
            api_name=False,
        )

        generate_btn.click(
            fn=handle_generate,
            inputs=[
                text_input,
                adjusted_text_box,
                auto_adjust_toggle,
                ref_dropdown,
                output_dir_box,
                filename_box,
                timestamp_toggle,
                include_model_toggle,
                engine_dropdown,
                language_dropdown,
                inter_chunk_gap_slider,
                verbose_logs_toggle,
                direction_enabled_toggle,
                direction_source_dropdown,
                direction_snapshot_box,
                logs_box,
                *param_widget_list,
            ],
            outputs=[
                adjusted_text_box,
                raw_audio,
                output_path_box,
                chunk_preview_box,
                logs_box,
                edited_audio,
                edited_path_box,
                raw_export_path_box,
                session_state,
            ],
            api_name=False,
        ).then(
            fn=handle_session_texts,
            inputs=[session_state],
            outputs=[
                session_editorial_box,
                session_tts_ready_box,
                session_prep_log_md,
                session_text_group,
            ],
            api_name=False,
        )
        stop_btn.click(
            fn=handle_stop_confirm,
            inputs=[confirm_state, logs_box],
            outputs=[
                raw_audio,
                output_path_box,
                logs_box,
                confirm_state,
                save_preset_btn,
                delete_preset_btn,
                stop_btn,
                edited_audio,
                edited_path_box,
                raw_export_path_box,
            ],
            api_name=False,
        )

    return demo


def main() -> None:
    demo = build_ui()
    port_env = os.environ.get("GRADIO_SERVER_PORT")
    debug_env = os.environ.get("GRADIO_DEBUG", "")
    launch_kwargs = {
        "allowed_paths": [str(BASE_DIR), str(Path.home())],
    }
    launch_kwargs.setdefault("server_name", "127.0.0.1")
    launch_kwargs.setdefault("share", False)
    launch_kwargs.setdefault("show_api", False)
    if debug_env.lower() in {"1", "true", "yes", "on"}:
        launch_kwargs.setdefault("debug", True)
    if port_env:
        try:
            launch_kwargs["server_port"] = int(port_env)
        except ValueError:
            LOGGER.warning("GRADIO_SERVER_PORT invalide (%s), fallback 7860.", port_env)
    demo.launch(**launch_kwargs)


if __name__ == "__main__":
    main()
