"""Gradio interface for the local Chatterbox French TTS tool."""

from __future__ import annotations

import datetime as dt
import dataclasses
import logging
import multiprocessing as mp
import os
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

from logging_utils import set_verbosity
from output_paths import make_output_filename, prepare_output_paths
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
    DEFAULT_COMMA_PAUSE_MS,
    DEFAULT_COLON_PAUSE_MS,
    DEFAULT_DASH_PAUSE_MS,
    DEFAULT_MAX_EST_SECONDS_PER_CHUNK,
    DEFAULT_MAX_WORDS_WITHOUT_TERMINATOR,
    DEFAULT_MIN_WORDS_PER_CHUNK,
    DEFAULT_NEWLINE_PAUSE_MS,
    DEFAULT_PERIOD_PAUSE_MS,
    DEFAULT_SEMICOLON_PAUSE_MS,
    ChunkInfo,
    SpeechSegment,
    adjust_text_to_duration,
    chunk_script,
    estimate_duration_with_pauses,
    normalize_text,
    prepare_adjusted_text,
    render_clean_text,
)
from backend_install.installer import run_install
from backend_install.status import backend_status
from tts_backends import get_backend, list_backends
from tts_backends.piper_assets import (
    ensure_default_voice_installed,
    list_piper_voices,
    piper_voice_supports_length_scale,
)
from tts_backends.base import BackendUnavailableError, coerce_language, pick_default_language
from tts_engine import (
    FADE_MS,
    SILENCE_MIN_MS,
    SILENCE_THRESHOLD,
    TTSEngine,
    ZERO_CROSS_RADIUS_MS,
)
from tts_pipeline import build_pause_plan, run_tts_pipeline


logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("chatterbox_app")

BASE_DIR = Path(__file__).resolve().parent
LEXIQUE_PATH = BASE_DIR / "lexique_tts_fr.json"

SAFE_PREVIEW_DIR = BASE_DIR / "output"
SAFE_PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
TMP_DIR = SAFE_PREVIEW_DIR / ".tmp"
TMP_DIR.mkdir(parents=True, exist_ok=True)

output_env = os.environ.get("CHATTERBOX_OUT_DIR")
if output_env:
    DEFAULT_OUTPUT_DIR = Path(output_env).expanduser()
else:
    DEFAULT_OUTPUT_DIR = SAFE_PREVIEW_DIR
DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ENGINE: Optional[TTSEngine] = None

_JOB_LOCK = Lock()
_JOB_STATE = {
    "current_proc": None,
    "current_tmp_path": None,
    "current_final_path": None,
    "job_running": False,
}
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


def get_engine() -> TTSEngine:
    global ENGINE
    if ENGINE is None:
        ENGINE = TTSEngine()
    return ENGINE


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
        meta = run_tts_pipeline(request).meta
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


def update_estimated_duration(
    text: str,
    comma_pause_ms: int,
    period_pause_ms: int,
    semicolon_pause_ms: int,
    colon_pause_ms: int,
    dash_pause_ms: int,
    newline_pause_ms: int,
) -> str:
    est = estimate_duration_with_pauses(
        text,
        comma_pause_ms=int(comma_pause_ms),
        period_pause_ms=int(period_pause_ms),
        semicolon_pause_ms=int(semicolon_pause_ms),
        colon_pause_ms=int(colon_pause_ms),
        dash_pause_ms=int(dash_pause_ms),
        newline_pause_ms=int(newline_pause_ms),
    )
    return f"Durée estimée (avec pauses): {est:.1f}s"


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
    comma_pause_ms: int,
    period_pause_ms: int,
    semicolon_pause_ms: int,
    colon_pause_ms: int,
    dash_pause_ms: int,
    newline_pause_ms: int,
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
    duration = update_estimated_duration(
        adjusted_text,
        comma_pause_ms=int(comma_pause_ms),
        period_pause_ms=int(period_pause_ms),
        semicolon_pause_ms=int(semicolon_pause_ms),
        colon_pause_ms=int(colon_pause_ms),
        dash_pause_ms=int(dash_pause_ms),
        newline_pause_ms=int(newline_pause_ms),
    )
    log_md = format_adjustment_log(changes, show_adjust_log and auto_adjust)
    return adjusted_text, clean_preview, duration, log_md


def handle_load_preset(
    preset_name: str | None,
    log_text: str | None,
):
    outputs = [gr.update() for _ in range(39 + len(all_param_keys()))]
    name_update = gr.update()
    chunk_status = "Etat: non appliqué"
    chunk_state = {"applied": False, "chunks": [], "signature": None}
    if not preset_name:
        log_text = append_log("Sélectionnez un preset à charger.", log_text)
        return (*outputs, name_update, chunk_status, chunk_state, log_text)

    data = load_preset(preset_name)
    if not data:
        log_text = append_log(f"Preset introuvable: {preset_name}", log_text)
        return (*outputs, name_update, chunk_status, chunk_state, log_text)

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
    updates = [
        gr.update(value=ref_value, visible=bool(backend and backend.supports_ref_audio)),
        ref_note_update,
        gr.update(value=data.get("out_dir") or str(DEFAULT_OUTPUT_DIR)),
        gr.update(value=data.get("user_filename", "")),
        gr.update(value=bool(data.get("add_timestamp", True))),
        gr.update(value=engine_id),
        lang_update,
        lang_locked_update,
        gr.update(value=engine_status_markdown(engine_id)),
        gr.update(visible=not backend_status(engine_id).get("installed")),
        gr.update(visible=backend_status(engine_id).get("installed") and engine_id != "chatterbox"),
        gr.update(interactive=backend_status(engine_id).get("installed")),
        voice_label_update,
        piper_status_update,
        piper_refresh_update,
        piper_install_update,
        piper_catalog_update,
        piper_speed_note,
        warning_update,
        *param_updates,
        gr.update(value=int(data.get("comma_pause_ms", DEFAULT_COMMA_PAUSE_MS))),
        gr.update(value=int(data.get("period_pause_ms", DEFAULT_PERIOD_PAUSE_MS))),
        gr.update(value=int(data.get("semicolon_pause_ms", DEFAULT_SEMICOLON_PAUSE_MS))),
        gr.update(value=int(data.get("colon_pause_ms", DEFAULT_COLON_PAUSE_MS))),
        gr.update(value=int(data.get("dash_pause_ms", DEFAULT_DASH_PAUSE_MS))),
        gr.update(value=int(data.get("newline_pause_ms", DEFAULT_NEWLINE_PAUSE_MS))),
        gr.update(value=int(data.get("min_words_per_chunk", DEFAULT_MIN_WORDS_PER_CHUNK))),
        gr.update(
            value=int(data.get("max_words_without_terminator", DEFAULT_MAX_WORDS_WITHOUT_TERMINATOR))
        ),
        gr.update(value=float(data.get("max_est_seconds_per_chunk", DEFAULT_MAX_EST_SECONDS_PER_CHUNK))),
        gr.update(value=bool(data.get("disable_newline_chunking", False))),
        gr.update(value=bool(data.get("verbose_logs", False))),
        gr.update(value=int(data.get("fade_ms", FADE_MS))),
        gr.update(value=int(data.get("zero_cross_radius_ms", ZERO_CROSS_RADIUS_MS))),
        gr.update(value=_coerce_float(data.get("silence_threshold"), SILENCE_THRESHOLD)),
        gr.update(value=int(data.get("silence_min_ms", SILENCE_MIN_MS))),
        gr.update(value=preset_name),
    ]
    name_update = gr.update(value=preset_name)
    persist_state(
        {
            "last_preset": preset_name,
            "last_tts_engine": engine_id,
        }
    )
    persist_engine_state(engine_id, language=final_lang, voice_id=final_voice, params=engine_params)
    log_text = append_log(f"Preset chargé: {preset_name}", log_text)
    return (*updates, name_update, chunk_status, chunk_state, log_text)


def handle_save_preset(
    preset_name: str | None,
    ref_name: str | None,
    out_dir: str | None,
    user_filename: str | None,
    add_timestamp: bool,
    tts_language: str | None,
    tts_engine: str,
    comma_pause_ms: int,
    period_pause_ms: int,
    semicolon_pause_ms: int,
    colon_pause_ms: int,
    dash_pause_ms: int,
    newline_pause_ms: int,
    min_words_per_chunk: int,
    max_words_without_terminator: int,
    max_est_seconds_per_chunk: float,
    disable_newline_chunking: bool,
    verbose_logs: bool,
    fade_ms: int,
    zero_cross_radius_ms: int,
    silence_threshold: float,
    silence_min_ms: int,
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
        "engines": {
            str(tts_engine): {
                "language": str(tts_language or "fr-FR"),
                "voice_id": voice_id,
                "params": engine_params,
            }
        },
        "comma_pause_ms": int(comma_pause_ms),
        "period_pause_ms": int(period_pause_ms),
        "semicolon_pause_ms": int(semicolon_pause_ms),
        "colon_pause_ms": int(colon_pause_ms),
        "dash_pause_ms": int(dash_pause_ms),
        "newline_pause_ms": int(newline_pause_ms),
        "min_words_per_chunk": int(min_words_per_chunk),
        "max_words_without_terminator": int(max_words_without_terminator),
        "max_est_seconds_per_chunk": float(max_est_seconds_per_chunk),
        "disable_newline_chunking": bool(disable_newline_chunking),
        "verbose_logs": bool(verbose_logs),
        "fade_ms": int(fade_ms),
        "zero_cross_radius_ms": int(zero_cross_radius_ms),
        "silence_threshold": float(silence_threshold),
        "silence_min_ms": int(silence_min_ms),
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


def handle_reset_chunk_defaults(log_text: str | None):
    log_text = append_ui_log("Reset chunking defaults.", log_text)
    return (
        gr.update(value=DEFAULT_MIN_WORDS_PER_CHUNK),
        gr.update(value=DEFAULT_MAX_WORDS_WITHOUT_TERMINATOR),
        gr.update(value=DEFAULT_MAX_EST_SECONDS_PER_CHUNK),
        gr.update(value=False),
        log_text,
    )


def handle_toggle_verbose_logs(verbose: bool, log_text: str | None):
    set_verbosity(bool(verbose))
    persist_state({"last_verbose_logs": bool(verbose)})
    log_text = append_ui_log("Verbosity terminal mise à jour.", log_text)
    return log_text


def handle_engine_change(
    engine_id: str,
    language: str | None,
    chatterbox_mode: str,
    chunk_state: dict | None,
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
    warning_update = gr.update(value=warning_md, visible=bool(warning_md))
    piper_status_update = gr.update(
        value=piper_voice_status_text(voices),
        visible=engine_id == "piper",
    )
    piper_refresh_update = gr.update(visible=engine_id == "piper")
    piper_install_update = gr.update(visible=engine_id == "piper")
    piper_catalog_update = gr.update(visible=engine_id == "piper")
    piper_speed_note = piper_speed_note_update(engine_id, final_voice, supports_speed)
    persist_state({"last_tts_engine": engine_id})
    persist_engine_state(engine_id, language=final_lang, voice_id=final_voice, params=engine_params)
    state, status = mark_chunk_dirty(chunk_state)
    return (
        lang_update,
        lang_locked_update,
        state,
        status,
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
        *param_updates,
    )


def handle_voice_change(
    engine_id: str,
    voice_id: str | None,
    language: str | None,
    chatterbox_mode: str,
    chunk_state: dict | None,
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
    chunk_state: dict | None,
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
    state, status = mark_chunk_dirty(chunk_state)
    return lang_update, lang_locked_update, warning_update, state, status, *param_updates


def handle_chatterbox_mode_change(
    chatterbox_mode: str,
    language: str | None,
    chunk_state: dict | None,
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
    state, status = mark_chunk_dirty(chunk_state)
    return lang_update, lang_locked_update, warning_update, state, status, *param_updates


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


def _chunk_signature(
    text: str,
    min_words_per_chunk: int,
    max_words_without_terminator: int,
    max_est_seconds_per_chunk: float,
    disable_newline_chunking: bool,
) -> tuple:
    return (
        text.strip(),
        int(min_words_per_chunk),
        int(max_words_without_terminator),
        float(max_est_seconds_per_chunk),
        bool(disable_newline_chunking),
    )


def _build_chunk_preview(chunks) -> str:
    lines = []
    for idx, chunk_info in enumerate(chunks, start=1):
        warn = f" warnings={','.join(chunk_info.warnings)}" if chunk_info.warnings else ""
        lines.append(
            f"[{idx}] words={chunk_info.word_count} est={chunk_info.estimated_duration:.1f}s "
            f"reason={chunk_info.reason}{warn}"
        )
    return "\n".join(lines)


def _append_chunk_warning_logs(chunks, log_text: str | None, verbose_logs: bool) -> str | None:
    for chunk_info in chunks:
        for warning in chunk_info.warnings:
            if warning == "newline_boundary_skipped_min_words":
                log_text = append_ui_log(
                    "newline_boundary_skipped_min_words",
                    log_text,
                    verbose=True,
                    enabled=verbose_logs,
                )
            elif warning.startswith("fallback_split_used:"):
                log_text = append_ui_log(
                    warning,
                    log_text,
                    verbose=True,
                    enabled=verbose_logs,
                )
            elif warning == "hard_split_no_punct":
                log_text = append_ui_log(
                    "hard_split_no_punct",
                    log_text,
                    verbose=True,
                    enabled=verbose_logs,
                )
    return log_text


def handle_apply_prechunk(
    text: str,
    adjusted_text: str,
    auto_adjust: bool,
    min_words_per_chunk: int,
    max_words_without_terminator: int,
    max_est_seconds_per_chunk: float,
    disable_newline_chunking: bool,
    chunk_state: dict | None,
    log_text: str | None,
    verbose_logs: bool,
):
    if not text or not text.strip():
        return "", "Etat: non appliqué", {"applied": False, "chunks": [], "signature": None}, log_text
    if auto_adjust:
        adjusted_text, changes = prepare_adjusted_text(text or "", LEXIQUE_PATH)
        log_text = summarize_adjustment_changes(changes, log_text, verbose_logs)
    else:
        adjusted_text = adjusted_text or text or ""
    normalized_text = normalize_text(adjusted_text or "")
    chunks = chunk_script(
        normalized_text,
        min_words_per_chunk=int(min_words_per_chunk),
        max_words_without_terminator=int(max_words_without_terminator),
        max_est_seconds_per_chunk=float(max_est_seconds_per_chunk),
        split_on_newline=not disable_newline_chunking,
    )
    if not chunks:
        log_text = append_ui_log("Aucun chunk généré.", log_text)
        return "", "Etat: non appliqué", {"applied": False, "chunks": [], "signature": None}, log_text
    preview = _build_chunk_preview(chunks)
    log_text = _append_chunk_warning_logs(chunks, log_text, verbose_logs)
    signature = _chunk_signature(
        normalized_text,
        min_words_per_chunk,
        max_words_without_terminator,
        max_est_seconds_per_chunk,
        disable_newline_chunking,
    )
    state = {"applied": True, "chunks": chunks, "signature": signature}
    log_text = append_ui_log("Pré-chunking appliqué.", log_text)
    return preview, "Etat: appliqué", state, log_text


def mark_chunk_dirty(chunk_state: dict | None):
    state = dict(chunk_state or {"chunks": [], "signature": None})
    state["applied"] = False
    state["signature"] = None
    state["chunks"] = []
    return state, "Etat: non appliqué"


def handle_generate(
    text: str,
    adjusted_text: str,
    auto_adjust: bool,
    ref_name: str | None,
    out_dir: str | None,
    user_filename: str | None,
    add_timestamp: bool,
    tts_engine: str,
    tts_language: str | None,
    comma_pause_ms: int,
    period_pause_ms: int,
    semicolon_pause_ms: int,
    colon_pause_ms: int,
    dash_pause_ms: int,
    newline_pause_ms: int,
    min_words_per_chunk: int,
    max_words_without_terminator: int,
    max_est_seconds_per_chunk: float,
    disable_newline_chunking: bool,
    verbose_logs: bool,
    fade_ms: int,
    zero_cross_radius_ms: int,
    silence_threshold: float,
    silence_min_ms: int,
    chunk_state: dict | None,
    log_text: str | None,
    *param_values,
):
    if not text or not text.strip():
        state = chunk_state or {"applied": False, "chunks": [], "signature": None}
        return "", None, "", "", "Etat: non appliqué", state, append_log("Erreur: texte vide.", log_text)

    log_text = append_ui_log("Initialisation de la génération...", log_text)

    if auto_adjust:
        adjusted_text, changes = prepare_adjusted_text(text or "", LEXIQUE_PATH)
        log_text = summarize_adjustment_changes(changes, log_text, verbose_logs)
    else:
        adjusted_text = adjusted_text or text or ""

    text_used = adjusted_text
    if auto_adjust:
        assert text_used == adjusted_text

    param_values_map = dict(zip(all_param_keys(), param_values))
    engine_params = collect_engine_params(tts_engine, param_values_map)
    voice_id = engine_params.get("voice_id")
    tts_language = tts_language or "fr-FR"
    requested_language = tts_language
    chatterbox_mode = engine_params.get("chatterbox_mode", "fr_finetune")
    normalized_text = normalize_text(text_used)
    if normalized_text != text_used and verbose_logs:
        before = len(re.findall(r"\bII\b", text_used))
        after = len(re.findall(r"\bII\b", normalized_text))
        ii_fix = max(before - after, 0)
        detail = f"II->Il x{ii_fix}" if ii_fix else "whitespace/ponctuation"
        log_text = append_ui_log(f"Normalisation: {detail}", log_text, verbose=True, enabled=True)

    output_dir = ensure_output_dir(out_dir)
    timestamp = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = make_output_filename(
        text=normalized_text,
        ref_name=ref_name,
        user_filename=user_filename,
        add_timestamp=bool(add_timestamp),
        timestamp=timestamp,
    )
    preview_path, user_path = prepare_output_paths(SAFE_PREVIEW_DIR, output_dir, filename)
    clean_preview = render_clean_text(normalized_text)
    if clean_preview:
        log_text = append_ui_log("Texte interprété prêt.", log_text)

    estimate = estimate_duration_with_pauses(
        normalized_text,
        comma_pause_ms=int(comma_pause_ms),
        period_pause_ms=int(period_pause_ms),
        semicolon_pause_ms=int(semicolon_pause_ms),
        colon_pause_ms=int(colon_pause_ms),
        dash_pause_ms=int(dash_pause_ms),
        newline_pause_ms=int(newline_pause_ms),
    )
    if estimate >= 35:
        log_text = append_ui_log("⚠️ Texte long détecté.", log_text)

    backend = get_backend(tts_engine)
    if backend is None:
        log_text = append_ui_log(f"Backend introuvable: {tts_engine}", log_text)
        state = chunk_state or {"applied": False, "chunks": [], "signature": None}
        return adjusted_text, None, "", "", "Etat: non appliqué", state, log_text
    if not backend.is_available():
        reason = backend.unavailable_reason() or "Dépendances manquantes."
        log_text = append_ui_log(f"Backend indisponible: {tts_engine}. {reason}", log_text)
        state = chunk_state or {"applied": False, "chunks": [], "signature": None}
        return adjusted_text, None, "", "", "Etat: non appliqué", state, log_text

    supported = supported_languages_for(tts_engine, backend, chatterbox_mode)
    tts_language, did_fallback = coerce_language(
        tts_language,
        supported,
        backend.default_language if backend else None,
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
            return adjusted_text, None, "", "", "Etat: non appliqué", chunk_state, log_text
    elif ref_name and not backend.supports_ref_audio:
        log_text = append_ui_log("Référence ignorée (backend sans voice ref).", log_text)

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
            state = chunk_state or {"applied": False, "chunks": [], "signature": None}
            return adjusted_text, None, "", "", "Etat: non appliqué", state, log_text
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

    chunk_preview_text = ""
    signature = _chunk_signature(
        normalized_text,
        min_words_per_chunk,
        max_words_without_terminator,
        max_est_seconds_per_chunk,
        disable_newline_chunking,
    )
    applied = bool(chunk_state and chunk_state.get("applied"))
    same_signature = bool(chunk_state and chunk_state.get("signature") == signature)
    if not applied or not same_signature:
        log_text = append_ui_log("auto_apply_before_generate", log_text)
        chunks = chunk_script(
            normalized_text,
            min_words_per_chunk=int(min_words_per_chunk),
            max_words_without_terminator=int(max_words_without_terminator),
            max_est_seconds_per_chunk=float(max_est_seconds_per_chunk),
            split_on_newline=not disable_newline_chunking,
        )
        chunk_preview_text = _build_chunk_preview(chunks)
        log_text = _append_chunk_warning_logs(chunks, log_text, verbose_logs)
        chunk_state = {"applied": True, "chunks": chunks, "signature": signature}
        chunk_status = "Etat: appliqué"
    else:
        chunks = chunk_state.get("chunks", [])
        chunk_preview_text = _build_chunk_preview(chunks)
        chunk_status = "Etat: appliqué"

    if not chunks:
        state = {"applied": False, "chunks": [], "signature": None}
        return (
            adjusted_text,
            None,
            "",
            chunk_preview_text,
            "Etat: non appliqué",
            state,
            append_ui_log("Aucun chunk généré.", log_text),
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
        "pause_plan": build_pause_plan(
            chunks,
            {
                "comma_pause_ms": int(comma_pause_ms),
                "period_pause_ms": int(period_pause_ms),
                "semicolon_pause_ms": int(semicolon_pause_ms),
                "colon_pause_ms": int(colon_pause_ms),
                "dash_pause_ms": int(dash_pause_ms),
                "newline_pause_ms": int(newline_pause_ms),
            },
        ),
        "voice_ref_path": audio_prompt,
        "out_path": str(tmp_path),
        "lang": str(backend_language or tts_language or "fr-FR"),
        "engine_params": {},
        "chunk_settings": {
            "min_words_per_chunk": int(min_words_per_chunk),
            "max_words_without_terminator": int(max_words_without_terminator),
            "max_est_seconds_per_chunk": float(max_est_seconds_per_chunk),
        },
        "pause_settings": {
            "comma_pause_ms": int(comma_pause_ms),
            "period_pause_ms": int(period_pause_ms),
            "semicolon_pause_ms": int(semicolon_pause_ms),
            "colon_pause_ms": int(colon_pause_ms),
            "dash_pause_ms": int(dash_pause_ms),
            "newline_pause_ms": int(newline_pause_ms),
        },
        "post_settings": {
            "zero_cross_radius_ms": int(zero_cross_radius_ms),
            "fade_ms": int(fade_ms),
            "silence_threshold": float(silence_threshold),
            "silence_min_ms": int(silence_min_ms),
        },
        "target_sr": 24000,
    }
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
        current_final_path=str(preview_path),
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
        return adjusted_text, None, "", chunk_preview_text, chunk_status, chunk_state, log_text

    meta = result.get("meta", {})
    if meta.get("piper_voice"):
        log_text = append_ui_log(f"piper_voice={meta.get('piper_voice')}", log_text)
    if meta.get("piper_model_path"):
        log_text = append_ui_log(f"piper_model_path={meta.get('piper_model_path')}", log_text)
    if verbose_logs and meta.get("backend_cmd"):
        log_text = append_ui_log(f"backend_cmd={meta.get('backend_cmd')}", log_text, verbose=True, enabled=True)
    preview_path_obj = Path(preview_path)
    try:
        preview_path_obj.parent.mkdir(parents=True, exist_ok=True)
        if tmp_path.exists():
            os.replace(tmp_path, preview_path_obj)
        else:
            _reset_job_state()
            log_text = append_ui_log("Annulé.", log_text)
            return adjusted_text, None, "", chunk_preview_text, chunk_status, chunk_state, log_text
    except Exception as exc:
        _cleanup_tmp(str(tmp_path))
        _reset_job_state()
        log_text = append_ui_log(f"Erreur TTS: {exc}", log_text)
        return adjusted_text, None, "", chunk_preview_text, chunk_status, chunk_state, log_text

    for idx, duration in enumerate(meta.get("durations", []), start=1):
        retry_flag = meta.get("retries", [])[idx - 1] if meta.get("retries") else False
        chunk_info = chunks[idx - 1] if idx - 1 < len(chunks) else None
        reason = chunk_info.reason if chunk_info else "n/a"
        est = chunk_info.estimated_duration if chunk_info else 0.0
        boundary_pauses = meta.get("boundary_pauses", [])
        boundary_kinds = meta.get("boundary_kinds", [])
        punct_fixes = meta.get("punct_fixes", [])
        pause_note = ""
        if idx - 1 < len(boundary_pauses):
            boundary_pause = boundary_pauses[idx - 1] if idx - 1 < len(boundary_pauses) else 0
            boundary_kind = boundary_kinds[idx - 1] if idx - 1 < len(boundary_kinds) else "none"
            pause_note = f" boundary={boundary_kind} boundary_pause={boundary_pause}ms"
        fix_note = ""
        if idx - 1 < len(punct_fixes) and punct_fixes[idx - 1]:
            fix_note = f" punct_fix={punct_fixes[idx - 1]}"
        retry_note = " retry" if retry_flag else ""
        log_text = append_ui_log(
            f"Chunk {idx}/{meta.get('chunks', len(meta.get('durations', [])))} "
            f"reason={reason} est={est:.1f}s measured={duration:.2f}s{retry_note}{pause_note}{fix_note}",
            log_text,
            verbose=True,
            enabled=verbose_logs,
        )
    pause_events = meta.get("pause_events", [])
    if verbose_logs and pause_events:
        for chunk_idx, events in enumerate(pause_events, start=1):
            for event in events:
                symbol = event.get("symbol", "?")
                duration_ms = event.get("duration_ms", 0)
                log_text = append_ui_log(
                    f"punct_pause_applied chunk={chunk_idx} symbol={symbol} duration_ms={duration_ms}",
                    log_text,
                    verbose=True,
                    enabled=True,
                )
    total_duration = meta.get("total_duration")
    if total_duration is not None:
        log_text = append_ui_log(f"Durée finale: {total_duration:.2f}s", log_text)
    if meta.get("total_pause_ms") is not None:
        log_text = append_ui_log(
            f"pause_total_ms={meta.get('total_pause_ms')}",
            log_text,
            verbose=True,
            enabled=verbose_logs,
        )
    if meta.get("segments_count_total") is not None:
        log_text = append_ui_log(
            f"segments_count_total={meta.get('segments_count_total')}",
            log_text,
            verbose=True,
            enabled=verbose_logs,
        )
    if meta.get("join_count") is not None:
        log_text = append_ui_log(
            f"join_count={meta.get('join_count')}",
            log_text,
            verbose=True,
            enabled=verbose_logs,
        )
    if verbose_logs and meta.get("punct_tokens"):
        token_items = " ".join(f"{tok}:{ms}" for tok, ms in meta.get("punct_tokens", []))
        log_text = append_ui_log(
            f"punct_tokens={token_items}",
            log_text,
            verbose=True,
            enabled=True,
        )
    if meta.get("sr"):
        log_text = append_ui_log(
            f"sr_final={meta.get('sr')}",
            log_text,
            verbose=True,
            enabled=verbose_logs,
        )

    user_path_obj = Path(user_path)
    preview_path_obj = Path(preview_path)
    if user_path_obj.resolve() != preview_path_obj.resolve():
        try:
            user_path_obj.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(preview_path_obj, user_path_obj)
            log_text = append_ui_log(f"Copie dossier utilisateur: {user_path_obj}", log_text)
        except Exception as exc:
            log_text = append_ui_log(f"Copie échouée: {exc}", log_text)
    _reset_job_state()
    persist_state(
        {
            "last_ref": ref_name,
            "last_out_dir": output_dir,
            "last_user_filename": user_filename or "",
            "last_add_timestamp": bool(add_timestamp),
            "last_comma_pause_ms": int(comma_pause_ms),
            "last_period_pause_ms": int(period_pause_ms),
            "last_semicolon_pause_ms": int(semicolon_pause_ms),
            "last_colon_pause_ms": int(colon_pause_ms),
            "last_dash_pause_ms": int(dash_pause_ms),
            "last_newline_pause_ms": int(newline_pause_ms),
            "last_min_words_per_chunk": int(min_words_per_chunk),
            "last_max_words_without_terminator": int(max_words_without_terminator),
            "last_max_est_seconds_per_chunk": float(max_est_seconds_per_chunk),
            "last_disable_newline_chunking": bool(disable_newline_chunking),
            "last_verbose_logs": bool(verbose_logs),
            "last_tts_engine": str(tts_engine),
        }
    )
    persist_engine_state(tts_engine, language=tts_language, voice_id=voice_id, params=engine_params)
    log_text = append_ui_log(f"Fichier pré-écoute: {preview_path_obj}", log_text)
    return (
        adjusted_text,
        str(preview_path_obj),
        str(user_path_obj),
        chunk_preview_text,
        chunk_status,
        chunk_state,
        log_text,
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
    tts_language: str | None,
    tts_engine: str,
    comma_pause_ms: int,
    period_pause_ms: int,
    semicolon_pause_ms: int,
    colon_pause_ms: int,
    dash_pause_ms: int,
    newline_pause_ms: int,
    min_words_per_chunk: int,
    max_words_without_terminator: int,
    max_est_seconds_per_chunk: float,
    disable_newline_chunking: bool,
    verbose_logs: bool,
    fade_ms: int,
    zero_cross_radius_ms: int,
    silence_threshold: float,
    silence_min_ms: int,
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
        tts_language,
        tts_engine,
        comma_pause_ms,
        period_pause_ms,
        semicolon_pause_ms,
        colon_pause_ms,
        dash_pause_ms,
        newline_pause_ms,
        min_words_per_chunk,
        max_words_without_terminator,
        max_est_seconds_per_chunk,
        disable_newline_chunking,
        verbose_logs,
        fade_ms,
        zero_cross_radius_ms,
        silence_threshold,
        silence_min_ms,
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

    default_out_dir_value = state_data.get("last_out_dir") or str(DEFAULT_OUTPUT_DIR)
    default_user_filename = state_data.get("last_user_filename", "")
    default_add_timestamp = _coerce_bool(state_data.get("last_add_timestamp"), True)
    default_comma_pause = int(_state_or_preset("comma_pause_ms", DEFAULT_COMMA_PAUSE_MS))
    default_period_pause = int(_state_or_preset("period_pause_ms", DEFAULT_PERIOD_PAUSE_MS))
    default_semicolon_pause = int(_state_or_preset("semicolon_pause_ms", DEFAULT_SEMICOLON_PAUSE_MS))
    default_colon_pause = int(_state_or_preset("colon_pause_ms", DEFAULT_COLON_PAUSE_MS))
    default_dash_pause = int(_state_or_preset("dash_pause_ms", DEFAULT_DASH_PAUSE_MS))
    default_newline_pause = int(_state_or_preset("newline_pause_ms", DEFAULT_NEWLINE_PAUSE_MS))
    default_min_words_per_chunk = int(
        _state_or_preset("min_words_per_chunk", DEFAULT_MIN_WORDS_PER_CHUNK)
    )
    default_max_words_without_terminator = int(
        _state_or_preset("max_words_without_terminator", DEFAULT_MAX_WORDS_WITHOUT_TERMINATOR)
    )
    default_max_est_seconds = float(
        _state_or_preset("max_est_seconds_per_chunk", DEFAULT_MAX_EST_SECONDS_PER_CHUNK)
    )
    default_auto_adjust = _coerce_bool(state_data.get("last_auto_adjust"), True)
    default_show_adjust_log = _coerce_bool(state_data.get("last_show_adjust_log"), False)
    default_disable_newline_chunking = _coerce_bool(
        state_data.get("last_disable_newline_chunking")
        if "last_disable_newline_chunking" in state_data
        else base_preset.get("disable_newline_chunking"),
        False,
    )
    default_verbose_logs = _coerce_bool(state_data.get("last_verbose_logs"), False)
    default_tts_engine = state_data.get("last_tts_engine") or _state_or_preset(
        "tts_engine", "chatterbox"
    )
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
    default_fade_ms = int(_state_or_preset("fade_ms", FADE_MS))
    default_zero_cross_radius_ms = int(
        _state_or_preset("zero_cross_radius_ms", ZERO_CROSS_RADIUS_MS)
    )
    default_silence_threshold = _coerce_float(
        _state_or_preset("silence_threshold", SILENCE_THRESHOLD),
        SILENCE_THRESHOLD,
    )
    default_silence_min_ms = int(_state_or_preset("silence_min_ms", SILENCE_MIN_MS))
    preset_choices = list_presets()
    default_preset = state_data.get("last_preset")
    if default_preset not in preset_choices:
        default_preset = "default"

    with gr.Blocks(
        title="Chatterbox TTS FR",
        css=(
                ".gradio-container { font-family: -apple-system, \"SF Pro Text\", \"SF Pro Display\", "
                "\"Helvetica Neue\", Helvetica, Arial, sans-serif; }"

                "/* Section headers (elem_classes=[\"section-title\"]) */"
                ".section-title {"
                "  background: #3f4046;"
                "  padding: 0.4rem 0.85rem;"
                "  border-radius: 8px 8px 0 0;"
                "  margin: 0.35rem 0 0.35rem 0;"   
                "}"
                ".section-title .prose { padding: 0 !important; margin: 0 !important; }"
                ".section-title p { margin: 0 !important; }"
                ".section-title h1, .section-title h2, .section-title h3 {"
                "  margin: 0 !important;"
                "  line-height: 1.05;"
                "  font-weight: 650;"
                "}"
                ".section-title h2 { font-size: 1.15rem; }"
                ".section-title h3 { font-size: 1.05rem; }"

                "/* Subsection titles and small info lines */"
                ".subhead {"
                "  margin: 0.65rem 0 0.45rem 0;"
                "  padding: 0.25rem 0.6rem;"
                "  border-radius: 6px;"
                "  background: rgba(255,255,255,0.03);"
                "}"
                ".subhead .prose, .subhead .gr-markdown { margin: 0 !important; }"
                ".subhead h3 { margin: 0 !important; line-height: 1.2; }"

                ".inline-info {"
                "  margin: 0.35rem 0 0.25rem 0;"
                "  padding: 0.2rem 0.6rem;"
                "  border-radius: 6px;"
                "}"
                ".inline-info .prose, .inline-info .gr-markdown { margin: 0 !important; }"
                ".inline-info p { margin: 0 !important; }"

                "/* Reduce extra vertical padding inside groups */"
                ".gradio-container .gr-group { padding-top: 0.25rem !important; padding-bottom: 0.25rem !important; }"
                ".gradio-container .gr-group > .wrap { padding-top: 0.25rem !important; padding-bottom: 0.25rem !important; }"
            ),
        ) as demo:
        set_verbosity(default_verbose_logs)
        gr.Markdown("""# 🎙️ Chatterbox TTS FR\nInterface locale pour générer des voix off expressives en français.""")

        with gr.Group():
            gr.Markdown("## Presets", elem_classes=["section-title"])
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
                load_preset_btn = gr.Button("Charger", elem_classes=["accent-btn"])
                save_preset_btn = gr.Button("Sauver", variant="secondary", elem_classes=["accent-btn"])
                delete_preset_btn = gr.Button("Supprimer", variant="secondary", elem_classes=["accent-btn"])
                cancel_confirm_btn = gr.Button("Annuler", size="sm")

        with gr.Group():
            gr.Markdown("## Références vocales", elem_classes=["section-title"])
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
                elem_classes=["inline-info"],
            )
            with gr.Accordion("Importer des fichiers audio", open=False):
                upload = gr.Files(
                    label="Drag & drop",
                    file_types=list(ALLOWED_EXTENSIONS),
                    file_count="multiple",
                )

        with gr.Group():
            gr.Markdown("## Texte", elem_classes=["section-title"])
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
            adjusted_text_box = gr.Textbox(
                label="Texte ajusté",
                lines=3,
                max_lines=16,
                interactive=False,
                placeholder="Le texte ajusté pour le TTS apparaîtra ici...",
            )
            adjust_log_box = gr.Markdown("", elem_classes=["inline-info"])
            with gr.Row():
                target_duration = gr.Number(label="Durée cible (s)", value=None, precision=1, show_label=False)
                adjust_btn = gr.Button("Ajuster le texte")
                apply_btn = gr.Button("Utiliser la suggestion")
            adjusted_preview = gr.Textbox(
                label="Suggestion texte",
                lines=3,
                max_lines=16,
                interactive=False,
                placeholder="Le texte ajusté apparaîtra ici...",
            )
            adjust_info = gr.Markdown("Durée estimée: --", elem_classes=["inline-info"])
            gr.Markdown("### Modèle / Langue", elem_classes=["subhead"])
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
                elem_classes=["inline-info"],
                visible=not show_lang_default,
            )
            voice_label_md = gr.Markdown(
                default_voice_label_text,
                elem_classes=["inline-info"],
                visible=default_voice_label_visible,
            )
            piper_voice_status_md = gr.Markdown(
                piper_voice_status_text(voices) if default_tts_engine == "piper" else "",
                elem_classes=["inline-info"],
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
                elem_classes=["inline-info"],
                visible=default_tts_engine == "piper",
            )
            default_supports_speed = bool(
                default_tts_engine == "piper"
                and default_voice_value
                and piper_voice_supports_length_scale(default_voice_value)
            )
            piper_speed_note_md = gr.Markdown(
                "Vitesse non supportée par cette voix",
                elem_classes=["inline-info"],
                visible=default_tts_engine == "piper" and bool(default_voice_value) and not default_supports_speed,
            )
            if speed_spec:
                speed_widget = create_param_widget(speed_spec, speed_value, speed_visible)
                param_widgets["speed"] = speed_widget
            warnings_md = gr.Markdown("", elem_classes=["inline-info"], visible=False)
            gr.Markdown("### Paramètres moteur", elem_classes=["subhead"])
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
            engine_status_md = gr.Markdown(
                engine_status_markdown(default_tts_engine),
                elem_classes=["inline-info"],
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
            install_logs_box = gr.Textbox(
                label="Logs installation",
                lines=4,
                max_lines=16,
                interactive=False,
            )
            gr.Markdown("### Pauses automatiques (ponctuation)", elem_classes=["subhead"])
            with gr.Row():
                comma_pause_slider = gr.Slider(
                    0,
                    1000,
                    value=default_comma_pause,
                    step=50,
                    label="Pause après virgule (ms)",
                    info="300ms = 0,3s",
                )
                period_pause_slider = gr.Slider(
                    0,
                    2000,
                    value=default_period_pause,
                    step=50,
                    label="Pause après point (ms)",
                    info="500ms = 0,5s",
                )
                semicolon_pause_slider = gr.Slider(
                    0,
                    2000,
                    value=default_semicolon_pause,
                    step=50,
                    label="Pause après point-virgule (ms)",
                )
            with gr.Row():
                colon_pause_slider = gr.Slider(
                    0,
                    2000,
                    value=default_colon_pause,
                    step=50,
                    label="Pause après deux-points (ms)",
                )
                dash_pause_slider = gr.Slider(
                    0,
                    1500,
                    value=default_dash_pause,
                    step=50,
                    label="Pause après tiret (ms)",
                )
                newline_pause_slider = gr.Slider(
                    0,
                    4000,
                    value=default_newline_pause,
                    step=100,
                    label="Pause après retour ligne (ms)",
                    info="1000ms = 1s",
                )
            duration_preview = gr.Markdown("Durée estimée (avec pauses): --", elem_classes=["inline-info"])
            with gr.Accordion("Voir le texte final", open=False):
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
            with gr.Accordion("Pré-chunking", open=False):
                with gr.Row():
                    min_words_slider = gr.Slider(
                        1,
                        20,
                        value=default_min_words_per_chunk,
                        step=1,
                        label="Mots minimum par chunk",
                        info="Augmenter améliore la stabilité prosodique, réduit les coupures.",
                    )
                    max_words_without_term_slider = gr.Slider(
                        6,
                        80,
                        value=default_max_words_without_terminator,
                        step=1,
                        label="Max mots sans terminator",
                        info="Seuil de fallback quand aucune fin de phrase n'est détectée.",
                    )
                    max_est_seconds_slider = gr.Slider(
                        4.0,
                        30.0,
                        value=default_max_est_seconds,
                        step=0.5,
                        label="Durée max/chunk (s)",
                        info="Garde-fou contre les dérives (10s recommandé).",
                    )
                    chunk_apply_btn = gr.Button("Appliquer", size="sm")
                    reset_chunk_btn = gr.Button("↺", size="sm")
                    verbose_logs_toggle = gr.Checkbox(
                        label="Logs détaillés",
                        value=default_verbose_logs,
                    )
                    disable_newline_chunking_toggle = gr.Checkbox(
                        label="Désactiver découpe auto sur retour ligne",
                        value=default_disable_newline_chunking,
                    )
                chunk_status = gr.Markdown("Etat: non appliqué")
                chunk_preview_box = gr.Textbox(
                    label="Aperçu des chunks",
                    lines=4,
                    max_lines=16,
                    interactive=False,
                    placeholder="Aperçu du pré-chunking.",
                )


        with gr.Group():
            gr.Markdown("## Traitement audio", elem_classes=["section-title"])
            with gr.Row():
                fade_slider = gr.Slider(
                    0,
                    200,
                    value=default_fade_ms,
                    step=1,
                    label="Fade (ms)",
                    info="Fondu doux sur les coupes.",
                )
                zero_cross_slider = gr.Slider(
                    0,
                    50,
                    value=default_zero_cross_radius_ms,
                    step=1,
                    label="Zero-cross radius (ms)",
                )
            with gr.Row():
                silence_threshold_slider = gr.Slider(
                    0.0,
                    0.1,
                    value=default_silence_threshold,
                    step=0.0005,
                    label="Silence threshold",
                    info="Amplitude max pour considérer un silence.",
                )
                silence_min_ms_slider = gr.Slider(
                    0,
                    500,
                    value=default_silence_min_ms,
                    step=5,
                    label="Silence min (ms)",
                    info="Durée min d'un silence pour appliquer le fade.",
                )

        with gr.Group():
            gr.Markdown("## Sortie", elem_classes=["section-title"])
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
            with gr.Row():
                generate_btn = gr.Button(
                    "Générer",
                    variant="primary",
                    interactive=backend_status(default_tts_engine).get("installed", True),
                )
                stop_btn = gr.Button("STOP", variant="secondary")
            result_audio = gr.Audio(label="Pré-écoute", type="filepath", autoplay=False)
            output_path_box = gr.Textbox(label="Fichier généré", interactive=False)

        logs_box = gr.Textbox(label="Logs", lines=4, max_lines=16, interactive=False)
        chunk_state = gr.State({"applied": False, "chunks": [], "signature": None})
        confirm_state = gr.State({"pending": None, "ts": 0.0})

        refresh_btn.click(refresh_dropdown, inputs=ref_dropdown, outputs=ref_dropdown)
        upload.upload(
            fn=handle_upload,
            inputs=[upload, ref_dropdown, logs_box],
            outputs=[ref_dropdown, logs_box],
        )
        adjust_btn.click(
            fn=handle_adjust,
            inputs=[text_input, target_duration, logs_box],
            outputs=[adjusted_preview, adjust_info, logs_box],
        )
        apply_btn.click(fn=apply_adjusted, inputs=adjusted_preview, outputs=text_input)
        text_input.change(
            fn=handle_text_adjustment,
            inputs=[
                text_input,
                auto_adjust_toggle,
                show_adjust_log_toggle,
                comma_pause_slider,
                period_pause_slider,
                semicolon_pause_slider,
                colon_pause_slider,
                dash_pause_slider,
                newline_pause_slider,
            ],
            outputs=[
                adjusted_text_box,
                clean_text_box,
                duration_preview,
                adjust_log_box,
            ],
        )
        auto_adjust_toggle.change(
            fn=handle_text_adjustment,
            inputs=[
                text_input,
                auto_adjust_toggle,
                show_adjust_log_toggle,
                comma_pause_slider,
                period_pause_slider,
                semicolon_pause_slider,
                colon_pause_slider,
                dash_pause_slider,
                newline_pause_slider,
            ],
            outputs=[
                adjusted_text_box,
                clean_text_box,
                duration_preview,
                adjust_log_box,
            ],
        )
        show_adjust_log_toggle.change(
            fn=handle_text_adjustment,
            inputs=[
                text_input,
                auto_adjust_toggle,
                show_adjust_log_toggle,
                comma_pause_slider,
                period_pause_slider,
                semicolon_pause_slider,
                colon_pause_slider,
                dash_pause_slider,
                newline_pause_slider,
            ],
            outputs=[
                adjusted_text_box,
                clean_text_box,
                duration_preview,
                adjust_log_box,
            ],
        )
        comma_pause_slider.change(
            fn=update_estimated_duration,
            inputs=[
                adjusted_text_box,
                comma_pause_slider,
                period_pause_slider,
                semicolon_pause_slider,
                colon_pause_slider,
                dash_pause_slider,
                newline_pause_slider,
            ],
            outputs=duration_preview,
        )
        period_pause_slider.change(
            fn=update_estimated_duration,
            inputs=[
                adjusted_text_box,
                comma_pause_slider,
                period_pause_slider,
                semicolon_pause_slider,
                colon_pause_slider,
                dash_pause_slider,
                newline_pause_slider,
            ],
            outputs=duration_preview,
        )
        semicolon_pause_slider.change(
            fn=update_estimated_duration,
            inputs=[
                adjusted_text_box,
                comma_pause_slider,
                period_pause_slider,
                semicolon_pause_slider,
                colon_pause_slider,
                dash_pause_slider,
                newline_pause_slider,
            ],
            outputs=duration_preview,
        )
        colon_pause_slider.change(
            fn=update_estimated_duration,
            inputs=[
                adjusted_text_box,
                comma_pause_slider,
                period_pause_slider,
                semicolon_pause_slider,
                colon_pause_slider,
                dash_pause_slider,
                newline_pause_slider,
            ],
            outputs=duration_preview,
        )
        dash_pause_slider.change(
            fn=update_estimated_duration,
            inputs=[
                adjusted_text_box,
                comma_pause_slider,
                period_pause_slider,
                semicolon_pause_slider,
                colon_pause_slider,
                dash_pause_slider,
                newline_pause_slider,
            ],
            outputs=duration_preview,
        )
        newline_pause_slider.change(
            fn=update_estimated_duration,
            inputs=[
                adjusted_text_box,
                comma_pause_slider,
                period_pause_slider,
                semicolon_pause_slider,
                colon_pause_slider,
                dash_pause_slider,
                newline_pause_slider,
            ],
            outputs=duration_preview,
        )
        param_widget_list = [param_widgets[key] for key in all_param_keys()]
        engine_dropdown.change(
            fn=handle_engine_change,
            inputs=[
                engine_dropdown,
                language_dropdown,
                chatterbox_mode_dropdown,
                chunk_state,
            ],
            outputs=[
                language_dropdown,
                lang_locked_md,
                chunk_state,
                chunk_status,
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
                *param_widget_list,
            ],
        )
        language_dropdown.change(
            fn=handle_language_change,
            inputs=[
                engine_dropdown,
                language_dropdown,
                chatterbox_mode_dropdown,
                chunk_state,
            ],
            outputs=[
                language_dropdown,
                lang_locked_md,
                warnings_md,
                chunk_state,
                chunk_status,
                *param_widget_list,
            ],
        )
        chatterbox_mode_dropdown.change(
            fn=handle_chatterbox_mode_change,
            inputs=[
                chatterbox_mode_dropdown,
                language_dropdown,
                chunk_state,
            ],
            outputs=[
                language_dropdown,
                lang_locked_md,
                warnings_md,
                chunk_state,
                chunk_status,
                *param_widget_list,
            ],
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
        )
        if "voice_id" in param_widgets:
            param_widgets["voice_id"].change(
                fn=handle_voice_change,
                inputs=[
                    engine_dropdown,
                    param_widgets["voice_id"],
                    language_dropdown,
                    chatterbox_mode_dropdown,
                    chunk_state,
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
        )
        text_input.change(
            fn=mark_chunk_dirty,
            inputs=[chunk_state],
            outputs=[chunk_state, chunk_status],
        )
        auto_adjust_toggle.change(
            fn=mark_chunk_dirty,
            inputs=[chunk_state],
            outputs=[chunk_state, chunk_status],
        )
        min_words_slider.change(
            fn=mark_chunk_dirty,
            inputs=[chunk_state],
            outputs=[chunk_state, chunk_status],
        )
        max_words_without_term_slider.change(
            fn=mark_chunk_dirty,
            inputs=[chunk_state],
            outputs=[chunk_state, chunk_status],
        )
        max_est_seconds_slider.change(
            fn=mark_chunk_dirty,
            inputs=[chunk_state],
            outputs=[chunk_state, chunk_status],
        )
        disable_newline_chunking_toggle.change(
            fn=mark_chunk_dirty,
            inputs=[chunk_state],
            outputs=[chunk_state, chunk_status],
        )

        choose_btn.click(
            fn=handle_choose_output,
            inputs=[output_dir_box, logs_box],
            outputs=[output_dir_box, logs_box],
        )
        reset_chunk_btn.click(
            fn=handle_reset_chunk_defaults,
            inputs=[logs_box],
            outputs=[
                min_words_slider,
                max_words_without_term_slider,
                max_est_seconds_slider,
                disable_newline_chunking_toggle,
                logs_box,
            ],
        )
        verbose_logs_toggle.change(
            fn=handle_toggle_verbose_logs,
            inputs=[verbose_logs_toggle, logs_box],
            outputs=[logs_box],
        )
        chunk_apply_btn.click(
            fn=handle_apply_prechunk,
            inputs=[
                text_input,
                adjusted_text_box,
                auto_adjust_toggle,
                min_words_slider,
                max_words_without_term_slider,
                max_est_seconds_slider,
                disable_newline_chunking_toggle,
                chunk_state,
                logs_box,
                verbose_logs_toggle,
            ],
            outputs=[chunk_preview_box, chunk_status, chunk_state, logs_box],
        )
        load_preset_btn.click(
            fn=handle_load_preset,
            inputs=[preset_dropdown, logs_box],
            outputs=[
                ref_dropdown,
                ref_support_note,
                output_dir_box,
                filename_box,
                timestamp_toggle,
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
                *param_widget_list,
                comma_pause_slider,
                period_pause_slider,
                semicolon_pause_slider,
                colon_pause_slider,
                dash_pause_slider,
                newline_pause_slider,
                min_words_slider,
                max_words_without_term_slider,
                max_est_seconds_slider,
                disable_newline_chunking_toggle,
                verbose_logs_toggle,
                fade_slider,
                zero_cross_slider,
                silence_threshold_slider,
                silence_min_ms_slider,
                preset_dropdown,
                preset_name_box,
                chunk_status,
                chunk_state,
                logs_box,
            ],
        )
        save_preset_btn.click(
            fn=handle_save_preset_confirm,
            inputs=[
                preset_name_box,
                ref_dropdown,
                output_dir_box,
                filename_box,
                timestamp_toggle,
                language_dropdown,
                engine_dropdown,
                comma_pause_slider,
                period_pause_slider,
                semicolon_pause_slider,
                colon_pause_slider,
                dash_pause_slider,
                newline_pause_slider,
                min_words_slider,
                max_words_without_term_slider,
                max_est_seconds_slider,
                disable_newline_chunking_toggle,
                verbose_logs_toggle,
                fade_slider,
                zero_cross_slider,
                silence_threshold_slider,
                silence_min_ms_slider,
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
                engine_dropdown,
                language_dropdown,
                comma_pause_slider,
                period_pause_slider,
                semicolon_pause_slider,
                colon_pause_slider,
                dash_pause_slider,
                newline_pause_slider,
                min_words_slider,
                max_words_without_term_slider,
                max_est_seconds_slider,
                disable_newline_chunking_toggle,
                verbose_logs_toggle,
                fade_slider,
                zero_cross_slider,
                silence_threshold_slider,
                silence_min_ms_slider,
                chunk_state,
                logs_box,
                *param_widget_list,
            ],
            outputs=[
                adjusted_text_box,
                result_audio,
                output_path_box,
                chunk_preview_box,
                chunk_status,
                chunk_state,
                logs_box,
            ],
        )
        stop_btn.click(
            fn=handle_stop_confirm,
            inputs=[confirm_state, logs_box],
            outputs=[
                result_audio,
                output_path_box,
                logs_box,
                confirm_state,
                save_preset_btn,
                delete_preset_btn,
                stop_btn,
            ],
        )

    return demo


def main() -> None:
    demo = build_ui()
    port_env = os.environ.get("GRADIO_SERVER_PORT")
    launch_kwargs = {}
    if port_env:
        try:
            launch_kwargs["server_port"] = int(port_env)
        except ValueError:
            LOGGER.warning("GRADIO_SERVER_PORT invalide (%s), fallback 7860.", port_env)
    demo.launch(**launch_kwargs)


if __name__ == "__main__":
    main()
