"""Gradio interface for the local Chatterbox French TTS tool."""

from __future__ import annotations

import datetime as dt
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
    adjust_text_to_duration,
    chunk_script,
    estimate_duration_with_pauses,
    normalize_text,
    render_clean_text,
)
from tts_engine import (
    FADE_MS,
    LANGUAGE_MAP,
    SILENCE_MIN_MS,
    SILENCE_THRESHOLD,
    TTSEngine,
    ZERO_CROSS_RADIUS_MS,
)


logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("chatterbox_app")

BASE_DIR = Path(__file__).resolve().parent

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
LANGUAGE_CHOICES = [
    ("Français (fr-FR)", "fr-FR"),
    ("English US (en-US)", "en-US"),
    ("English UK (en-GB)", "en-GB"),
    ("Español (es-ES)", "es-ES"),
    ("Deutsch (de-DE)", "de-DE"),
    ("Italiano (it-IT)", "it-IT"),
    ("Português (pt-PT)", "pt-PT"),
    ("Nederlands (nl-NL)", "nl-NL"),
]


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
        engine = TTSEngine()
        _, _, meta = engine.generate_longform(**payload)
        result_queue.put({"status": "ok", "meta": meta})
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


def handle_load_preset(
    preset_name: str | None,
    log_text: str | None,
):
    outputs = [gr.update() for _ in range(26)]
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

    model_mode = data.get("tts_model_mode", "fr_finetune")
    language = coerce_tts_language(model_mode, data.get("tts_language", "fr-FR"))
    updates = [
        gr.update(value=ref_value),
        gr.update(value=data.get("out_dir") or str(DEFAULT_OUTPUT_DIR)),
        gr.update(value=data.get("user_filename", "")),
        gr.update(value=bool(data.get("add_timestamp", True))),
        gr.update(value=model_mode),
        gr.update(
            value=language,
            visible=(model_mode == "multilang"),
            interactive=(model_mode == "multilang"),
        ),
        gr.update(
            value=float(data.get("multilang_cfg_weight", 0.5)),
            visible=(model_mode == "multilang"),
            interactive=(model_mode == "multilang"),
        ),
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
        gr.update(value=bool(data.get("verbose_logs", False))),
        gr.update(value=_coerce_float(data.get("exaggeration"), 0.5)),
        gr.update(value=_coerce_float(data.get("cfg_weight"), 0.6)),
        gr.update(value=_coerce_float(data.get("temperature"), 0.5)),
        gr.update(value=_coerce_float(data.get("repetition_penalty"), 1.35)),
        gr.update(value=int(data.get("fade_ms", FADE_MS))),
        gr.update(value=int(data.get("zero_cross_radius_ms", ZERO_CROSS_RADIUS_MS))),
        gr.update(value=_coerce_float(data.get("silence_threshold"), SILENCE_THRESHOLD)),
        gr.update(value=int(data.get("silence_min_ms", SILENCE_MIN_MS))),
        gr.update(value=preset_name),
    ]
    name_update = gr.update(value=preset_name)
    persist_state({"last_preset": preset_name})
    log_text = append_log(f"Preset chargé: {preset_name}", log_text)
    return (*updates, name_update, chunk_status, chunk_state, log_text)


def handle_save_preset(
    preset_name: str | None,
    ref_name: str | None,
    out_dir: str | None,
    user_filename: str | None,
    add_timestamp: bool,
    tts_model_mode: str,
    tts_language: str | None,
    multilang_cfg_weight: float,
    comma_pause_ms: int,
    period_pause_ms: int,
    semicolon_pause_ms: int,
    colon_pause_ms: int,
    dash_pause_ms: int,
    newline_pause_ms: int,
    min_words_per_chunk: int,
    max_words_without_terminator: int,
    max_est_seconds_per_chunk: float,
    verbose_logs: bool,
    exaggeration: float,
    cfg_weight: float,
    temperature: float,
    repetition_penalty: float,
    fade_ms: int,
    zero_cross_radius_ms: int,
    silence_threshold: float,
    silence_min_ms: int,
    log_text: str | None,
):
    if not preset_name:
        log_text = append_log("Nom de preset requis.", log_text)
        return gr.update(), gr.update(), log_text

    data = {
        "ref_name": ref_name,
        "out_dir": out_dir,
        "user_filename": user_filename or "",
        "add_timestamp": bool(add_timestamp),
        "tts_model_mode": str(tts_model_mode),
        "tts_language": str(coerce_tts_language(tts_model_mode, tts_language)),
        "multilang_cfg_weight": float(multilang_cfg_weight),
        "comma_pause_ms": int(comma_pause_ms),
        "period_pause_ms": int(period_pause_ms),
        "semicolon_pause_ms": int(semicolon_pause_ms),
        "colon_pause_ms": int(colon_pause_ms),
        "dash_pause_ms": int(dash_pause_ms),
        "newline_pause_ms": int(newline_pause_ms),
        "min_words_per_chunk": int(min_words_per_chunk),
        "max_words_without_terminator": int(max_words_without_terminator),
        "max_est_seconds_per_chunk": float(max_est_seconds_per_chunk),
        "verbose_logs": bool(verbose_logs),
        "exaggeration": float(exaggeration),
        "cfg_weight": float(cfg_weight),
        "temperature": float(temperature),
        "repetition_penalty": float(repetition_penalty),
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
        log_text,
    )


def handle_toggle_verbose_logs(verbose: bool, log_text: str | None):
    set_verbosity(bool(verbose))
    persist_state({"last_verbose_logs": bool(verbose)})
    log_text = append_ui_log("Verbosity terminal mise à jour.", log_text)
    return log_text


def coerce_tts_language(mode: str, language: str | None) -> str:
    if mode == "fr_finetune":
        return "fr-FR"
    return language or "fr-FR"


def handle_model_change(
    mode: str, language: str | None, multilang_cfg_weight: float, chunk_state: dict | None
):
    coerced = coerce_tts_language(mode, language)
    show_language = mode == "multilang"
    lang_update = gr.update(value=coerced, visible=show_language, interactive=show_language)
    cfg_update = gr.update(
        value=float(multilang_cfg_weight), visible=show_language, interactive=show_language
    )
    state, status = mark_chunk_dirty(chunk_state)
    return lang_update, cfg_update, state, status


def _chunk_signature(
    text: str,
    min_words_per_chunk: int,
    max_words_without_terminator: int,
    max_est_seconds_per_chunk: float,
) -> tuple:
    return (
        text.strip(),
        int(min_words_per_chunk),
        int(max_words_without_terminator),
        float(max_est_seconds_per_chunk),
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
    min_words_per_chunk: int,
    max_words_without_terminator: int,
    max_est_seconds_per_chunk: float,
    chunk_state: dict | None,
    log_text: str | None,
    verbose_logs: bool,
):
    if not text or not text.strip():
        return "", "Etat: non appliqué", {"applied": False, "chunks": [], "signature": None}, log_text
    normalized_text = normalize_text(text)
    chunks = chunk_script(
        normalized_text,
        min_words_per_chunk=int(min_words_per_chunk),
        max_words_without_terminator=int(max_words_without_terminator),
        max_est_seconds_per_chunk=float(max_est_seconds_per_chunk),
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
    ref_name: str | None,
    out_dir: str | None,
    user_filename: str | None,
    add_timestamp: bool,
    tts_model_mode: str,
    tts_language: str | None,
    multilang_cfg_weight: float,
    comma_pause_ms: int,
    period_pause_ms: int,
    semicolon_pause_ms: int,
    colon_pause_ms: int,
    dash_pause_ms: int,
    newline_pause_ms: int,
    min_words_per_chunk: int,
    max_words_without_terminator: int,
    max_est_seconds_per_chunk: float,
    verbose_logs: bool,
    exaggeration: float,
    cfg_weight: float,
    temperature: float,
    repetition_penalty: float,
    fade_ms: int,
    zero_cross_radius_ms: int,
    silence_threshold: float,
    silence_min_ms: int,
    chunk_state: dict | None,
    log_text: str | None,
):
    if not text or not text.strip():
        state = chunk_state or {"applied": False, "chunks": [], "signature": None}
        return None, "", "", "Etat: non appliqué", state, append_log("Erreur: texte vide.", log_text)

    log_text = append_ui_log("Initialisation de la génération...", log_text)

    audio_prompt = None
    if ref_name:
        try:
            audio_prompt = resolve_ref_path(ref_name)
        except FileNotFoundError:
            log_text = append_ui_log(f"Référence introuvable: {ref_name}", log_text)
            return None, "", "", log_text

    normalized_text = normalize_text(text)
    if normalized_text != text and verbose_logs:
        before = len(re.findall(r"\bII\b", text))
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

    tts_language = coerce_tts_language(tts_model_mode, tts_language)
    backend_language = tts_language
    effective_cfg = cfg_weight
    if tts_model_mode == "multilang":
        backend_language = LANGUAGE_MAP.get(tts_language, "fr")
        effective_cfg = float(multilang_cfg_weight)
        log_text = append_ui_log(
            f"requested_language_bcp47={tts_language}",
            log_text,
            verbose=False,
            enabled=True,
        )
        log_text = append_ui_log(
            f"backend_language_id={backend_language}",
            log_text,
            verbose=False,
            enabled=True,
        )
    log_text = append_ui_log(
        f"voice_mode={tts_model_mode}",
        log_text,
        verbose=False,
        enabled=True,
    )
    log_text = append_ui_log(
        f"cfg_weight={effective_cfg}",
        log_text,
        verbose=False,
        enabled=True,
    )

    chunk_preview_text = ""
    signature = _chunk_signature(
        normalized_text,
        min_words_per_chunk,
        max_words_without_terminator,
        max_est_seconds_per_chunk,
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
        "script": normalized_text,
        "chunks": chunks,
        "audio_prompt_path": audio_prompt,
        "out_path": str(tmp_path),
        "exaggeration": exaggeration,
        "cfg_weight": cfg_weight,
        "temperature": temperature,
        "repetition_penalty": repetition_penalty,
        "tts_model_mode": str(tts_model_mode),
        "tts_language": str(tts_language),
        "multilang_cfg_weight": float(multilang_cfg_weight),
        "comma_pause_ms": int(comma_pause_ms),
        "period_pause_ms": int(period_pause_ms),
        "semicolon_pause_ms": int(semicolon_pause_ms),
        "colon_pause_ms": int(colon_pause_ms),
        "dash_pause_ms": int(dash_pause_ms),
        "newline_pause_ms": int(newline_pause_ms),
        "min_words_per_chunk": int(min_words_per_chunk),
        "max_words_without_terminator": int(max_words_without_terminator),
        "max_est_seconds_per_chunk": float(max_est_seconds_per_chunk),
        "fade_ms": int(fade_ms),
        "zero_cross_radius_ms": int(zero_cross_radius_ms),
        "silence_threshold": float(silence_threshold),
        "silence_min_ms": int(silence_min_ms),
    }
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
        if result and result.get("status") == "error":
            log_text = append_ui_log(f"Erreur TTS: {result.get('error')}", log_text)
        else:
            log_text = append_ui_log("Annulé.", log_text)
        return None, "", chunk_preview_text, chunk_status, chunk_state, log_text

    meta = result.get("meta", {})
    preview_path_obj = Path(preview_path)
    try:
        preview_path_obj.parent.mkdir(parents=True, exist_ok=True)
        if tmp_path.exists():
            os.replace(tmp_path, preview_path_obj)
        else:
            _reset_job_state()
            log_text = append_ui_log("Annulé.", log_text)
            return None, "", chunk_preview_text, chunk_status, chunk_state, log_text
    except Exception as exc:
        _cleanup_tmp(str(tmp_path))
        _reset_job_state()
        log_text = append_ui_log(f"Erreur TTS: {exc}", log_text)
        return None, "", chunk_preview_text, chunk_status, chunk_state, log_text

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
            "last_exaggeration": float(exaggeration),
            "last_cfg_weight": float(cfg_weight),
            "last_temperature": float(temperature),
            "last_repetition_penalty": float(repetition_penalty),
            "last_fade_ms": int(fade_ms),
            "last_zero_cross_radius_ms": int(zero_cross_radius_ms),
            "last_silence_threshold": float(silence_threshold),
            "last_silence_min_ms": int(silence_min_ms),
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
            "last_verbose_logs": bool(verbose_logs),
            "last_tts_model_mode": str(tts_model_mode),
            "last_tts_language": str(tts_language),
            "last_multilang_cfg_weight": float(multilang_cfg_weight),
        }
    )
    log_text = append_ui_log(f"Fichier pré-écoute: {preview_path_obj}", log_text)
    return (
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
    tts_model_mode: str,
    tts_language: str | None,
    multilang_cfg_weight: float,
    comma_pause_ms: int,
    period_pause_ms: int,
    semicolon_pause_ms: int,
    colon_pause_ms: int,
    dash_pause_ms: int,
    newline_pause_ms: int,
    min_words_per_chunk: int,
    max_words_without_terminator: int,
    max_est_seconds_per_chunk: float,
    verbose_logs: bool,
    exaggeration: float,
    cfg_weight: float,
    temperature: float,
    repetition_penalty: float,
    fade_ms: int,
    zero_cross_radius_ms: int,
    silence_threshold: float,
    silence_min_ms: int,
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
        tts_model_mode,
        tts_language,
        multilang_cfg_weight,
        comma_pause_ms,
        period_pause_ms,
        semicolon_pause_ms,
        colon_pause_ms,
        dash_pause_ms,
        newline_pause_ms,
        min_words_per_chunk,
        max_words_without_terminator,
        max_est_seconds_per_chunk,
        verbose_logs,
        exaggeration,
        cfg_weight,
        temperature,
        repetition_penalty,
        fade_ms,
        zero_cross_radius_ms,
        silence_threshold,
        silence_min_ms,
        log_text,
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
    default_verbose_logs = _coerce_bool(state_data.get("last_verbose_logs"), False)
    default_tts_model_mode = state_data.get("last_tts_model_mode") or _state_or_preset(
        "tts_model_mode", "fr_finetune"
    )
    default_tts_language = state_data.get("last_tts_language") or _state_or_preset(
        "tts_language", "fr-FR"
    )
    default_tts_language = coerce_tts_language(default_tts_model_mode, default_tts_language)
    default_multilang_cfg_weight = _coerce_float(
        state_data.get("last_multilang_cfg_weight")
        or _state_or_preset("multilang_cfg_weight", 0.5),
        0.5,
    )
    default_exaggeration = _coerce_float(_state_or_preset("exaggeration", 0.5), 0.5)
    default_cfg = _coerce_float(_state_or_preset("cfg_weight", 0.6), 0.6)
    default_temperature = _coerce_float(_state_or_preset("temperature", 0.5), 0.5)
    default_repetition = _coerce_float(_state_or_preset("repetition_penalty", 1.35), 1.35)
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
                )
                refresh_btn = gr.Button("Refresh", size="sm")
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
                model_mode_dropdown = gr.Dropdown(
                    label="Modèle",
                    choices=[
                        ("FR fine-tuné (spécialisé)", "fr_finetune"),
                        ("Chatterbox multilangue", "multilang"),
                    ],
                    value=default_tts_model_mode,
                )
                language_dropdown = gr.Dropdown(
                    label="Langue",
                    choices=LANGUAGE_CHOICES,
                    value=default_tts_language,
                    visible=default_tts_model_mode == "multilang",
                    interactive=default_tts_model_mode == "multilang",
                )
                multilang_cfg_slider = gr.Slider(
                    0.0,
                    1.0,
                    value=default_multilang_cfg_weight,
                    step=0.05,
                    label="CFG multilangue",
                    info="Réduire pour limiter l'accent bleed en cross-language.",
                    visible=default_tts_model_mode == "multilang",
                    interactive=default_tts_model_mode == "multilang",
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
                        20.0,
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
                chunk_status = gr.Markdown("Etat: non appliqué")
                chunk_preview_box = gr.Textbox(
                    label="Aperçu des chunks",
                    lines=4,
                    max_lines=16,
                    interactive=False,
                    placeholder="Aperçu du pré-chunking.",
                )

        with gr.Group():
            gr.Markdown("## Paramètres modèle", elem_classes=["section-title"])
            with gr.Row():
                exaggeration_slider = gr.Slider(
                    0.0,
                    1.5,
                    value=default_exaggeration,
                    step=0.05,
                    label="Exagération",
                    info="Expressivité globale (0.5 recommandé)",
                )
                cfg_slider = gr.Slider(
                    0.0,
                    1.0,
                    value=default_cfg,
                    step=0.05,
                    label="CFG",
                    info="Plus haut = voix plus rigoureuse",
                )
            with gr.Row():
                temperature_slider = gr.Slider(
                    0.1,
                    1.0,
                    value=default_temperature,
                    step=0.05,
                    label="Température",
                    info="Stabilité vs créativité",
                )
                repetition_slider = gr.Slider(
                    0.8,
                    2.0,
                    value=default_repetition,
                    step=0.05,
                    label="Repetition Penalty",
                    info="Limite les répétitions",
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
                generate_btn = gr.Button("Générer", variant="primary")
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
        text_input.change(fn=update_clean_preview, inputs=text_input, outputs=clean_text_box)
        text_input.change(
            fn=update_estimated_duration,
            inputs=[
                text_input,
                comma_pause_slider,
                period_pause_slider,
                semicolon_pause_slider,
                colon_pause_slider,
                dash_pause_slider,
                newline_pause_slider,
            ],
            outputs=duration_preview,
        )
        comma_pause_slider.change(
            fn=update_estimated_duration,
            inputs=[
                text_input,
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
                text_input,
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
                text_input,
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
                text_input,
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
                text_input,
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
                text_input,
                comma_pause_slider,
                period_pause_slider,
                semicolon_pause_slider,
                colon_pause_slider,
                dash_pause_slider,
                newline_pause_slider,
            ],
            outputs=duration_preview,
        )
        model_mode_dropdown.change(
            fn=handle_model_change,
            inputs=[model_mode_dropdown, language_dropdown, multilang_cfg_slider, chunk_state],
            outputs=[language_dropdown, multilang_cfg_slider, chunk_state, chunk_status],
        )
        language_dropdown.change(
            fn=mark_chunk_dirty,
            inputs=[chunk_state],
            outputs=[chunk_state, chunk_status],
        )
        text_input.change(
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
                min_words_slider,
                max_words_without_term_slider,
                max_est_seconds_slider,
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
                output_dir_box,
                filename_box,
                timestamp_toggle,
                model_mode_dropdown,
                language_dropdown,
                multilang_cfg_slider,
                comma_pause_slider,
                period_pause_slider,
                semicolon_pause_slider,
                colon_pause_slider,
                dash_pause_slider,
                newline_pause_slider,
                min_words_slider,
                max_words_without_term_slider,
                max_est_seconds_slider,
                verbose_logs_toggle,
                exaggeration_slider,
                cfg_slider,
                temperature_slider,
                repetition_slider,
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
                model_mode_dropdown,
                language_dropdown,
                multilang_cfg_slider,
                comma_pause_slider,
                period_pause_slider,
                semicolon_pause_slider,
                colon_pause_slider,
                dash_pause_slider,
                newline_pause_slider,
                min_words_slider,
                max_words_without_term_slider,
                max_est_seconds_slider,
                verbose_logs_toggle,
                exaggeration_slider,
                cfg_slider,
                temperature_slider,
                repetition_slider,
                fade_slider,
                zero_cross_slider,
                silence_threshold_slider,
                silence_min_ms_slider,
                confirm_state,
                logs_box,
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
                ref_dropdown,
                output_dir_box,
                filename_box,
                timestamp_toggle,
                model_mode_dropdown,
                language_dropdown,
                multilang_cfg_slider,
                comma_pause_slider,
                period_pause_slider,
                semicolon_pause_slider,
                colon_pause_slider,
                dash_pause_slider,
                newline_pause_slider,
                min_words_slider,
                max_words_without_term_slider,
                max_est_seconds_slider,
                verbose_logs_toggle,
                exaggeration_slider,
                cfg_slider,
                temperature_slider,
                repetition_slider,
                fade_slider,
                zero_cross_slider,
                silence_threshold_slider,
                silence_min_ms_slider,
                chunk_state,
                logs_box,
            ],
            outputs=[
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
