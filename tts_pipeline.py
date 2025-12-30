"""Shared TTS pipeline for all backends."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import soundfile as sf

from tts_backends import get_backend
from tts_backends.base import BackendUnavailableError
from tts_engine import _start_at_zero_and_fade_in, _trim_to_zero_and_fade_out
from text_tools import (
    DEFAULT_MAX_EST_SECONDS_PER_CHUNK,
    DEFAULT_MAX_WORDS_WITHOUT_TERMINATOR,
    DEFAULT_MIN_WORDS_PER_CHUNK,
    ChunkInfo,
    chunk_script,
    ensure_strong_ending,
    render_clean_text_from_segments,
    stabilize_trailing_punct,
    strip_legacy_tokens,
)


TARGET_SR = 24000
_PUNCT_TOKENS = {",", ".", "!", "?", ";", ":", "—", "-", "\n"}


@dataclass(frozen=True)
class PipelineResult:
    out_path: str
    meta: dict[str, Any]


def _resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return audio
    if audio.ndim == 1:
        return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
    channels = []
    for idx in range(audio.shape[1]):
        channels.append(librosa.resample(audio[:, idx], orig_sr=orig_sr, target_sr=target_sr))
    min_len = min(len(ch) for ch in channels) if channels else 0
    if min_len == 0:
        return np.zeros(0, dtype=np.float32)
    return np.stack([ch[:min_len] for ch in channels], axis=1)


def _apply_post_processing(
    audio: np.ndarray,
    sr: int,
    zero_cross_radius_ms: int,
    fade_ms: int,
    silence_threshold: float,
    silence_min_ms: int,
) -> np.ndarray:
    def _trim_edges(channel: np.ndarray) -> np.ndarray:
        if channel.size == 0:
            return channel
        threshold = float(silence_threshold)
        mask = np.abs(channel) > threshold
        if not np.any(mask):
            return channel
        start = int(np.argmax(mask))
        end = len(channel) - int(np.argmax(mask[::-1]))
        trimmed = channel[start:end].copy()
        trimmed = _start_at_zero_and_fade_in(
            trimmed,
            sr,
            radius_ms=int(zero_cross_radius_ms),
            fade_ms=int(fade_ms),
        )
        trimmed = _trim_to_zero_and_fade_out(
            trimmed,
            sr,
            radius_ms=int(zero_cross_radius_ms),
            fade_ms=int(fade_ms),
        )
        return trimmed

    if audio.ndim == 1:
        return _trim_edges(audio)
    channels = []
    for idx in range(audio.shape[1]):
        channels.append(_trim_edges(audio[:, idx]))
    min_len = min(len(ch) for ch in channels) if channels else 0
    if min_len == 0:
        return np.zeros(0, dtype=np.float32)
    return np.stack([ch[:min_len] for ch in channels], axis=1)


def _boundary_pause_ms(
    boundary_kind: str | None,
    *,
    comma_pause_ms: int,
    period_pause_ms: int,
    semicolon_pause_ms: int,
    colon_pause_ms: int,
    dash_pause_ms: int,
    newline_pause_ms: int,
) -> int:
    if boundary_kind == "newline":
        return int(newline_pause_ms)
    if boundary_kind == "terminator":
        return int(period_pause_ms)
    if boundary_kind == ";":
        return int(semicolon_pause_ms)
    if boundary_kind == ":":
        return int(colon_pause_ms)
    if boundary_kind in ("—", "-"):
        return int(dash_pause_ms)
    if boundary_kind == ",":
        return int(comma_pause_ms)
    return 0


def _split_text_with_punctuation(text: str, pause_settings: dict) -> list[tuple[str, int, str | None]]:
    pause_map = {
        ",": int(pause_settings.get("comma_pause_ms", 0)),
        ".": int(pause_settings.get("period_pause_ms", 0)),
        "!": int(pause_settings.get("period_pause_ms", 0)),
        "?": int(pause_settings.get("period_pause_ms", 0)),
        ";": int(pause_settings.get("semicolon_pause_ms", 0)),
        ":": int(pause_settings.get("colon_pause_ms", 0)),
        "—": int(pause_settings.get("dash_pause_ms", 0)),
        "-": int(pause_settings.get("dash_pause_ms", 0)),
        "\n": int(pause_settings.get("newline_pause_ms", 0)),
    }
    units: list[tuple[str, int, str | None]] = []
    buffer: list[str] = []
    for ch in text:
        if ch in _PUNCT_TOKENS:
            if ch != "\n":
                buffer.append(ch)
            sub_text = "".join(buffer).strip()
            if sub_text:
                units.append((sub_text, max(pause_map.get(ch, 0), 0), ch))
            buffer = []
            continue
        buffer.append(ch)
    tail = "".join(buffer).strip()
    if tail:
        units.append((tail, 0, None))
    return units


def build_pause_plan(chunks: list[ChunkInfo], pause_settings: dict) -> list[dict]:
    plan = []
    for idx, chunk in enumerate(chunks):
        if idx >= len(chunks) - 1:
            break
        pause_ms = _boundary_pause_ms(
            chunk.boundary_kind,
            comma_pause_ms=int(pause_settings.get("comma_pause_ms", 0)),
            period_pause_ms=int(pause_settings.get("period_pause_ms", 0)),
            semicolon_pause_ms=int(pause_settings.get("semicolon_pause_ms", 0)),
            colon_pause_ms=int(pause_settings.get("colon_pause_ms", 0)),
            dash_pause_ms=int(pause_settings.get("dash_pause_ms", 0)),
            newline_pause_ms=int(pause_settings.get("newline_pause_ms", 0)),
        )
        plan.append({"pause_ms": pause_ms, "reason": chunk.boundary_kind})
    return plan


def _coerce_audio_result(result, default_sr: int | None = None):
    if isinstance(result, tuple):
        if len(result) >= 2:
            return result[0], int(result[1])
    if isinstance(result, dict) and "audio" in result:
        sr = result.get("sr", default_sr)
        return result["audio"], int(sr) if sr is not None else None
    raise TypeError(f"Unsupported audio result: {type(result)}")


def run_tts_pipeline(request: dict) -> PipelineResult:
    backend_id = request.get("tts_backend")
    backend = get_backend(backend_id)
    if backend is None:
        raise BackendUnavailableError(f"Backend introuvable: {backend_id}")
    if not backend.is_available():
        reason = backend.unavailable_reason() or "Dépendances manquantes."
        raise BackendUnavailableError(f"Backend indisponible: {backend_id}. {reason}")

    script = request.get("script") or ""
    if not script.strip():
        raise ValueError("Le texte est vide.")

    chunks = request.get("chunks")
    if chunks is None:
        chunks = []
    if chunks and not isinstance(chunks[0], ChunkInfo):
        raise ValueError("chunks must be ChunkInfo list")

    if not chunks:
        chunks = []
        chunk_settings = request.get("chunk_settings") or {}
        chunks = list(
            chunk_script(
                script,
                min_words_per_chunk=int(
                    chunk_settings.get("min_words_per_chunk", DEFAULT_MIN_WORDS_PER_CHUNK)
                ),
                max_words_without_terminator=int(
                    chunk_settings.get(
                        "max_words_without_terminator", DEFAULT_MAX_WORDS_WITHOUT_TERMINATOR
                    )
                ),
                max_est_seconds_per_chunk=float(
                    chunk_settings.get(
                        "max_est_seconds_per_chunk", DEFAULT_MAX_EST_SECONDS_PER_CHUNK
                    )
                ),
            )
        )
    if not chunks:
        raise ValueError("Aucun chunk généré.")

    pause_settings = request.get("pause_settings") or {}
    comma_pause_ms = int(pause_settings.get("comma_pause_ms", 0))
    period_pause_ms = int(pause_settings.get("period_pause_ms", 0))
    semicolon_pause_ms = int(pause_settings.get("semicolon_pause_ms", 0))
    colon_pause_ms = int(pause_settings.get("colon_pause_ms", 0))
    dash_pause_ms = int(pause_settings.get("dash_pause_ms", 0))
    newline_pause_ms = int(pause_settings.get("newline_pause_ms", 0))

    post_settings = request.get("post_settings") or {}
    zero_cross_radius_ms = int(post_settings.get("zero_cross_radius_ms", 10))
    fade_ms = int(post_settings.get("fade_ms", 50))
    silence_threshold = float(post_settings.get("silence_threshold", 0.002))
    silence_min_ms = int(post_settings.get("silence_min_ms", 20))

    target_sr = int(request.get("target_sr") or TARGET_SR)
    engine_params = request.get("engine_params") or {}
    lang = request.get("lang_code") or request.get("lang")
    voice_ref_path = request.get("voice_ref_path")
    out_path = request.get("out_path")
    if not out_path:
        raise ValueError("out_path must be provided")

    durations: list[float] = []
    retries: list[bool] = []
    boundary_kinds: list[str | None] = []
    boundary_pauses: list[int] = []
    punct_fixes: list[str | None] = []
    pause_events_by_chunk: list[list[dict]] = []

    audio_chunks: list[np.ndarray] = []
    pending_fade_in = False
    total_pause_ms = 0
    pause_plan = request.get("pause_plan") or []
    join_count = 0
    segments_count_total = 0
    punct_tokens: list[tuple[str, int]] = []
    backend_meta_last: dict[str, Any] = {}
    backend_logs: list[str] = []

    for idx, chunk_info in enumerate(chunks, start=1):
        chunk_segments = list(chunk_info.segments)
        ensure_strong_ending(chunk_segments)
        clean_text = render_clean_text_from_segments(chunk_segments)
        fix_note = None
        injected_hard = False
        fixed_text, fix_note = stabilize_trailing_punct(clean_text)
        clean_text = fixed_text
        if fix_note:
            injected_hard = True
        boundary_kind = chunk_info.boundary_kind
        if boundary_kind in (None, "hard") and not clean_text.rstrip().endswith(
            (".", "!", "?", "…")
        ):
            clean_text = clean_text.rstrip() + "."
            injected_hard = True
            fix_note = "hard '.' injected"
        punct_fixes.append(fix_note)

        clean_text = strip_legacy_tokens(clean_text)
        subunits = _split_text_with_punctuation(clean_text, pause_settings)
        segments_count_total += len(subunits)
        pause_events_by_chunk.append([])
        last_subunit_pause = 0
        for sub_text, pause_after_ms, token in subunits:
            if token and pause_after_ms > 0:
                punct_tokens.append((token, pause_after_ms))
            result = backend.synthesize_chunk(
                sub_text,
                voice_ref_path=voice_ref_path,
                lang=lang,
                **engine_params,
            )
            meta = {}
            if isinstance(result, tuple) and len(result) >= 3 and isinstance(result[2], dict):
                meta = result[2]
            elif isinstance(result, dict):
                meta = result.get("meta") if isinstance(result.get("meta"), dict) else {}
            if meta:
                backend_meta_last = dict(meta)
                stdout = meta.get("stdout")
                stderr = meta.get("stderr")
                if stdout:
                    backend_logs.append(f"stdout: {stdout}")
                if stderr:
                    backend_logs.append(f"stderr: {stderr}")
            audio, sr = _coerce_audio_result(result, default_sr=target_sr)
            if sr is None:
                sr = target_sr
            audio = np.asarray(audio, dtype=np.float32)
            if sr != target_sr:
                audio = _resample_audio(audio, sr, target_sr)
            duration = len(audio) / float(target_sr) if target_sr else 0.0
            durations.append(duration)
            retries.append(bool(meta.get("retry")))

            if pending_fade_in:
                audio = _start_at_zero_and_fade_in(
                    audio,
                    target_sr,
                    radius_ms=zero_cross_radius_ms,
                    fade_ms=fade_ms,
                )
                pending_fade_in = False
            if audio_chunks:
                audio_chunks[-1] = _trim_to_zero_and_fade_out(
                    audio_chunks[-1],
                    target_sr,
                    radius_ms=zero_cross_radius_ms,
                    fade_ms=fade_ms,
                )
                join_count += 1
            audio_chunks.append(audio)
            if pause_after_ms > 0:
                frames = int(target_sr * (pause_after_ms / 1000.0))
                audio_chunks.append(np.zeros(frames, dtype=np.float32))
                total_pause_ms += pause_after_ms
                pending_fade_in = True
            last_subunit_pause = pause_after_ms

        final_boundary_kind = boundary_kind
        if injected_hard:
            final_boundary_kind = "terminator"
        boundary_kinds.append(final_boundary_kind)
        if last_subunit_pause > 0:
            boundary_pause = 0
        elif idx <= len(pause_plan):
            plan = pause_plan[idx - 1] if idx - 1 < len(pause_plan) else None
            if isinstance(plan, dict) and "pause_ms" in plan:
                boundary_pause = int(plan.get("pause_ms") or 0)
            else:
                boundary_pause = 0
        else:
            boundary_pause = _boundary_pause_ms(
                final_boundary_kind,
                comma_pause_ms=comma_pause_ms,
                period_pause_ms=period_pause_ms,
                semicolon_pause_ms=semicolon_pause_ms,
                colon_pause_ms=colon_pause_ms,
                dash_pause_ms=dash_pause_ms,
                newline_pause_ms=newline_pause_ms,
            )
        boundary_pauses.append(boundary_pause)
        if idx < len(chunks):
            if audio_chunks:
                audio_chunks[-1] = _trim_to_zero_and_fade_out(
                    audio_chunks[-1],
                    target_sr,
                    radius_ms=zero_cross_radius_ms,
                    fade_ms=fade_ms,
                )
                join_count += 1
            if boundary_pause > 0:
                frames = int(target_sr * (boundary_pause / 1000.0))
                audio_chunks.append(np.zeros(frames, dtype=np.float32))
                total_pause_ms += boundary_pause
            pending_fade_in = True

    final_audio = np.concatenate(audio_chunks) if audio_chunks else np.zeros(0, dtype=np.float32)
    final_audio = _apply_post_processing(
        final_audio,
        target_sr,
        zero_cross_radius_ms,
        fade_ms,
        silence_threshold,
        silence_min_ms,
    )

    out_path = str(Path(out_path).expanduser().resolve())
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(out_path, final_audio, target_sr)

    meta = {
        "backend_id": backend_id,
        "backend_lang": lang,
        "chunks": len(chunks),
        "durations": durations,
        "retries": retries,
        "boundary_kinds": boundary_kinds,
        "boundary_pauses": boundary_pauses,
        "punct_fixes": punct_fixes,
        "pause_events": pause_events_by_chunk,
        "pause_plan": pause_plan,
        "total_duration": len(final_audio) / float(target_sr) if target_sr else 0.0,
        "total_pause_ms": total_pause_ms,
        "sr": target_sr,
        "segments_count_total": segments_count_total,
        "join_count": join_count,
        "punct_tokens": punct_tokens,
        "backend_meta": backend_meta_last,
        "backend_logs": backend_logs,
        "warnings": [],
    }
    return PipelineResult(out_path=out_path, meta=meta)
