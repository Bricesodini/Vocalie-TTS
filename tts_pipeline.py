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
from text_tools import (
    DEFAULT_MAX_EST_SECONDS_PER_CHUNK,
    DEFAULT_MAX_WORDS_WITHOUT_TERMINATOR,
    DEFAULT_MIN_WORDS_PER_CHUNK,
    ChunkInfo,
    chunk_script,
    render_clean_text_from_segments,
    strip_legacy_tokens,
)


TARGET_SR = 24000


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


def _snap_zero_crossing(
    audio: np.ndarray,
    idx: int,
    *,
    radius_samples: int,
) -> int:
    if audio.size == 0:
        return idx
    idx = max(min(int(idx), audio.size - 1), 0)
    start = max(idx - radius_samples, 1)
    end = min(idx + radius_samples, audio.size - 1)
    best_idx = idx
    best_dist = radius_samples + 1
    for i in range(start, end + 1):
        prev_val = float(audio[i - 1])
        curr_val = float(audio[i])
        if prev_val == 0.0 or curr_val == 0.0 or (prev_val < 0.0 <= curr_val) or (prev_val > 0.0 >= curr_val):
            dist = abs(i - idx)
            if dist < best_dist:
                best_dist = dist
                best_idx = i
                if best_dist == 0:
                    break
    return best_idx


def _fade_in(audio: np.ndarray, fade_frames: int) -> np.ndarray:
    if audio.size == 0:
        return audio
    fade_frames = max(0, min(int(fade_frames), len(audio)))
    if fade_frames == 0:
        return audio
    ramp = np.linspace(0.0, 1.0, fade_frames, endpoint=True, dtype=np.float32)
    audio[:fade_frames] *= ramp
    return audio


def _fade_out(audio: np.ndarray, fade_frames: int) -> np.ndarray:
    if audio.size == 0:
        return audio
    fade_frames = max(0, min(int(fade_frames), len(audio)))
    if fade_frames == 0:
        return audio
    ramp = np.linspace(1.0, 0.0, fade_frames, endpoint=True, dtype=np.float32)
    audio[-fade_frames:] *= ramp
    return audio


def _apply_inter_chunk_gap(
    audio_chunks: list[np.ndarray],
    *,
    sr: int,
    gap_ms: int,
    fade_ms: int = 10,
) -> np.ndarray:
    if not audio_chunks:
        return np.zeros(0, dtype=np.float32)
    if gap_ms <= 0 or len(audio_chunks) == 1:
        return np.concatenate(audio_chunks)
    gap_frames = max(0, int(sr * (int(gap_ms) / 1000.0)))
    fade_frames = max(0, int(sr * (int(fade_ms) / 1000.0)))
    out: list[np.ndarray] = []
    last_idx = len(audio_chunks) - 1
    for idx, chunk in enumerate(audio_chunks):
        audio = np.asarray(chunk, dtype=np.float32)
        needs_fade = fade_frames > 0 and (idx < last_idx or idx > 0)
        if needs_fade:
            audio = audio.copy()
            if idx < last_idx:
                _fade_out(audio, fade_frames)
            if idx > 0:
                _fade_in(audio, fade_frames)
        out.append(audio)
        if idx < last_idx and gap_frames > 0:
            out.append(np.zeros(gap_frames, dtype=np.float32))
    return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)


def _find_active_range(
    mono: np.ndarray,
    *,
    threshold: float,
    min_silence_frames: int,
) -> tuple[int, int]:
    if mono.size == 0:
        return 0, 0
    mask = np.abs(mono) > float(threshold)
    if not np.any(mask):
        return 0, len(mono)
    start = int(np.argmax(mask))
    end = len(mono) - int(np.argmax(mask[::-1]))
    if start < min_silence_frames:
        start = 0
    if len(mono) - end < min_silence_frames:
        end = len(mono)
    return start, end


def minimal_post_process(
    raw_path: str | Path,
    processed_path: str | Path,
    *,
    zero_cross_radius_ms: int = 10,
    fade_ms: int = 10,
    silence_threshold: float = 0.002,
    silence_min_ms: int = 20,
    normalize_peak_db: float = -1.0,
) -> dict[str, Any]:
    raw_path = Path(raw_path)
    processed_path = Path(processed_path)
    if raw_path.resolve() == processed_path.resolve():
        raise ValueError("Le traitement doit écrire dans un fichier différent du raw.")
    audio, sr = sf.read(str(raw_path), always_2d=False)
    if not isinstance(sr, (int, float)):
        raise ValueError("Sample rate invalide pour traitement.")
    sr = int(sr)
    audio = np.asarray(audio, dtype=np.float32)
    mono = np.mean(audio, axis=1) if audio.ndim > 1 else audio
    min_silence_frames = int(sr * (int(silence_min_ms) / 1000.0))
    start_idx, end_idx = _find_active_range(
        mono,
        threshold=float(silence_threshold),
        min_silence_frames=min_silence_frames,
    )
    radius_samples = int(sr * (int(zero_cross_radius_ms) / 1000.0))
    if mono.size:
        start_idx = _snap_zero_crossing(mono, start_idx, radius_samples=radius_samples)
        end_idx = _snap_zero_crossing(mono, max(end_idx - 1, start_idx), radius_samples=radius_samples) + 1
    if end_idx <= start_idx:
        start_idx = 0
        end_idx = audio.shape[0]
    trimmed = audio[start_idx:end_idx].copy()
    fade_frames = int(sr * (int(fade_ms) / 1000.0))
    if trimmed.ndim == 1:
        trimmed = _fade_in(trimmed, fade_frames)
        trimmed = _fade_out(trimmed, fade_frames)
    else:
        for ch_idx in range(trimmed.shape[1]):
            trimmed[:, ch_idx] = _fade_in(trimmed[:, ch_idx], fade_frames)
            trimmed[:, ch_idx] = _fade_out(trimmed[:, ch_idx], fade_frames)
    peak_before = float(np.max(np.abs(trimmed))) if trimmed.size else 0.0
    scale = 1.0
    target_peak = float(10 ** (float(normalize_peak_db) / 20.0))
    if peak_before > 0.0 and target_peak > 0.0:
        scale = target_peak / peak_before
        trimmed = trimmed * scale
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(processed_path), trimmed, sr)
    return {
        "trim": {
            "start_sample": int(start_idx),
            "end_sample": int(end_idx),
        },
        "fade_ms": int(fade_ms),
        "zero_cross_radius_ms": int(zero_cross_radius_ms),
        "silence_threshold": float(silence_threshold),
        "silence_min_ms": int(silence_min_ms),
        "normalize_peak_db": float(normalize_peak_db),
        "normalize_scale": float(scale),
        "peak_before": float(peak_before),
    }


def generate_raw_wav(request: dict, progress_cb=None) -> PipelineResult:
    request = dict(request)
    return run_tts_pipeline(request, progress_cb=progress_cb)


def _coerce_audio_result(result, default_sr: int | None = None):
    if isinstance(result, tuple):
        if len(result) >= 2:
            return result[0], int(result[1])
    if isinstance(result, dict) and "audio" in result:
        sr = result.get("sr", default_sr)
        return result["audio"], int(sr) if sr is not None else None
    raise TypeError(f"Unsupported audio result: {type(result)}")


def run_tts_pipeline(request: dict, progress_cb=None) -> PipelineResult:
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

    target_sr = int(request.get("target_sr") or TARGET_SR)
    engine_params = request.get("engine_params") or {}
    lang = request.get("lang_code") or request.get("lang")
    voice_ref_path = request.get("voice_ref_path")
    out_path = request.get("out_path")
    if not out_path:
        raise ValueError("out_path must be provided")

    durations: list[float] = []
    retries: list[bool] = []
    audio_chunks: list[np.ndarray] = []
    segments_count_total = 0
    backend_meta_last: dict[str, Any] = {}
    backend_logs: list[str] = []

    if progress_cb:
        progress_cb(0.0)

    for idx, chunk_info in enumerate(chunks, start=1):
        chunk_segments = list(chunk_info.segments)
        clean_text = render_clean_text_from_segments(chunk_segments)
        clean_text = strip_legacy_tokens(clean_text)
        segments_count_total += 1
        result = backend.synthesize_chunk(
            clean_text,
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
        audio_chunks.append(audio)
        if progress_cb and chunks:
            progress_cb(idx / float(len(chunks)))

    inter_chunk_gap_ms = int(request.get("inter_chunk_gap_ms") or 0)
    if backend_id not in {"chatterbox", "qwen3"}:
        inter_chunk_gap_ms = 0
    gap_applied = bool(backend_id in {"chatterbox", "qwen3"} and len(audio_chunks) > 1 and inter_chunk_gap_ms > 0)
    if gap_applied:
        final_audio = _apply_inter_chunk_gap(audio_chunks, sr=target_sr, gap_ms=inter_chunk_gap_ms)
    else:
        final_audio = np.concatenate(audio_chunks) if audio_chunks else np.zeros(0, dtype=np.float32)

    out_path = str(Path(out_path).expanduser().resolve())
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(out_path, final_audio, target_sr)

    meta = {
        "backend_id": backend_id,
        "backend_lang": lang,
        "chunks": len(chunks),
        "durations": durations,
        "retries": retries,
        "total_duration": len(final_audio) / float(target_sr) if target_sr else 0.0,
        "duration_sec": len(final_audio) / float(target_sr) if target_sr else 0.0,
        "sr": target_sr,
        "segments_count_total": segments_count_total,
        "num_subunits": segments_count_total,
        "backend_meta": backend_meta_last,
        "backend_logs": backend_logs,
        "warnings": [],
        "inter_chunk_gap_ms": inter_chunk_gap_ms,
        "inter_chunk_gap_applied": gap_applied,
        "inter_chunk_gap_engine": backend_id,
        "inter_chunk_gap_chunks": len(chunks),
    }
    return PipelineResult(out_path=out_path, meta=meta)
