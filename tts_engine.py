"""Wrapper around Chatterbox TTS with Thomcles fine-tune replacement."""

from __future__ import annotations

import inspect
import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import soundfile as sf
import torch
try:
    from chatterbox.tts import ChatterboxTTS
except ModuleNotFoundError:  # pragma: no cover - optional dependency in tests
    ChatterboxTTS = None
try:
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS
except ImportError:  # pragma: no cover - optional backend depending on package version
    ChatterboxMultilingualTTS = None
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from text_tools import (
    ChunkInfo,
    DEFAULT_COMMA_PAUSE_MS,
    DEFAULT_COLON_PAUSE_MS,
    DEFAULT_DASH_PAUSE_MS,
    DEFAULT_MAX_COMMA_SUBSEGMENTS,
    DEFAULT_MAX_EST_SECONDS_PER_CHUNK,
    DEFAULT_MIN_WORDS_PER_CHUNK,
    DEFAULT_MAX_WORDS_WITHOUT_TERMINATOR,
    DEFAULT_NEWLINE_PAUSE_MS,
    DEFAULT_PERIOD_PAUSE_MS,
    DEFAULT_SEMICOLON_PAUSE_MS,
    SpeechSegment,
    chunk_script,
    ensure_strong_ending,
    render_clean_text_from_segments,
    split_text_and_pauses,
    stabilize_trailing_punct,
    strip_legacy_tokens,
)
from logging_utils import is_verbose, verbosity_context

LOGGER = logging.getLogger("chatterbox_tts")

BASE_REPO = "ResembleAI/chatterbox"
FR_REPO = "Thomcles/Chatterbox-TTS-French"
LANGUAGE_MAP = {
    "fr-FR": "fr",
    "en-US": "en",
    "en-GB": "en",
    "es-ES": "es",
    "de-DE": "de",
    "it-IT": "it",
    "pt-PT": "pt",
    "nl-NL": "nl",
}
ZERO_CROSS_RADIUS_MS = 10
FADE_MS = 50
SILENCE_THRESHOLD = 0.002
SILENCE_MIN_MS = 20


def _find_zero_crossing_near(audio: np.ndarray, center_idx: int, radius: int) -> int:
    if audio.size == 0:
        return 0
    center_idx = max(0, min(int(center_idx), len(audio) - 1))
    radius = max(0, int(radius))
    start = max(0, center_idx - radius)
    end = min(len(audio) - 1, center_idx + radius)
    window = audio[start : end + 1]
    if window.size == 0:
        return center_idx
    offset = int(np.argmin(np.abs(window)))
    return start + offset


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


def _trim_to_zero_and_fade_out(
    audio: np.ndarray,
    sr: int,
    radius_ms: int = ZERO_CROSS_RADIUS_MS,
    fade_ms: int = FADE_MS,
) -> np.ndarray:
    if audio.size == 0:
        return audio
    radius_frames = int(sr * (radius_ms / 1000.0))
    fade_frames = int(sr * (fade_ms / 1000.0))
    end_idx = _find_zero_crossing_near(audio, len(audio) - 1, radius_frames)
    trimmed = audio[: end_idx + 1].copy()
    return _fade_out(trimmed, fade_frames)


def _start_at_zero_and_fade_in(
    audio: np.ndarray,
    sr: int,
    radius_ms: int = ZERO_CROSS_RADIUS_MS,
    fade_ms: int = FADE_MS,
) -> np.ndarray:
    if audio.size == 0:
        return audio
    radius_frames = int(sr * (radius_ms / 1000.0))
    fade_frames = int(sr * (fade_ms / 1000.0))
    start_idx = _find_zero_crossing_near(audio, 0, radius_frames)
    trimmed = audio[start_idx:].copy()
    return _fade_in(trimmed, fade_frames)


def _apply_silence_edge_fades(
    audio: np.ndarray,
    sr: int,
    silence_threshold: float,
    silence_min_ms: int,
    fade_ms: int,
) -> np.ndarray:
    if audio.size == 0:
        return audio
    fade_frames = int(sr * (fade_ms / 1000.0))
    min_silence_frames = int(sr * (silence_min_ms / 1000.0))
    if fade_frames <= 0 or min_silence_frames <= 0:
        return audio
    silence_mask = np.abs(audio) <= float(silence_threshold)
    if not np.any(silence_mask):
        return audio
    audio = audio.copy()
    in_silence = False
    start_idx = 0
    for idx, is_silent in enumerate(silence_mask):
        if is_silent and not in_silence:
            in_silence = True
            start_idx = idx
        elif not is_silent and in_silence:
            end_idx = idx - 1
            if end_idx - start_idx + 1 >= min_silence_frames:
                fade_out_start = max(0, start_idx - fade_frames)
                fade_out_end = start_idx
                fade_in_start = end_idx + 1
                fade_in_end = min(len(audio), fade_in_start + fade_frames)
                if fade_out_end > fade_out_start:
                    audio[fade_out_start:fade_out_end] = _fade_out(
                        audio[fade_out_start:fade_out_end],
                        fade_out_end - fade_out_start,
                    )
                if fade_in_end > fade_in_start:
                    audio[fade_in_start:fade_in_end] = _fade_in(
                        audio[fade_in_start:fade_in_end],
                        fade_in_end - fade_in_start,
                    )
            in_silence = False
    if in_silence:
        end_idx = len(audio) - 1
        if end_idx - start_idx + 1 >= min_silence_frames:
            fade_out_start = max(0, start_idx - fade_frames)
            fade_out_end = start_idx
            if fade_out_end > fade_out_start:
                audio[fade_out_start:fade_out_end] = _fade_out(
                    audio[fade_out_start:fade_out_end],
                    fade_out_end - fade_out_start,
                )
    return audio


def _ensure_repo_override() -> None:
    """Override chatterbox.tts.REPO_ID before calling from_pretrained."""

    import chatterbox.tts as tts_mod  # local import to avoid import-time side effects

    if getattr(tts_mod, "REPO_ID", None) != BASE_REPO:
        tts_mod.REPO_ID = BASE_REPO


class TTSEngine:
    """Load and run Chatterbox TTS with Thomcles fine-tune on Apple Silicon."""

    def __init__(self, device: Optional[str] = None) -> None:
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self._tts_fr_finetune: Optional[ChatterboxTTS] = None
        self._tts_multilang: Optional[ChatterboxTTS] = None
        self.sample_rate: Optional[int] = None
        self._lang_support: dict[int, bool] = {}

    def _load_fr_backend(self) -> ChatterboxTTS:
        if ChatterboxTTS is None:  # pragma: no cover - handled in tests without chatterbox
            raise RuntimeError("ChatterboxTTS backend is unavailable.")
        _ensure_repo_override()
        LOGGER.info("Loading base Chatterbox model on %s", self.device)
        tts = ChatterboxTTS.from_pretrained(self.device)
        LOGGER.info("Loading Thomcles fine-tune weights for T3")
        fr_t3_path = hf_hub_download(repo_id=FR_REPO, filename="t3_cfg.safetensors")
        fr_t3_state = load_file(fr_t3_path)
        if "model" in fr_t3_state:
            fr_t3_state = fr_t3_state["model"][0]
        tts.t3.load_state_dict(fr_t3_state)
        tts.t3.to(self.device).eval()
        self.sample_rate = tts.sr
        return tts

    def _load_multilang_backend(self) -> ChatterboxTTS:
        if ChatterboxMultilingualTTS is None:
            raise RuntimeError("ChatterboxMultilingualTTS backend is unavailable.")
        _ensure_repo_override()
        LOGGER.info("Loading base Chatterbox model on %s", self.device)
        try:
            tts = ChatterboxMultilingualTTS.from_pretrained(self.device)
        except RuntimeError as exc:
            msg = str(exc)
            if "deserialize object on a CUDA device" not in msg and "torch.cuda.is_available() is False" not in msg:
                raise
            LOGGER.warning("multilang_cuda_checkpoint_detected -> forcing map_location=cpu")
            orig_load = torch.load

            def _safe_load(*args, **kwargs):
                if "map_location" not in kwargs:
                    kwargs["map_location"] = torch.device("cpu")
                return orig_load(*args, **kwargs)

            torch.load = _safe_load
            try:
                try:
                    tts = ChatterboxMultilingualTTS.from_pretrained(self.device)
                except Exception:
                    tts = ChatterboxMultilingualTTS.from_pretrained("cpu")
            finally:
                torch.load = orig_load
        if hasattr(tts, "to"):
            try:
                tts.to(self.device)
            except Exception:
                pass
        LOGGER.info("tts_backend_class=%s", tts.__class__.__name__)
        LOGGER.info("multilang_loaded_device=%s", self.device)
        self.sample_rate = tts.sr
        return tts

    def _get_backend(self, mode: str) -> ChatterboxTTS:
        if mode == "fr_finetune":
            if self._tts_fr_finetune is None:
                self._tts_fr_finetune = self._load_fr_backend()
            return self._tts_fr_finetune
        if mode == "multilang":
            if self._tts_multilang is None:
                self._tts_multilang = self._load_multilang_backend()
            return self._tts_multilang
        raise ValueError(f"Unsupported tts_model_mode: {mode}")

    def _supports_language(self, tts: ChatterboxTTS) -> bool:
        key = id(tts)
        cached = self._lang_support.get(key)
        if cached is not None:
            return cached
        try:
            sig = inspect.signature(tts.generate)
            supports = "language" in sig.parameters
        except (TypeError, ValueError):
            supports = False
        self._lang_support[key] = supports
        return supports

    def _supports_language_id(self, tts: ChatterboxTTS) -> bool:
        key = id(tts) + 1
        cached = self._lang_support.get(key)
        if cached is not None:
            return cached
        try:
            sig = inspect.signature(tts.generate)
            supports = "language_id" in sig.parameters
        except (TypeError, ValueError):
            supports = False
        self._lang_support[key] = supports
        return supports

    def _resolve_language(self, mode: str, language: Optional[str]) -> Optional[str]:
        if mode == "fr_finetune":
            return "fr-FR"
        return language or "fr-FR"

    def _map_multilang_language(self, language: str | None) -> str:
        return LANGUAGE_MAP.get(language or "fr-FR", "fr")

    def _synthesize_text(
        self,
        tts: ChatterboxTTS,
        text: str,
        audio_prompt_path: Optional[str],
        exaggeration: float,
        cfg_weight: float,
        temperature: float,
        repetition_penalty: float,
        language: Optional[str],
        language_kw_preference: str | None,
        log_language_kw: bool,
    ) -> tuple[np.ndarray, str | None]:
        kwargs = {
            "audio_prompt_path": audio_prompt_path,
            "exaggeration": float(exaggeration),
            "cfg_weight": float(cfg_weight),
            "temperature": float(temperature),
            "repetition_penalty": float(repetition_penalty),
        }
        lang_kw_used = None
        if language:
            if language_kw_preference == "language_id":
                kwargs["language_id"] = language
                lang_kw_used = "language_id"
            elif language_kw_preference == "language":
                kwargs["language"] = language
                lang_kw_used = "language"
            else:
                if self._supports_language_id(tts):
                    kwargs["language_id"] = language
                    lang_kw_used = "language_id"
                elif self._supports_language(tts):
                    kwargs["language"] = language
                    lang_kw_used = "language"
        with verbosity_context(verbose=is_verbose()):
            try:
                wav = tts.generate(text, **kwargs)
            except TypeError as exc:
                if "language_id" in kwargs and language:
                    kwargs.pop("language_id", None)
                    if self._supports_language(tts) or language_kw_preference == "language":
                        kwargs["language"] = language
                        lang_kw_used = "language"
                        wav = tts.generate(text, **kwargs)
                    else:
                        raise exc
                else:
                    raise exc
        return wav.squeeze(0).detach().cpu().numpy().astype(np.float32), lang_kw_used

    def _build_audio_from_text(
        self,
        tts: ChatterboxTTS,
        text: str,
        audio_prompt_path: Optional[str],
        exaggeration: float,
        cfg_weight: float,
        temperature: float,
        repetition_penalty: float,
        language: Optional[str],
        language_kw_preference: str | None,
        log_language_kw: bool,
        comma_pause_ms: int,
        period_pause_ms: int,
        semicolon_pause_ms: int,
        colon_pause_ms: int,
        dash_pause_ms: int,
        newline_pause_ms: int,
        max_subsegments: int,
        suppress_final_pause: bool,
        zero_cross_radius_ms: int,
        fade_ms: int,
    ) -> tuple[np.ndarray, List[dict]]:
        sr = self.sample_rate or (tts.sr if tts else 24000)
        cleaned = strip_legacy_tokens(text)
        cleaned = re.sub(r"([,;:!?])\s*(?=\S)", r"\1 ", cleaned)
        segments, pause_events = split_text_and_pauses(
            cleaned,
            comma_pause_ms=int(comma_pause_ms),
            period_pause_ms=int(period_pause_ms),
            semicolon_pause_ms=int(semicolon_pause_ms),
            colon_pause_ms=int(colon_pause_ms),
            dash_pause_ms=int(dash_pause_ms),
            newline_pause_ms=int(newline_pause_ms),
            suppress_final_pause=bool(suppress_final_pause),
            return_events=True,
        )
        audio_chunks: List[np.ndarray] = []
        segments = _prepare_segments_for_synthesis(segments)
        if max_subsegments > 0 and len(segments) > max_subsegments:
            segments = [seg for seg in segments if seg.kind == "text"]
            pause_events = []
        logged_kw = False
        pending_fade_in = False
        for segment in segments:
            if segment.kind == "silence":
                if audio_chunks:
                    audio_chunks[-1] = _trim_to_zero_and_fade_out(
                        audio_chunks[-1],
                        sr,
                        radius_ms=zero_cross_radius_ms,
                        fade_ms=fade_ms,
                    )
                frames = int(sr * (segment.duration_ms / 1000.0))
                if frames > 0:
                    audio_chunks.append(np.zeros(frames, dtype=np.float32))
                    pending_fade_in = True
                continue
            chunk_text = segment.content.strip()
            if not chunk_text:
                continue
            should_log = log_language_kw and not logged_kw
            audio_chunk, lang_kw_used = self._synthesize_text(
                    tts,
                    chunk_text,
                    audio_prompt_path,
                    exaggeration,
                    cfg_weight,
                    temperature,
                    repetition_penalty,
                    language,
                    language_kw_preference,
                    should_log,
                )
            if pending_fade_in:
                audio_chunk = _start_at_zero_and_fade_in(
                    audio_chunk,
                    sr,
                    radius_ms=zero_cross_radius_ms,
                    fade_ms=fade_ms,
                )
                pending_fade_in = False
            audio_chunks.append(audio_chunk)
            if should_log and lang_kw_used:
                LOGGER.info("multilang_lang_kw=%s", lang_kw_used)
            if should_log:
                logged_kw = True
        pause_log = [{"symbol": evt.symbol, "duration_ms": evt.duration_ms} for evt in pause_events]
        if not audio_chunks:
            return np.zeros(0, dtype=np.float32), pause_log
        return np.concatenate(audio_chunks), pause_log

    def generate(
        self,
        text: str,
        audio_prompt_path: Optional[str],
        exaggeration: float = 0.5,
        cfg_weight: float = 0.6,
        temperature: float = 0.5,
        repetition_penalty: float = 1.35,
        tts_model_mode: str = "fr_finetune",
        tts_language: str = "fr-FR",
        multilang_cfg_weight: float = 0.5,
        comma_pause_ms: int = DEFAULT_COMMA_PAUSE_MS,
        period_pause_ms: int = DEFAULT_PERIOD_PAUSE_MS,
        semicolon_pause_ms: int = DEFAULT_SEMICOLON_PAUSE_MS,
        colon_pause_ms: int = DEFAULT_COLON_PAUSE_MS,
        dash_pause_ms: int = DEFAULT_DASH_PAUSE_MS,
        newline_pause_ms: int = DEFAULT_NEWLINE_PAUSE_MS,
        max_comma_subsegments: int = DEFAULT_MAX_COMMA_SUBSEGMENTS,
        zero_cross_radius_ms: int = ZERO_CROSS_RADIUS_MS,
        fade_ms: int = FADE_MS,
        silence_threshold: float = SILENCE_THRESHOLD,
        silence_min_ms: int = SILENCE_MIN_MS,
        out_path: Optional[str] = None,
    ) -> tuple[str, int]:
        if not text.strip():
            raise ValueError("Le texte est vide.")

        tts = self._get_backend(tts_model_mode)
        requested_language = self._resolve_language(tts_model_mode, tts_language)
        backend_language = requested_language
        effective_cfg = cfg_weight
        LOGGER.info("voice_mode=%s", tts_model_mode)
        if tts_model_mode == "multilang":
            backend_language = self._map_multilang_language(requested_language)
            effective_cfg = multilang_cfg_weight
            LOGGER.info("requested_language_bcp47=%s", requested_language)
            LOGGER.info("backend_language_id=%s", backend_language)
        LOGGER.info("cfg_weight=%s", float(effective_cfg))
        LOGGER.info("Generating TTS using prompt %s", audio_prompt_path)
        LOGGER.info("tts_backend_class=%s", tts.__class__.__name__)
        if tts_model_mode == "multilang":
            LOGGER.info("multilang_call_begin backend_language_id=%s", backend_language)
        audio, _ = self._build_audio_from_text(
            tts,
            text,
            audio_prompt_path,
            exaggeration,
            effective_cfg,
            temperature,
            repetition_penalty,
            backend_language,
            "language_id" if tts_model_mode == "multilang" else None,
            tts_model_mode == "multilang",
            int(comma_pause_ms),
            int(period_pause_ms),
            int(semicolon_pause_ms),
            int(colon_pause_ms),
            int(dash_pause_ms),
            int(newline_pause_ms),
            int(max_comma_subsegments),
            False,
            int(zero_cross_radius_ms),
            int(fade_ms),
        )
        audio = _apply_silence_edge_fades(
            audio,
            self.sample_rate or tts.sr,
            float(silence_threshold),
            int(silence_min_ms),
            int(fade_ms),
        )

        if out_path is None:
            raise ValueError("out_path must be provided")

        out_path = str(Path(out_path).expanduser().resolve())
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        sr = self.sample_rate or tts.sr
        sf.write(out_path, audio, sr)
        LOGGER.info("Saved audio to %s", out_path)
        return out_path, sr

    def generate_longform(
        self,
        script: str,
        audio_prompt_path: Optional[str],
        *,
        chunks: Optional[List[ChunkInfo]] = None,
        out_path: str,
        exaggeration: float = 0.5,
        cfg_weight: float = 0.6,
        temperature: float = 0.5,
        repetition_penalty: float = 1.35,
        tts_model_mode: str = "fr_finetune",
        tts_language: str = "fr-FR",
        multilang_cfg_weight: float = 0.5,
        comma_pause_ms: int = DEFAULT_COMMA_PAUSE_MS,
        period_pause_ms: int = DEFAULT_PERIOD_PAUSE_MS,
        semicolon_pause_ms: int = DEFAULT_SEMICOLON_PAUSE_MS,
        colon_pause_ms: int = DEFAULT_COLON_PAUSE_MS,
        dash_pause_ms: int = DEFAULT_DASH_PAUSE_MS,
        newline_pause_ms: int = DEFAULT_NEWLINE_PAUSE_MS,
        min_words_per_chunk: int = DEFAULT_MIN_WORDS_PER_CHUNK,
        max_words_without_terminator: int = DEFAULT_MAX_WORDS_WITHOUT_TERMINATOR,
        max_est_seconds_per_chunk: float = DEFAULT_MAX_EST_SECONDS_PER_CHUNK,
        max_comma_subsegments: int = DEFAULT_MAX_COMMA_SUBSEGMENTS,
        stabilize_punctuation: bool = True,
        zero_cross_radius_ms: int = ZERO_CROSS_RADIUS_MS,
        fade_ms: int = FADE_MS,
        silence_threshold: float = SILENCE_THRESHOLD,
        silence_min_ms: int = SILENCE_MIN_MS,
    ) -> tuple[str, int, Dict]:
        if not script.strip():
            raise ValueError("Le texte est vide.")

        tts = self._get_backend(tts_model_mode)
        requested_language = self._resolve_language(tts_model_mode, tts_language)
        backend_language = requested_language
        effective_cfg = cfg_weight
        LOGGER.info("voice_mode=%s", tts_model_mode)
        if tts_model_mode == "multilang":
            backend_language = self._map_multilang_language(requested_language)
            effective_cfg = multilang_cfg_weight
            LOGGER.info("requested_language_bcp47=%s", requested_language)
            LOGGER.info("backend_language_id=%s", backend_language)
        LOGGER.info("cfg_weight=%s", float(effective_cfg))
        if chunks is None:
            chunks = chunk_script(
                script,
                min_words_per_chunk=int(min_words_per_chunk),
                max_words_without_terminator=int(max_words_without_terminator),
                max_est_seconds_per_chunk=float(max_est_seconds_per_chunk),
            )
        if not chunks:
            raise ValueError("Aucun chunk généré.")

        LOGGER.info("tts_backend_class=%s", tts.__class__.__name__)
        if tts_model_mode == "multilang":
            LOGGER.info("multilang_call_begin backend_language_id=%s", backend_language)

        sr = self.sample_rate or tts.sr
        audio_chunks: List[np.ndarray] = []
        durations: List[float] = []
        clean_texts: List[str] = []
        retries: List[bool] = []
        boundary_kinds: List[str | None] = []
        boundary_pauses: List[int] = []
        comma_counts: List[int] = []
        punct_fixes: List[str | None] = []
        pause_events_by_chunk: List[List[dict]] = []

        pending_fade_in = False
        for idx, chunk_info in enumerate(chunks, start=1):
            chunk_segments = list(chunk_info.segments)
            ensure_strong_ending(chunk_segments)
            clean_text = render_clean_text_from_segments(chunk_segments)
            fix_note = None
            injected_hard = False
            if stabilize_punctuation:
                fixed_text, fix_note = stabilize_trailing_punct(clean_text)
                clean_text = fixed_text
                if fix_note:
                    injected_hard = True
            boundary_kind = chunk_info.boundary_kind
            if stabilize_punctuation and boundary_kind in (None, "hard") and not clean_text.rstrip().endswith(
                (".", "!", "?", "…")
            ):
                clean_text = clean_text.rstrip() + "."
                injected_hard = True
                fix_note = "hard '.' injected"
            punct_fixes.append(fix_note)
            clean_texts.append(clean_text)
            comma_counts.append(clean_text.count(","))
            audio, pause_events = self._build_audio_from_text(
                tts,
                clean_text,
                audio_prompt_path,
                exaggeration,
                effective_cfg,
                temperature,
                repetition_penalty,
                backend_language,
                "language_id" if tts_model_mode == "multilang" else None,
                tts_model_mode == "multilang",
                int(comma_pause_ms),
                int(period_pause_ms),
                int(semicolon_pause_ms),
                int(colon_pause_ms),
                int(dash_pause_ms),
                int(newline_pause_ms),
                int(max_comma_subsegments),
                True,
                int(zero_cross_radius_ms),
                int(fade_ms),
            )
            pause_events_by_chunk.append(pause_events)
            duration = len(audio) / sr
            retried = False
            if len(clean_text) > 80 and duration < 1.2:
                retried = True
                LOGGER.info(
                    "early-EOS detected (chunk %s) -> retry with cfg+0.05 / temp-0.05",
                    idx,
                )
                base_duration = duration
                audio, pause_events = self._build_audio_from_text(
                    tts,
                    clean_text,
                    audio_prompt_path,
                    exaggeration,
                    min(1.5, cfg_weight + 0.05),
                    max(0.2, temperature - 0.05),
                    repetition_penalty,
                    backend_language,
                    "language_id" if tts_model_mode == "multilang" else None,
                    tts_model_mode == "multilang",
                    int(comma_pause_ms),
                    int(period_pause_ms),
                    int(semicolon_pause_ms),
                    int(colon_pause_ms),
                    int(dash_pause_ms),
                    int(newline_pause_ms),
                    int(max_comma_subsegments),
                    True,
                    int(zero_cross_radius_ms),
                    int(fade_ms),
                )
                pause_events_by_chunk[-1] = pause_events
                duration = len(audio) / sr
                if duration > base_duration:
                    LOGGER.info("retry success (chunk %s)", idx)
                else:
                    LOGGER.info("retry failed (kept first) (chunk %s)", idx)
            retries.append(retried)
            durations.append(duration)
            if pending_fade_in:
                audio = _start_at_zero_and_fade_in(
                    audio,
                    sr,
                    radius_ms=zero_cross_radius_ms,
                    fade_ms=fade_ms,
                )
                pending_fade_in = False
            audio_chunks.append(audio)
            boundary_kind = chunk_info.boundary_kind
            if stabilize_punctuation and injected_hard:
                boundary_kind = "terminator"
            boundary_pause = 0
            if boundary_kind == "newline":
                boundary_pause = int(newline_pause_ms)
            elif boundary_kind == "terminator":
                boundary_pause = int(period_pause_ms)
            elif boundary_kind == ";":
                boundary_pause = int(semicolon_pause_ms)
            elif boundary_kind == ":":
                boundary_pause = int(colon_pause_ms)
            elif boundary_kind in ("—", "-"):
                boundary_pause = int(dash_pause_ms)
            elif boundary_kind == ",":
                boundary_pause = int(comma_pause_ms)
            boundary_kinds.append(boundary_kind)
            boundary_pauses.append(boundary_pause)
            if idx < len(chunks):
                if audio_chunks:
                    audio_chunks[-1] = _trim_to_zero_and_fade_out(
                        audio_chunks[-1],
                        sr,
                        radius_ms=zero_cross_radius_ms,
                        fade_ms=fade_ms,
                    )
                if boundary_pause > 0:
                    frames = int(sr * (boundary_pause / 1000.0))
                    audio_chunks.append(np.zeros(frames, dtype=np.float32))
                pending_fade_in = True

        final_audio = np.concatenate(audio_chunks) if audio_chunks else np.zeros(0, dtype=np.float32)
        final_audio = _apply_silence_edge_fades(
            final_audio,
            sr,
            float(silence_threshold),
            int(silence_min_ms),
            int(fade_ms),
        )
        out_path = str(Path(out_path).expanduser().resolve())
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        sf.write(out_path, final_audio, sr)
        meta = {
            "chunks": len(chunks),
            "durations": durations,
            "clean_texts": clean_texts,
            "retries": retries,
            "boundary_kinds": boundary_kinds,
            "boundary_pauses": boundary_pauses,
            "comma_counts": comma_counts,
            "punct_fixes": punct_fixes,
            "pause_events": pause_events_by_chunk,
            "tts_model_mode": tts_model_mode,
            "requested_language_bcp47": requested_language,
            "backend_language_id": backend_language,
            "cfg_weight": float(effective_cfg),
            "total_duration": len(final_audio) / sr if sr else 0.0,
        }
        return out_path, sr, meta


def _is_too_short_text(text: str, min_useful_chars: int = 12) -> bool:
    if not text or not text.strip():
        return True
    useful = re.sub(r"[\W_]+", "", text, flags=re.UNICODE)
    if not useful:
        return True
    return len(useful) < min_useful_chars


def _starts_with_sticky_punct(text: str) -> bool:
    stripped = text.lstrip()
    if not stripped:
        return False
    return stripped[0] in {",", ";", ":", ".", "!", "?", "…", ")", "]", "»"}


def _useful_length(text: str) -> int:
    return len(re.sub(r"[\W_]+", "", text, flags=re.UNICODE))


def _has_alnum(text: str) -> bool:
    return any(ch.isalnum() for ch in text)


def _merge_text_chunks(left: str, right: str) -> str:
    if not left:
        return right
    if not right:
        return left
    right_stripped = right.lstrip()
    if _starts_with_sticky_punct(right):
        return f"{left}{right_stripped}"
    if left[-1].isalnum() and right_stripped and right_stripped[0].isalnum():
        return f"{left} {right}"
    return f"{left}{right}"


def _merge_short_text_block(texts: List[str]) -> List[str]:
    merged: List[str] = []
    pending = ""
    for text in texts:
        has_alnum = _has_alnum(text)
        too_short = _is_too_short_text(text)
        if not text.strip() or (not has_alnum and too_short):
            LOGGER.debug(
                "segment_action=SKIP len=%s useful=%s text=[%s]",
                len(text),
                _useful_length(text),
                text,
            )
            continue
        if too_short and has_alnum:
            if merged:
                merged[-1] = _merge_text_chunks(merged[-1], text)
                LOGGER.debug(
                    "segment_action=MERGE_PREV len=%s useful=%s text=[%s]",
                    len(text),
                    _useful_length(text),
                    text,
                )
            else:
                pending = _merge_text_chunks(pending, text)
                LOGGER.debug(
                    "segment_action=MERGE_NEXT len=%s useful=%s text=[%s]",
                    len(text),
                    _useful_length(text),
                    text,
                )
            continue
        if pending:
            text = _merge_text_chunks(pending, text)
            pending = ""
        merged.append(text)
        LOGGER.debug(
            "segment_action=KEEP len=%s useful=%s text=[%s]",
            len(text),
            _useful_length(text),
            text,
        )
    if pending:
        if _has_alnum(pending):
            if merged:
                merged[-1] = _merge_text_chunks(merged[-1], pending)
                LOGGER.debug(
                    "segment_action=MERGE_PREV len=%s useful=%s text=[%s]",
                    len(pending),
                    _useful_length(pending),
                    pending,
                )
            else:
                merged.append(pending)
                LOGGER.debug(
                    "segment_action=KEEP len=%s useful=%s text=[%s]",
                    len(pending),
                    _useful_length(pending),
                    pending,
                )
        else:
            LOGGER.debug(
                "segment_action=SKIP len=%s useful=%s text=[%s]",
                len(pending),
                _useful_length(pending),
                pending,
            )
    return merged


def _prepare_segments_for_synthesis(segments: List[SpeechSegment]) -> List[SpeechSegment]:
    prepared: List[SpeechSegment] = []
    block: List[str] = []

    def flush_block() -> None:
        if not block:
            return
        merged_texts = _merge_short_text_block(block)
        for text in merged_texts:
            if _is_too_short_text(text):
                if prepared and prepared[-1].kind == "text":
                    merged = _merge_text_chunks(prepared[-1].content, text)
                    prepared[-1] = SpeechSegment("text", merged)
                else:
                    LOGGER.debug("skip_short_segment")
                continue
            prepared.append(SpeechSegment("text", text))
        block.clear()

    for segment in segments:
        if segment.kind == "text":
            block.append(segment.content)
            continue
        flush_block()
        prepared.append(segment)
    flush_block()
    return prepared


__all__ = [
    "TTSEngine",
    "LANGUAGE_MAP",
    "ZERO_CROSS_RADIUS_MS",
    "FADE_MS",
    "SILENCE_THRESHOLD",
    "SILENCE_MIN_MS",
]
