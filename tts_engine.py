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
    DEFAULT_MAX_EST_SECONDS_PER_CHUNK,
    DEFAULT_MIN_WORDS_PER_CHUNK,
    DEFAULT_MAX_WORDS_WITHOUT_TERMINATOR,
    SpeechSegment,
    chunk_script,
    render_clean_text_from_segments,
    strip_legacy_tokens,
)
from logging_utils import is_verbose, verbosity_context
from audio_defaults import SILENCE_MIN_MS, SILENCE_THRESHOLD

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
    ) -> tuple[np.ndarray, List[dict]]:
        cleaned = strip_legacy_tokens(text)
        cleaned = re.sub(r"([,;:!?])\s*(?=\S)", r"\1 ", cleaned)
        if not cleaned.strip():
            return np.zeros(0, dtype=np.float32), []
        should_log = bool(log_language_kw)
        audio_chunk, lang_kw_used = self._synthesize_text(
            tts,
            cleaned,
            audio_prompt_path,
            exaggeration,
            cfg_weight,
            temperature,
            repetition_penalty,
            language,
            language_kw_preference,
            should_log,
        )
        if should_log and lang_kw_used:
            LOGGER.info("multilang_lang_kw=%s", lang_kw_used)
        return audio_chunk, []

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
        min_words_per_chunk: int = DEFAULT_MIN_WORDS_PER_CHUNK,
        max_words_without_terminator: int = DEFAULT_MAX_WORDS_WITHOUT_TERMINATOR,
        max_est_seconds_per_chunk: float = DEFAULT_MAX_EST_SECONDS_PER_CHUNK,
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
        comma_counts: List[int] = []
        for idx, chunk_info in enumerate(chunks, start=1):
            chunk_segments = list(chunk_info.segments)
            clean_text = render_clean_text_from_segments(chunk_segments)
            clean_texts.append(clean_text)
            comma_counts.append(clean_text.count(","))
            audio, _ = self._build_audio_from_text(
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
            )
            duration = len(audio) / sr
            retried = False
            if len(clean_text) > 80 and duration < 1.2:
                retried = True
                LOGGER.info(
                    "early-EOS detected (chunk %s) -> retry with cfg+0.05 / temp-0.05",
                    idx,
                )
                base_duration = duration
                audio_retry, _ = self._build_audio_from_text(
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
                )
                duration_retry = len(audio_retry) / sr
                if duration_retry > base_duration:
                    audio = audio_retry
                    duration = duration_retry
                    LOGGER.info("retry success (chunk %s)", idx)
                else:
                    LOGGER.info("retry failed (kept first) (chunk %s)", idx)
            retries.append(retried)
            durations.append(duration)
            audio_chunks.append(audio)
            boundary_kinds.append(chunk_info.boundary_kind)

        final_audio = np.concatenate(audio_chunks) if audio_chunks else np.zeros(0, dtype=np.float32)
        out_path = str(Path(out_path).expanduser().resolve())
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        sf.write(out_path, final_audio, sr)
        meta = {
            "chunks": len(chunks),
            "durations": durations,
            "clean_texts": clean_texts,
            "retries": retries,
            "boundary_kinds": boundary_kinds,
            "comma_counts": comma_counts,
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
