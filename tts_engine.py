"""Wrapper around Chatterbox TTS with Thomcles fine-tune replacement."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import soundfile as sf
import torch
from chatterbox.tts import ChatterboxTTS
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from text_tools import (
    SpeechSegment,
    chunk_script,
    ensure_strong_ending,
    render_clean_text_from_segments,
    split_text_and_pauses,
    stitch_segments,
)

LOGGER = logging.getLogger("chatterbox_tts")

BASE_REPO = "ResembleAI/chatterbox"
FR_REPO = "Thomcles/Chatterbox-TTS-French"


def _ensure_repo_override() -> None:
    """Override chatterbox.tts.REPO_ID before calling from_pretrained."""

    import chatterbox.tts as tts_mod  # local import to avoid import-time side effects

    if getattr(tts_mod, "REPO_ID", None) != BASE_REPO:
        tts_mod.REPO_ID = BASE_REPO


class TTSEngine:
    """Load and run Chatterbox TTS with Thomcles fine-tune on Apple Silicon."""

    def __init__(self, device: Optional[str] = None) -> None:
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.tts: Optional[ChatterboxTTS] = None
        self.sample_rate: Optional[int] = None
        self._load_model()

    def _load_model(self) -> None:
        _ensure_repo_override()
        LOGGER.info("Loading base Chatterbox model on %s", self.device)
        self.tts = ChatterboxTTS.from_pretrained(self.device)
        LOGGER.info("Loading Thomcles fine-tune weights for T3")
        fr_t3_path = hf_hub_download(repo_id=FR_REPO, filename="t3_cfg.safetensors")
        fr_t3_state = load_file(fr_t3_path)
        if "model" in fr_t3_state:
            fr_t3_state = fr_t3_state["model"][0]
        self.tts.t3.load_state_dict(fr_t3_state)
        self.tts.t3.to(self.device).eval()
        self.sample_rate = self.tts.sr

    def _synthesize_text(
        self,
        text: str,
        audio_prompt_path: Optional[str],
        exaggeration: float,
        cfg_weight: float,
        temperature: float,
        repetition_penalty: float,
    ) -> np.ndarray:
        assert self.tts is not None
        wav = self.tts.generate(
            text,
            audio_prompt_path=audio_prompt_path,
            exaggeration=float(exaggeration),
            cfg_weight=float(cfg_weight),
            temperature=float(temperature),
            repetition_penalty=float(repetition_penalty),
        )
        return wav.squeeze(0).detach().cpu().numpy().astype(np.float32)

    def _build_audio_from_segments(
        self,
        segments: List[SpeechSegment],
        audio_prompt_path: Optional[str],
        exaggeration: float,
        cfg_weight: float,
        temperature: float,
        repetition_penalty: float,
    ) -> np.ndarray:
        sr = self.sample_rate or (self.tts.sr if self.tts else 24000)

        def _synth(text: str) -> np.ndarray:
            return self._synthesize_text(
                text,
                audio_prompt_path,
                exaggeration,
                cfg_weight,
                temperature,
                repetition_penalty,
            )

        audio = stitch_segments(segments, sr, _synth)
        if audio.size == 0:
            raise ValueError("Aucun segment audio généré.")
        return audio

    def generate(
        self,
        text: str,
        audio_prompt_path: Optional[str],
        exaggeration: float = 0.5,
        cfg_weight: float = 0.6,
        temperature: float = 0.5,
        repetition_penalty: float = 1.35,
        out_path: Optional[str] = None,
    ) -> tuple[str, int]:
        if not text.strip():
            raise ValueError("Le texte est vide.")

        assert self.tts is not None
        LOGGER.info("Generating TTS using prompt %s", audio_prompt_path)
        segments = split_text_and_pauses(text)
        audio = self._build_audio_from_segments(
            segments,
            audio_prompt_path,
            exaggeration,
            cfg_weight,
            temperature,
            repetition_penalty,
        )

        if out_path is None:
            raise ValueError("out_path must be provided")

        out_path = str(Path(out_path).expanduser().resolve())
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        sr = self.sample_rate or self.tts.sr
        sf.write(out_path, audio, sr)
        LOGGER.info("Saved audio to %s", out_path)
        return out_path, sr

    def generate_longform(
        self,
        script: str,
        audio_prompt_path: Optional[str],
        *,
        out_path: str,
        exaggeration: float = 0.5,
        cfg_weight: float = 0.6,
        temperature: float = 0.5,
        repetition_penalty: float = 1.35,
        max_chars: int = 380,
        max_sentences: int = 3,
    ) -> tuple[str, int, Dict]:
        if not script.strip():
            raise ValueError("Le texte est vide.")

        assert self.tts is not None
        chunks = chunk_script(script, max_chars=max_chars, max_sentences=max_sentences)
        if not chunks:
            raise ValueError("Aucun chunk généré.")

        sr = self.sample_rate or self.tts.sr
        audio_chunks: List[np.ndarray] = []
        durations: List[float] = []
        clean_texts: List[str] = []
        retries: List[bool] = []

        for idx, chunk_info in enumerate(chunks, start=1):
            chunk_segments = list(chunk_info.segments)
            ensure_strong_ending(chunk_segments)
            clean_text = render_clean_text_from_segments(chunk_segments)
            clean_texts.append(clean_text)
            audio = self._build_audio_from_segments(
                chunk_segments,
                audio_prompt_path,
                exaggeration,
                cfg_weight,
                temperature,
                repetition_penalty,
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
                audio = self._build_audio_from_segments(
                    chunk_segments,
                    audio_prompt_path,
                    exaggeration,
                    min(1.5, cfg_weight + 0.05),
                    max(0.2, temperature - 0.05),
                    repetition_penalty,
                )
                duration = len(audio) / sr
                if duration > base_duration:
                    LOGGER.info("retry success (chunk %s)", idx)
                else:
                    LOGGER.info("retry failed (kept first) (chunk %s)", idx)
            retries.append(retried)
            durations.append(duration)
            audio_chunks.append(audio)

        final_audio = np.concatenate(audio_chunks) if audio_chunks else np.zeros(0, dtype=np.float32)
        out_path = str(Path(out_path).expanduser().resolve())
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        sf.write(out_path, final_audio, sr)
        meta = {
            "chunks": len(chunks),
            "durations": durations,
            "clean_texts": clean_texts,
            "retries": retries,
            "total_duration": len(final_audio) / sr if sr else 0.0,
        }
        return out_path, sr, meta


__all__ = ["TTSEngine"]
