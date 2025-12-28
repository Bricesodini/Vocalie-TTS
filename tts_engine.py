"""Wrapper around Chatterbox TTS with Thomcles fine-tune replacement."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import soundfile as sf
import torch
from chatterbox.tts import ChatterboxTTS
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from text_tools import SpeechSegment, split_text_and_pauses

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
        audio_chunks: List[np.ndarray] = []

        for segment in segments:
            if segment.kind == "silence":
                frames = int(sr * (segment.duration_ms / 1000.0))
                if frames <= 0:
                    continue
                audio_chunks.append(np.zeros(frames, dtype=np.float32))
                continue

            clean = segment.content.strip()
            if not clean:
                continue
            chunk = self._synthesize_text(
                clean,
                audio_prompt_path,
                exaggeration,
                cfg_weight,
                temperature,
                repetition_penalty,
            )
            audio_chunks.append(chunk)

        if not audio_chunks:
            raise ValueError("Aucun segment audio généré.")

        return np.concatenate(audio_chunks)

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


__all__ = ["TTSEngine"]
