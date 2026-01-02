"""Chatterbox-only implementation for the isolated runner."""

from __future__ import annotations

import inspect
import logging
from typing import Optional

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

try:
    from chatterbox.tts import ChatterboxTTS
except ModuleNotFoundError:  # pragma: no cover - handled in venv
    ChatterboxTTS = None
try:
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS
except ImportError:  # pragma: no cover - optional backend depending on package version
    ChatterboxMultilingualTTS = None

LOGGER = logging.getLogger("chatterbox_runner")

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


def _ensure_repo_override() -> None:
    """Override chatterbox.tts.REPO_ID before calling from_pretrained."""
    import chatterbox.tts as tts_mod

    if getattr(tts_mod, "REPO_ID", None) != BASE_REPO:
        tts_mod.REPO_ID = BASE_REPO


class ChatterboxEngine:
    """Minimal engine for the isolated Chatterbox runner."""

    def __init__(self, device: Optional[str] = None) -> None:
        if ChatterboxTTS is None:
            raise RuntimeError("ChatterboxTTS backend is unavailable.")
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self._tts_fr_finetune: Optional[ChatterboxTTS] = None
        self._tts_multilang: Optional[ChatterboxTTS] = None
        self.sample_rate: Optional[int] = None
        self._lang_support: dict[int, bool] = {}

    def _load_fr_backend(self) -> ChatterboxTTS:
        _ensure_repo_override()
        tts = ChatterboxTTS.from_pretrained(self.device)
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

    def synthesize_text(
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
