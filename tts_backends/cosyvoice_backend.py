"""CosyVoice TTS backend wrapper (subprocess runner via SubprocessBackendMixin).

CosyVoice 3 supports three synthesis modes:
- instruct: text + emotion/style/role instruction + optional ref audio
- clone: zero-shot voice cloning from reference audio (≥3s)
- cross: cross-lingual synthesis (e.g. French voice speaking English text)

All modes support streaming (150ms first-packet) and inter-chunk gap.
License: Apache 2.0 — no usage restrictions.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import soundfile as sf

from backend_install.status import backend_status

from .base import BackendUnavailableError, ModelInfo, ParamSpec, TTSBackend, coerce_bool
from .base_runner import SubprocessBackendMixin


COSYVOICE_ASSETS_DIR = Path(__file__).resolve().parents[1] / ".assets" / "cosyvoice"

COSYVOICE_DEFAULT_MODELS = {
    "clone": "FunAudioLLM/Fun-CosyVoice3-0.5B-2512",
    "instruct": "FunAudioLLM/Fun-CosyVoice3-0.5B-2512",
    "cross_lingual": "FunAudioLLM/Fun-CosyVoice3-0.5B-2512",
}

COSYVOICE_LANGUAGE_MAP: Dict[str, str] = {
    "fr-FR": "French",
    "fr-CA": "French",
    "en-US": "English",
    "en-GB": "English",
    "zh-CN": "Chinese",
    "zh-TW": "Chinese",
    "ja-JP": "Japanese",
    "ko-KR": "Korean",
    "de-DE": "German",
    "es-ES": "Spanish",
    "it-IT": "Italian",
    "ru-RU": "Russian",
    "pt-PT": "Portuguese",
    "pt-BR": "Portuguese",
}

INSTRUCT_CHOICES = [
    ("Aucune", ""),
    ("Joyeux", "用开心的语气说"),
    ("Triste", "用伤心的语气说"),
    ("Colère", "用生气的语气说"),
    ("Surpris", "用惊讶的语气说"),
    ("Calme", "用冷静的语气说"),
    ("Rapide", "快速"),
    ("Lent", "慢速"),
]


def _ensure_wav_ref(path: str, tmp_dir: Path) -> str:
    """Convert any audio file to a normalized WAV for CosyVoice.

    CosyVoice accepts WAV at any sample rate; we normalise to 24kHz mono s16
    for consistency with the reference pipeline.
    """
    ref_path = Path(path)
    if ref_path.suffix.lower() == ".wav":
        return str(ref_path)
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise BackendUnavailableError(
            "CosyVoice voice clone requiert un WAV (ffmpeg introuvable)."
        )
    out_path = tmp_dir / f"cosyvoice_ref_{ref_path.stem}.wav"
    cmd = [
        ffmpeg, "-y", "-i", str(ref_path),
        "-ac", "1", "-ar", "24000", "-sample_fmt", "s16",
        str(out_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise BackendUnavailableError(
            f"ffmpeg conversion failed: {result.stderr[:200]}"
        )
    return str(out_path)


def _validate_ref_audio(path: str) -> dict:
    """Validate reference audio: duration ≥ 3s, RMS ≥ 0.001."""
    info = sf.info(path)
    sr = int(info.samplerate)
    audio, _ = sf.read(path, dtype="float32")
    duration_s = float(len(audio) / sr)
    if duration_s < 3.0:
        raise BackendUnavailableError(
            f"Audio de référence trop court ({duration_s:.1f}s < 3.0s requis pour CosyVoice)."
        )
    rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
    if rms < 0.001:
        raise BackendUnavailableError(
            f"Audio de référence trop silencieux (RMS={rms:.4f})."
        )
    return {"duration_s": duration_s, "rms": rms, "sample_rate": sr}


class CosyVoiceBackend(TTSBackend, SubprocessBackendMixin):
    runner_module = "cosyvoice_runner"
    runner_venv = "cosyvoice"
    default_timeout = 300.0

    id = "cosyvoice"
    display_name = "CosyVoice 3"
    supports_ref_audio = True
    supports_inter_chunk_gap = True
    uses_internal_voices = False

    _ENGINE_MODE_MAP = {
        "cosyvoice_instruct": "instruct",
        "cosyvoice_clone": "clone",
        "cosyvoice_cross": "cross_lingual",
    }

    # ------------------------------------------------------------------ #
    # Registration
    # ------------------------------------------------------------------ #

    @classmethod
    def engine_variants(cls) -> list[dict[str, str]]:
        return [
            {"id": "cosyvoice_instruct", "label": "CosyVoice (Instruct)"},
            {"id": "cosyvoice_clone", "label": "CosyVoice (Voice Clone)"},
            {"id": "cosyvoice_cross", "label": "CosyVoice (Cross-lingual)"},
        ]

    @classmethod
    def is_available(cls) -> bool:
        return backend_status("cosyvoice").get("installed", False)

    @classmethod
    def unavailable_reason(cls) -> str | None:
        return backend_status("cosyvoice").get("reason")

    # ------------------------------------------------------------------ #
    # Language
    # ------------------------------------------------------------------ #

    def supported_languages(self) -> list[str]:
        return list(COSYVOICE_LANGUAGE_MAP.keys())

    def default_language(self) -> str:
        return "fr-FR"

    def map_language(self, bcp47: Optional[str]) -> Optional[str]:
        if not bcp47:
            return "French"
        return COSYVOICE_LANGUAGE_MAP.get(bcp47, "Auto")

    # ------------------------------------------------------------------ #
    # Models
    # ------------------------------------------------------------------ #

    def list_models(self) -> list[ModelInfo]:
        return [
            ModelInfo(
                id="FunAudioLLM/Fun-CosyVoice3-0.5B-2512",
                label="CosyVoice3 0.5B (RL)",
            ),
            ModelInfo(
                id="FunAudioLLM/CosyVoice2-0.5B",
                label="CosyVoice2 0.5B",
            ),
        ]

    # ------------------------------------------------------------------ #
    # Capabilities & ref audio
    # ------------------------------------------------------------------ #

    def supports_ref_for_engine(self, engine_id: str) -> bool:
        return engine_id in {"cosyvoice_clone", "cosyvoice_cross", "cosyvoice_instruct"}

    def capabilities(self, engine_id: str | None = None) -> Dict[str, bool | list]:
        caps = super().capabilities(engine_id)
        caps["supports_instruct"] = engine_id == "cosyvoice_instruct"
        caps["supports_cross_lingual"] = engine_id == "cosyvoice_cross"
        caps["supports_streaming"] = True
        caps["supports_emotion"] = engine_id == "cosyvoice_instruct"
        caps["supports_fine_grained_control"] = engine_id == "cosyvoice_instruct"
        return caps

    # ------------------------------------------------------------------ #
    # Parameter resolution
    # ------------------------------------------------------------------ #

    def auto_resolved_keys(self, engine_id: str | None = None) -> list[str]:
        return ["cosyvoice_mode"]

    def resolve_engine_params(
        self, engine_id: str, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        default_mode = self._ENGINE_MODE_MAP.get(engine_id)
        if default_mode:
            requested = params.get("cosyvoice_mode")
            if requested in {"instruct", "clone", "cross_lingual"}:
                params["cosyvoice_mode"] = requested
            else:
                params["cosyvoice_mode"] = default_mode
        return params

    # ------------------------------------------------------------------ #
    # Schema
    # ------------------------------------------------------------------ #

    def params_schema(self) -> dict[str, ParamSpec]:
        return {
            "cosyvoice_mode": ParamSpec(
                key="cosyvoice_mode",
                type="choice",
                default="instruct",
                choices=[
                    ("Instruct (émotion/style)", "instruct"),
                    ("Clone voix", "clone"),
                    ("Cross-lingual", "cross_lingual"),
                ],
                label="Mode CosyVoice",
                help="Instruct (texte + consigne), Clone (ref audio), Cross-lingual (voix FR → texte EN).",
                visible_if={"supports_ref": False},
            ),
            "instruct_text": ParamSpec(
                key="instruct_text",
                type="str",
                default="",
                label="Instruction",
                help="Émotion, style, dialecte (ex: '用开心的语气说', '快速').",
                visible_if={"cosyvoice_mode": "instruct"},
            ),
            "instruct_preset": ParamSpec(
                key="instruct_preset",
                type="choice",
                default="",
                choices=INSTRUCT_CHOICES,
                label="Émotion preset",
                help="Preset d'émotion (remplit instruction si vide).",
                visible_if={"cosyvoice_mode": "instruct"},
            ),
            "prompt_text": ParamSpec(
                key="prompt_text",
                type="str",
                default="",
                label="Texte de référence (transcript)",
                help="Transcript exact de l'audio de référence (améliore qualité clone).",
                visible_if={"cosyvoice_mode": "clone"},
            ),
            "streaming": ParamSpec(
                key="streaming",
                type="bool",
                default=False,
                label="Streaming",
                help="Activer le streaming (150ms premier paquet).",
            ),
        }

    # ------------------------------------------------------------------ #
    # Synthesis
    # ------------------------------------------------------------------ #

    def synthesize(
        self,
        script: str,
        out_path: str,
        voice_ref_path: Optional[str] = None,
        lang: Optional[str] = None,
        **params: Any,
    ) -> Dict[str, Any]:
        raise BackendUnavailableError(
            "CosyVoice backend should use synthesize_chunk."
        )

    def synthesize_chunk(
        self,
        text: str,
        *,
        voice_ref_path: Optional[str] = None,
        lang: Optional[str] = None,
        **params: Any,
    ):
        engine_id = params.get("engine_id") or "cosyvoice_clone"
        mode = self._ENGINE_MODE_MAP.get(engine_id, "clone")

        # Allow explicit mode override via param
        explicit_mode = params.get("cosyvoice_mode")
        if explicit_mode in {"instruct", "clone", "cross_lingual"}:
            mode = explicit_mode

        # Fallback: if voice_ref provided and mode is instruct, switch to clone
        if mode == "instruct" and voice_ref_path and not params.get("instruct_text"):
            # If user provides ref audio but no instruct text, clone is more appropriate
            pass  # instruct with ref audio is valid in CosyVoice3 (inference_instruct2)

        if mode == "clone" and not voice_ref_path:
            raise BackendUnavailableError(
                "CosyVoice clone requiert un audio de référence (≥3s)."
            )
        if mode == "cross_lingual" and not voice_ref_path:
            raise BackendUnavailableError(
                "CosyVoice cross-lingual requiert un audio de référence."
            )

        COSYVOICE_ASSETS_DIR.mkdir(parents=True, exist_ok=True)
        tmp_dir = COSYVOICE_ASSETS_DIR / ".tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)

        ref_audio_path = None
        if voice_ref_path:
            ref_audio_path = _ensure_wav_ref(voice_ref_path, tmp_dir)
            _validate_ref_audio(ref_audio_path)

        model_id = params.get("model_id") or COSYVOICE_DEFAULT_MODELS.get(
            mode, "FunAudioLLM/Fun-CosyVoice3-0.5B-2512"
        )

        # Resolve instruct text: explicit > preset > empty
        instruct_text = params.get("instruct_text") or ""
        if not instruct_text:
            preset = params.get("instruct_preset") or ""
            if preset:
                instruct_text = preset

        prompt_text = params.get("prompt_text") or ""
        streaming = coerce_bool(params.get("streaming"), False)

        payload_suffix = {
            "mode": mode,
            "model_id": model_id,
            "language": self.map_language(lang),
            "instruct_text": instruct_text,
            "prompt_text": prompt_text,
            "streaming": streaming,
            "assets_dir": str(COSYVOICE_ASSETS_DIR),
        }
        if ref_audio_path:
            payload_suffix["voice_ref_path"] = ref_audio_path
        elif voice_ref_path:
            payload_suffix["voice_ref_path"] = voice_ref_path

        timeout_s = 600.0 if mode == "clone" else 300.0
        audio, sr, meta = self._run_subprocess_chunk(
            text,
            voice_ref_path=None,  # Already in payload_suffix
            lang=lang,
            payload_suffix=payload_suffix,
            timeout_s=timeout_s,
        )
        meta["backend_id"] = self.id
        meta["backend_lang"] = lang
        meta["cosyvoice_mode"] = mode
        meta["cosyvoice_model"] = model_id
        meta["cosyvoice_streaming"] = streaming
        return audio, sr, meta


__all__ = ["CosyVoiceBackend", "COSYVOICE_LANGUAGE_MAP"]