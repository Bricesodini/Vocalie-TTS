"""Qwen3 TTS backend wrapper (subprocess runner via SubprocessBackendMixin)."""

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
from .catalog import QWEN3_LANGUAGE_MAP


QWEN3_ASSETS_DIR = Path(__file__).resolve().parents[1] / ".assets" / "qwen3"
QWEN3_DEFAULT_MODELS = {
    "custom_voice": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "voice_design": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    "voice_clone": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
}

SPEAKER_CHOICES = [
    ("Vivian (F, Chinese)", "Vivian"),
    ("Serena (F, Chinese)", "Serena"),
    ("Uncle_Fu (M, Chinese)", "Uncle_Fu"),
    ("Dylan (M, English)", "Dylan"),
    ("Eric (M, English)", "Eric"),
    ("Ryan (M, English)", "Ryan"),
    ("Aiden (M, English)", "Aiden"),
    ("Ono_Anna (F, Japanese)", "Ono_Anna"),
    ("Sohee (F, Korean)", "Sohee"),
]


def _ensure_wav_ref(path: str, tmp_dir: Path) -> str:
    """Convert any audio file to a normalized WAV suitable for voice cloning.

    Applies ffmpeg normalisation: mono, 24 kHz, s16 sample format,
    loudnorm filter (with fallback if loudnorm is unavailable).
    """
    ref_path = Path(path)
    if ref_path.suffix.lower() == ".wav":
        return str(ref_path)
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise BackendUnavailableError("Qwen3 voice clone requiert un WAV (ffmpeg introuvable).")
    out_path = tmp_dir / f"qwen3_ref_{ref_path.stem}.wav"
    # Try with loudnorm filter first for consistent loudness
    cmd = [
        ffmpeg, "-y", "-i", str(ref_path),
        "-ac", "1", "-ar", "24000", "-sample_fmt", "s16",
        "-af", "loudnorm",
        str(out_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        # Fallback without audio filter (loudnorm may not be available)
        cmd_simple = [
            ffmpeg, "-y", "-i", str(ref_path),
            "-ac", "1", "-ar", "24000", "-sample_fmt", "s16",
            str(out_path),
        ]
        subprocess.run(cmd_simple, check=True, capture_output=True, text=True)
    return str(out_path)


def _validate_ref_audio(path: str) -> dict:
    """Validate that a reference audio file is suitable for voice cloning.

    Checks duration (>= 1s) and RMS level (not silence).
    Returns a metrics dict on success, raises BackendUnavailableError on failure.
    """
    info = sf.info(path)
    sr = int(info.samplerate)
    audio, _ = sf.read(path, dtype="float32")
    duration_s = float(len(audio) / sr)
    if duration_s < 1.0:
        raise BackendUnavailableError(
            f"Audio de reference trop court ({duration_s:.1f}s < 1.0s)."
        )
    rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
    if rms < 0.001:
        raise BackendUnavailableError(
            f"Audio de reference trop silencieux (RMS={rms:.4f})."
        )
    return {"duration_s": duration_s, "rms": rms, "sample_rate": sr}


class Qwen3Backend(TTSBackend, SubprocessBackendMixin):
    runner_module = "qwen3_runner"
    runner_venv = "qwen3"
    default_timeout = 300.0

    id = "qwen3"
    display_name = "Qwen3 TTS"
    supports_ref_audio = False
    uses_internal_voices = False
    supports_inter_chunk_gap = True

    _ENGINE_MODE_MAP = {
        "qwen3_custom": "custom_voice",
        "qwen3_clone": "voice_clone",
    }

    @classmethod
    def engine_variants(cls) -> list[dict[str, str]]:
        return [
            {"id": "qwen3_custom", "label": "Qwen3 (CustomVoice/Design)"},
            {"id": "qwen3_clone", "label": "Qwen3 (Voice clone)"},
        ]

    @classmethod
    def is_available(cls) -> bool:
        return backend_status("qwen3").get("installed", False)

    @classmethod
    def unavailable_reason(cls) -> str | None:
        return backend_status("qwen3").get("reason")

    def supported_languages(self) -> list[str]:
        return list(QWEN3_LANGUAGE_MAP.keys())

    def default_language(self) -> str:
        return "fr-FR"

    def list_models(self) -> list[ModelInfo]:
        return [
            ModelInfo(
                id=v, label=f"Qwen3-TTS {mode.replace('_', ' ').title()}",
                meta={"mode": mode},
            )
            for mode, v in QWEN3_DEFAULT_MODELS.items()
        ]

    def supports_ref_for_engine(self, engine_id: str) -> bool:
        return engine_id == "qwen3_clone"

    def auto_resolved_keys(self, engine_id: str | None = None) -> list[str]:
        return ["qwen3_mode"]

    def capabilities(self, engine_id: str | None = None) -> Dict[str, bool | list]:
        caps = super().capabilities(engine_id)
        caps["can_refresh_speakers"] = True
        caps["supports_voice_design"] = engine_id == "qwen3_custom"
        return caps

    def resolve_engine_params(self, engine_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
        default_mode = self._ENGINE_MODE_MAP.get(engine_id)
        if default_mode:
            requested = params.get("qwen3_mode")
            if requested in {"custom_voice", "voice_design", "voice_clone"}:
                params["qwen3_mode"] = requested
            else:
                params["qwen3_mode"] = default_mode
        return params

    def params_schema(self) -> dict[str, ParamSpec]:
        return {
            "qwen3_mode": ParamSpec(
                key="qwen3_mode",
                type="choice",
                default="custom_voice",
                choices=[
                    ("Voix CustomVoice", "custom_voice"),
                    ("Voice design", "voice_design"),
                ],
                label="Mode Qwen3",
                help="CustomVoice (speakers) ou VoiceDesign (instruction).",
                visible_if={"supports_ref": False},
            ),
            "speaker": ParamSpec(
                key="speaker",
                type="select",
                default="Vivian",
                choices=SPEAKER_CHOICES,
                label="Speaker",
                help="Selectionne un speaker CustomVoice.",
                visible_if={
                    "supports_ref": False,
                    "qwen3_mode": "custom_voice",
                },
            ),
            "emotion": ParamSpec(
                key="emotion",
                type="choice",
                default="neutral",
                choices=[
                    ("Neutre", "neutral"),
                    ("Joyeux", "Very happy"),
                    ("Triste", "Sad"),
                    ("Colere", "Angry"),
                    ("Excite", "Excited"),
                    ("Calme", "Calm"),
                ],
                label="Emotion",
                help="Ajoute une instruction si aucune instruction manuelle.",
                visible_if={"supports_ref": False},
            ),
            "instruct": ParamSpec(
                key="instruct",
                type="str",
                default="",
                label="Instruction",
                help="Style/intonation (optionnel).",
                visible_if={"supports_ref": False},
            ),
            "x_vector_only_mode": ParamSpec(
                key="x_vector_only_mode",
                type="bool",
                default=True,
                label="x-vector only",
                help="Pas besoin de transcript; clonage un peu moins precis.",
                visible_if={"supports_ref": True},
            ),
            "ref_text": ParamSpec(
                key="ref_text",
                type="str",
                default="",
                label="Texte de reference",
                help="Transcript exact de l'audio de reference.",
                visible_if={"supports_ref": True, "x_vector_only_mode": False},
            ),
        }

    def map_language(self, bcp47: Optional[str]) -> Optional[str]:
        if not bcp47:
            return "French"
        return QWEN3_LANGUAGE_MAP.get(bcp47, "Auto")

    def synthesize(
        self,
        script: str,
        out_path: str,
        voice_ref_path: Optional[str] = None,
        lang: Optional[str] = None,
        **params: Any,
    ) -> Dict[str, Any]:
        raise BackendUnavailableError("Qwen3 backend should use synthesize_chunk.")

    def synthesize_chunk(
        self,
        text: str,
        *,
        voice_ref_path: Optional[str] = None,
        lang: Optional[str] = None,
        **params: Any,
    ):
        mode = str(params.get("qwen3_mode") or "custom_voice")
        if mode not in {"custom_voice", "voice_design", "voice_clone"}:
            mode = "custom_voice"
        if mode == "custom_voice" and voice_ref_path and "qwen3_mode" not in params:
            mode = "voice_clone"

        if mode == "voice_clone" and not voice_ref_path:
            raise BackendUnavailableError("Qwen3 voice clone requiert un ref audio.")

        QWEN3_ASSETS_DIR.mkdir(parents=True, exist_ok=True)
        tmp_dir = QWEN3_ASSETS_DIR / ".tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)

        ref_audio_path = None
        if mode == "voice_clone" and voice_ref_path:
            ref_audio_path = _ensure_wav_ref(voice_ref_path, tmp_dir)
            _validate_ref_audio(ref_audio_path)

        model_id = params.get("model_id") or QWEN3_DEFAULT_MODELS.get(mode)
        speaker = params.get("voice") or params.get("voice_id") or params.get("speaker")
        if mode != "custom_voice":
            speaker = None
        instruct = params.get("instruct") or ""
        emotion = params.get("emotion")
        if not instruct and emotion:
            if str(emotion) != "neutral":
                instruct = str(emotion)

        payload_suffix = {
            "mode": mode,
            "model_id": model_id,
            "language": self.map_language(lang),
            "speaker": speaker,
            "instruct": instruct,
            "ref_text": params.get("ref_text") or "",
            "x_vector_only_mode": coerce_bool(params.get("x_vector_only_mode"), True),
            "assets_dir": str(QWEN3_ASSETS_DIR),
            "params": {
                "device": params.get("device"),
                "dtype": params.get("dtype"),
                "attn_implementation": params.get("attn_implementation"),
            },
        }
        if ref_audio_path:
            payload_suffix["voice_ref_path"] = ref_audio_path
        elif voice_ref_path:
            payload_suffix["voice_ref_path"] = voice_ref_path

        timeout_s = 900.0 if mode == "voice_clone" else 300.0
        audio, sr, meta = self._run_subprocess_chunk(
            text,
            voice_ref_path=None,  # Already in payload_suffix
            lang=lang,
            payload_suffix=payload_suffix,
            timeout_s=timeout_s,
        )
        meta["backend_id"] = self.id
        meta["backend_lang"] = lang
        meta["qwen3_mode"] = mode
        meta["qwen3_model"] = model_id
        meta["qwen3_speaker"] = speaker
        return audio, sr, meta


__all__ = ["Qwen3Backend"]