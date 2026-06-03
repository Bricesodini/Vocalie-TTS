"""Qwen3 TTS backend wrapper (optional dependency)."""

from __future__ import annotations

import json
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import soundfile as sf

from backend_install.paths import ROOT, python_path
from backend_install.status import backend_status

from .base import BackendUnavailableError, ModelInfo, ParamSpec, TTSBackend, coerce_bool
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

def _runner_path() -> Path:
    return Path(__file__).resolve().parent / "qwen3_runner.py"


def _run_qwen3_runner(payload: dict, timeout_s: float = 300.0) -> dict:
    py = python_path("qwen3")
    if not py.exists():
        raise BackendUnavailableError("Qwen3 venv introuvable.")
    runner = _runner_path()
    if not runner.exists():
        raise BackendUnavailableError("Runner Qwen3 introuvable.")
    try:
        result = subprocess.run(
            [str(py), str(runner)],
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as exc:
        raise BackendUnavailableError("Qwen3 runner timeout.") from exc

    stdout = (result.stdout or "").strip()
    stderr = (result.stderr or "").strip()
    if result.returncode != 0:
        raise BackendUnavailableError(stderr or stdout or "Qwen3 runner failed.")
    if not stdout:
        raise BackendUnavailableError("Qwen3 runner returned no output.")
    try:
        data = json.loads(stdout)
    except json.JSONDecodeError:
        lines = [line for line in stdout.splitlines() if line.strip()]
        if not lines:
            raise BackendUnavailableError("Qwen3 runner returned invalid JSON.")
        try:
            data = json.loads(lines[-1])
        except json.JSONDecodeError as exc:
            raise BackendUnavailableError("Qwen3 runner returned invalid JSON.") from exc
    if not data.get("ok"):
        error = data.get("error") or "Qwen3 runner failed."
        detail = data.get("detail")
        if detail:
            error = f"{error}\n{detail}"
        raise BackendUnavailableError(error)
    if stderr:
        data["stderr"] = stderr
    return data


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


class Qwen3Backend(TTSBackend):
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
        """qwen3_clone requires a reference voice; others do not."""
        return engine_id == "qwen3_clone"

    def auto_resolved_keys(self, engine_id: str | None = None) -> list[str]:
        """qwen3_mode is resolved from engine_id by resolve_engine_params."""
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
        py = python_path("qwen3")
        if not py.exists():
            raise BackendUnavailableError("Qwen3 venv introuvable.")

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
        run_id = uuid.uuid4().hex
        tmp_wav = tmp_dir / f"qwen3_{run_id}.wav"
        debug_log_path = tmp_dir / "qwen3_last.log"
        try:
            debug_log_path.write_text("", encoding="utf-8")
        except OSError:
            pass
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

        payload = {
            "text": text,
            "out_path": str(tmp_wav),
            "mode": mode,
            "model_id": model_id,
            "language": self.map_language(lang),
            "speaker": speaker,
            "instruct": instruct,
            "ref_text": params.get("ref_text") or "",
            "x_vector_only_mode": coerce_bool(params.get("x_vector_only_mode"), True),
            "voice_ref_path": ref_audio_path or voice_ref_path,
            "assets_dir": str(QWEN3_ASSETS_DIR),
            "debug_log_path": str(debug_log_path),
            "params": {
                "device": params.get("device"),
                "dtype": params.get("dtype"),
                "attn_implementation": params.get("attn_implementation"),
            },
        }
        timeout_s = 900.0 if mode == "voice_clone" else 300.0
        result = _run_qwen3_runner(payload, timeout_s=timeout_s)
        if not tmp_wav.exists():
            raise BackendUnavailableError("Qwen3 runner n'a pas produit de WAV.")
        audio, sr = sf.read(str(tmp_wav), dtype="float32")
        try:
            tmp_wav.unlink(missing_ok=True)
        except OSError:
            pass
        meta = {
            "backend_id": self.id,
            "backend_lang": lang,
            "qwen3_mode": mode,
            "qwen3_model": model_id,
            "qwen3_speaker": speaker,
        }
        if debug_log_path.exists():
            meta["debug_log_path"] = str(debug_log_path)
        if result.get("stderr"):
            meta["stderr"] = result.get("stderr")
        return np.asarray(audio, dtype=np.float32), int(sr), meta


__all__ = ["Qwen3Backend"]
