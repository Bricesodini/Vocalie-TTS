"""XTTS-v2 backend wrapper (optional dependency)."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any, Dict, Optional

import numpy as np
import soundfile as sf

from backend_install.paths import python_path
from backend_install.status import backend_status

from .base import BackendUnavailableError, ParamSpec, TTSBackend

XTTS_DEFAULT_MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"
XTTS_ASSETS_DIR = Path(__file__).resolve().parents[1] / ".assets" / "xtts"


def _xtts_lang_code(bcp47: Optional[str]) -> Optional[str]:
    if not bcp47:
        return None
    return bcp47.split("-")[0]


def xtts_model_available() -> bool:
    if not XTTS_ASSETS_DIR.exists():
        return False
    candidates = [
        XTTS_ASSETS_DIR / "tts_models--multilingual--multi-dataset--xtts_v2",
        XTTS_ASSETS_DIR / "tts_models" / "multilingual" / "multi-dataset" / "xtts_v2",
    ]
    for path in candidates:
        if path.exists():
            return True
    return False


def _ensure_wav_ref(path: str, tmp_dir: Path) -> str:
    ref_path = Path(path)
    if ref_path.suffix.lower() == ".wav":
        return str(ref_path)
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise BackendUnavailableError("XTTS nécessite un ref_audio WAV (ffmpeg introuvable).")
    out_path = tmp_dir / f"xtts_ref_{ref_path.stem}.wav"
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(ref_path),
        "-ac",
        "1",
        "-ar",
        "22050",
        str(out_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return str(out_path)


class XTTSBackend(TTSBackend):
    id = "xtts"
    display_name = "XTTS v2 (voice cloning)"
    supports_ref_audio = True
    uses_internal_voices = False

    @classmethod
    def is_available(cls) -> bool:
        return backend_status("xtts").get("installed", False)

    @classmethod
    def unavailable_reason(cls) -> str | None:
        return backend_status("xtts").get("reason")

    def supported_languages(self) -> list[str]:
        return ["fr-FR", "en-US", "es-ES", "de-DE", "it-IT", "pt-PT", "nl-NL"]

    def default_language(self) -> str:
        return "fr-FR"

    def params_schema(self) -> dict[str, ParamSpec]:
        return {
            "speed": ParamSpec(
                key="speed",
                type="float",
                default=1.0,
                min=0.5,
                max=2.0,
                step=0.05,
                label="Vitesse",
            ),
        }

    def map_language(self, bcp47: Optional[str]) -> Optional[str]:
        if not bcp47:
            return None
        return bcp47.split("-")[0]

    def synthesize(
        self,
        script: str,
        out_path: str,
        voice_ref_path: Optional[str] = None,
        lang: Optional[str] = None,
        **params: Any,
    ) -> Dict[str, Any]:
        raise BackendUnavailableError("XTTS backend should use synthesize_chunk.")

    @staticmethod
    def build_command(
        py: Path,
        runner: Path,
        text: str,
        out_path: str,
        speaker_wav: str,
        language: Optional[str],
        model_id: str,
        speed: float | None,
        meta_json: str,
    ) -> list[str]:
        cmd = [
            str(py),
            str(runner),
            "--text",
            text,
            "--speaker_wav",
            speaker_wav,
            "--out_path",
            out_path,
            "--model_id",
            model_id,
            "--meta_json",
            meta_json,
        ]
        if language:
            cmd.extend(["--language", language])
        if speed is not None:
            cmd.extend(["--speed", str(speed)])
        return cmd

    def synthesize_chunk(
        self,
        text: str,
        *,
        voice_ref_path: Optional[str] = None,
        lang: Optional[str] = None,
        **params: Any,
    ):
        if not voice_ref_path:
            raise BackendUnavailableError("XTTS nécessite une référence vocale.")
        py = python_path("xtts")
        if not py.exists():
            raise BackendUnavailableError("XTTS venv introuvable.")

        XTTS_ASSETS_DIR.mkdir(parents=True, exist_ok=True)
        runner = Path(__file__).resolve().parent / "xtts_runner.py"
        tmp_dir = XTTS_ASSETS_DIR / ".tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        ref_path = _ensure_wav_ref(voice_ref_path, tmp_dir)
        tmp_wav = tmp_dir / f"xtts_{abs(hash(text))}.wav"
        meta_json = tmp_dir / f"xtts_{abs(hash(text))}.json"
        log_path = tmp_dir / "xtts_runner.log"
        model_id = params.get("model_id", XTTS_DEFAULT_MODEL)
        speed = params.get("speed")
        language = _xtts_lang_code(lang or "fr-FR")
        cmd = self.build_command(
            py,
            runner,
            text,
            str(tmp_wav),
            ref_path,
            language,
            model_id,
            float(speed) if speed is not None else None,
            str(meta_json),
        )
        env = dict(**os.environ)
        env["TTS_HOME"] = str(XTTS_ASSETS_DIR)
        env["COQUI_TOS_AGREED"] = "1"
        env["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        try:
            with log_path.open("w", encoding="utf-8") as log_fh:
                proc = subprocess.run(
                    cmd,
                    check=True,
                    stdout=log_fh,
                    stderr=log_fh,
                    input="y\n",
                    text=True,
                    env=env,
                )
        except subprocess.CalledProcessError as exc:
            raise BackendUnavailableError(exc.stderr or exc.stdout or "XTTS failed") from exc

        audio, sr = sf.read(str(tmp_wav), dtype="float32")
        meta = {}
        if meta_json.exists():
            try:
                meta = json.loads(meta_json.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                meta = {}
        meta["log_path"] = str(log_path)
        try:
            log_text = log_path.read_text(encoding="utf-8").strip()
        except OSError:
            log_text = ""
        if log_text:
            meta["stdout"] = log_text
        return np.asarray(audio, dtype=np.float32), int(sr), meta
