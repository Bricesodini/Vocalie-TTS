"""Piper backend wrapper (optional dependency)."""

from __future__ import annotations

from typing import Any, Dict, Optional

import subprocess
from pathlib import Path

from backend_install.paths import python_path
from backend_install.status import backend_status

from .base import BackendUnavailableError, ParamSpec, TTSBackend, VoiceInfo
from .piper_assets import DEFAULT_VOICE_ID, get_voice_info, list_installed_voices, list_piper_voices


class PiperBackend(TTSBackend):
    id = "piper"
    display_name = "Piper (fast offline)"
    supports_ref_audio = False
    uses_internal_voices = True

    def __init__(self) -> None:
        self._voice_ids = list_installed_voices()
        if self._voice_ids and DEFAULT_VOICE_ID in self._voice_ids:
            self._default_voice = DEFAULT_VOICE_ID
        elif self._voice_ids:
            self._default_voice = self._voice_ids[0]
        else:
            self._default_voice = None

    @classmethod
    def is_available(cls) -> bool:
        return backend_status("piper").get("installed", False)

    @classmethod
    def unavailable_reason(cls) -> str | None:
        return backend_status("piper").get("reason")

    def supported_languages(self) -> list[str]:
        return ["fr-FR"]

    def default_language(self) -> str:
        return "fr-FR"

    def list_voices(self) -> list[VoiceInfo]:
        voices = list_piper_voices()
        return [voice for voice in voices if voice.installed]

    def params_schema(self) -> dict[str, ParamSpec]:
        voices = self.list_voices()
        choices = [(voice.label, voice.id) for voice in voices]
        default_voice = self._default_voice if choices else None
        if choices and default_voice not in [voice.id for voice in voices]:
            default_voice = voices[0].id
        return {
            "voice_id": ParamSpec(
                key="voice_id",
                type="select",
                default=default_voice,
                choices=choices,
                label="Voix Piper",
                visible_if={"uses_internal_voices": True, "voice_count_min": 1},
            ),
            "speed": ParamSpec(
                key="speed",
                type="float",
                default=1.0,
                min=0.5,
                max=1.5,
                step=0.05,
                label="Vitesse",
                help="1.0 = normal",
                visible_if={"uses_internal_voices": True, "piper_supports_speed": True},
            ),
        }

    def map_language(self, bcp47: Optional[str]) -> Optional[str]:
        if not bcp47:
            return None
        return bcp47.split("-")[0]

    @staticmethod
    def build_command(
        py: Path,
        runner: Path,
        script: str,
        out_path: str,
        info: dict,
        lang: Optional[str],
        length_scale: float | None = None,
    ):
        cmd = [
            str(py),
            str(runner),
            "--text",
            script,
            "--out_wav",
            out_path,
            "--voice",
            info["voice_id"],
            "--model_dir",
            info["model_dir"],
        ]
        if lang:
            cmd.extend(["--lang", lang])
        if length_scale is not None:
            cmd.extend(["--length_scale", str(length_scale)])
        return cmd

    def synthesize(
        self,
        script: str,
        out_path: str,
        voice_ref_path: Optional[str] = None,
        lang: Optional[str] = None,
        **params: Any,
    ) -> Dict[str, Any]:
        py = python_path("piper")
        if not py.exists():
            raise BackendUnavailableError("Piper venv introuvable.")

        voice_id = params.get("voice") or params.get("voice_id") or self._default_voice
        info = get_voice_info(voice_id) if voice_id else None
        if info is None:
            raise BackendUnavailableError("Piper voice unavailable: aucune voix installée.")

        length_scale = None
        speed = params.get("speed")
        if speed is not None:
            try:
                speed_value = float(speed)
                if speed_value > 0:
                    length_scale = 1.0 / speed_value
            except (TypeError, ValueError):
                length_scale = None

        runner = Path(__file__).resolve().parent / "piper_runner.py"
        cmd = self.build_command(py, runner, script, out_path, info, lang, length_scale)
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(exc.stderr or exc.stdout or "Piper failed") from exc

        return {
            "backend_id": self.id,
            "backend_lang": lang,
            "warnings": [],
            "total_duration": 0.0,
            "durations": [],
            "chunks": 1,
            "piper_voice": info["voice_id"],
            "piper_model_path": info["model_path"],
            "backend_cmd": " ".join(cmd),
        }
