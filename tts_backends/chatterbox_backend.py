"""Chatterbox TTS backend wrapper (subprocess runner)."""

from __future__ import annotations

import json
import subprocess
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import soundfile as sf

from backend_install.paths import ROOT, python_path
from backend_install.status import backend_status

from .base import BackendUnavailableError, ParamSpec, TTSBackend


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


def _runner_path() -> Path:
    return Path(__file__).resolve().parent / "chatterbox_runner.py"


def _run_chatterbox_runner(payload: dict, timeout_s: float = 180.0) -> dict:
    py = python_path("chatterbox")
    if not py.exists():
        raise BackendUnavailableError("Chatterbox venv introuvable.")
    runner = _runner_path()
    if not runner.exists():
        raise BackendUnavailableError("Runner Chatterbox introuvable.")
    try:
        result = subprocess.run(
            [str(py), str(runner)],
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as exc:
        raise BackendUnavailableError("Chatterbox runner timeout.") from exc

    stdout = (result.stdout or "").strip()
    stderr = (result.stderr or "").strip()
    if result.returncode != 0:
        raise BackendUnavailableError(stderr or stdout or "Chatterbox runner failed.")
    if not stdout:
        raise BackendUnavailableError("Chatterbox runner returned no output.")
    try:
        data = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise BackendUnavailableError("Chatterbox runner returned invalid JSON.") from exc
    if not data.get("ok"):
        error = data.get("error") or "Chatterbox runner failed."
        trace = data.get("trace")
        if trace:
            error = f"{error}\n{trace}"
        raise BackendUnavailableError(error)
    if stderr:
        data["stderr"] = stderr
    return data


class ChatterboxBackend(TTSBackend):
    id = "chatterbox"
    display_name = "Chatterbox (stable long-form)"
    supports_ref_audio = True
    uses_internal_voices = False

    @classmethod
    def is_available(cls) -> bool:
        return backend_status("chatterbox").get("installed", False)

    @classmethod
    def unavailable_reason(cls) -> str | None:
        return backend_status("chatterbox").get("reason")

    def supported_languages(self) -> list[str]:
        return list(LANGUAGE_MAP.keys())

    def default_language(self) -> str:
        return "fr-FR"

    def params_schema(self) -> dict[str, ParamSpec]:
        return {
            "chatterbox_mode": ParamSpec(
                key="chatterbox_mode",
                type="choice",
                default="fr_finetune",
                choices=[
                    ("FR fine-tuné (spécialisé)", "fr_finetune"),
                    ("Chatterbox multilangue", "multilang"),
                ],
                label="Mode Chatterbox",
                help="Fine-tune FR ou multilangue.",
            ),
            "multilang_cfg_weight": ParamSpec(
                key="multilang_cfg_weight",
                type="float",
                default=0.5,
                min=0.0,
                max=1.0,
                step=0.05,
                label="CFG multilangue",
                help="Réduire pour limiter l'accent bleed.",
                visible_if={"chatterbox_mode": "multilang"},
            ),
            "exaggeration": ParamSpec(
                key="exaggeration",
                type="float",
                default=0.5,
                min=0.0,
                max=1.0,
                step=0.05,
                label="Exaggeration",
            ),
            "cfg_weight": ParamSpec(
                key="cfg_weight",
                type="float",
                default=0.6,
                min=0.0,
                max=1.0,
                step=0.05,
                label="CFG",
            ),
            "temperature": ParamSpec(
                key="temperature",
                type="float",
                default=0.5,
                min=0.0,
                max=1.0,
                step=0.05,
                label="Température",
            ),
            "repetition_penalty": ParamSpec(
                key="repetition_penalty",
                type="float",
                default=1.35,
                min=0.5,
                max=2.0,
                step=0.05,
                label="Repetition penalty",
            ),
        }

    def map_language(self, bcp47: Optional[str]) -> Optional[str]:
        if not bcp47:
            return "fr"
        return LANGUAGE_MAP.get(bcp47, bcp47.split("-")[0])

    def synthesize(
        self,
        script: str,
        out_path: str,
        voice_ref_path: Optional[str] = None,
        lang: Optional[str] = None,
        **params: Any,
    ) -> Dict[str, Any]:
        payload = _build_runner_payload(
            text=script,
            out_wav_path=out_path,
            voice_ref_path=voice_ref_path,
            lang=lang,
            params=params,
        )
        result = _run_chatterbox_runner(payload)
        meta = {
            "backend_id": self.id,
            "backend_lang": lang,
            "out_path": result.get("out_path") or out_path,
            "duration_s": result.get("duration_s"),
            "retry": bool(result.get("retry")),
        }
        logs = result.get("logs")
        if logs:
            meta["runner_logs"] = logs
        if result.get("stderr"):
            meta["stderr"] = result.get("stderr")
        return meta

    def synthesize_chunk(
        self,
        text: str,
        *,
        voice_ref_path: Optional[str] = None,
        lang: Optional[str] = None,
        **params: Any,
    ):
        tmp_dir = ROOT / ".assets" / "chatterbox" / ".tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp_wav = tmp_dir / f"chatterbox_{uuid.uuid4().hex}.wav"
        payload = _build_runner_payload(
            text=text,
            out_wav_path=str(tmp_wav),
            voice_ref_path=voice_ref_path,
            lang=lang,
            params=params,
        )
        result = _run_chatterbox_runner(payload)
        if not tmp_wav.exists():
            raise BackendUnavailableError("Chatterbox runner n'a pas produit de WAV.")
        audio, sr = sf.read(str(tmp_wav), dtype="float32")
        try:
            tmp_wav.unlink(missing_ok=True)
        except OSError:
            pass
        meta = {
            "retry": bool(result.get("retry")),
            "duration_s": result.get("duration_s"),
        }
        logs = result.get("logs")
        if logs:
            meta["runner_logs"] = logs
        if result.get("stderr"):
            meta["stderr"] = result.get("stderr")
        return np.asarray(audio, dtype=np.float32), int(sr), meta


def _build_runner_payload(
    *,
    text: str,
    out_wav_path: str,
    voice_ref_path: Optional[str],
    lang: Optional[str],
    params: dict[str, Any],
) -> dict:
    tts_model_mode = params.get("tts_model_mode", params.get("chatterbox_mode", "fr_finetune"))
    payload = {
        "text": text,
        "out_wav_path": out_wav_path,
        "tts_model_mode": tts_model_mode,
        "lang": lang,
        "multilang_cfg_weight": params.get("multilang_cfg_weight", 0.5),
        "exaggeration": params.get("exaggeration", 0.5),
        "cfg_weight": params.get("cfg_weight", 0.6),
        "temperature": params.get("temperature", 0.5),
        "repetition_penalty": params.get("repetition_penalty", 1.35),
    }
    if voice_ref_path:
        payload["ref_audio_path"] = voice_ref_path
    return payload
