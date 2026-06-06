"""Chatterbox TTS backend wrapper (subprocess runner via SubprocessBackendMixin)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from backend_install.status import backend_status

from .base import BackendUnavailableError, ModelInfo, ParamSpec, TTSBackend
from .base_runner import SubprocessBackendMixin
from .catalog import CHATTERBOX_LANGUAGE_MAP


class ChatterboxBackend(TTSBackend, SubprocessBackendMixin):
    runner_module = "chatterbox_runner"
    runner_venv = "chatterbox"
    default_timeout = 180.0

    id = "chatterbox"
    display_name = "Chatterbox (stable long-form)"
    supports_ref_audio = True
    uses_internal_voices = False
    supports_inter_chunk_gap = True

    _ENGINE_MODE_MAP = {
        "chatterbox_native": "multilang",
        "chatterbox_finetune_fr": "fr_finetune",
    }

    @classmethod
    def engine_variants(cls) -> list[dict[str, str]]:
        return [
            {"id": "chatterbox_native", "label": "Chatterbox (native multilang)"},
            {"id": "chatterbox_finetune_fr", "label": "Chatterbox (FR fine-tune)"},
        ]

    @classmethod
    def is_available(cls) -> bool:
        return backend_status("chatterbox").get("installed", False)

    @classmethod
    def unavailable_reason(cls) -> str | None:
        return backend_status("chatterbox").get("reason")

    def supported_languages(self) -> list[str]:
        return list(CHATTERBOX_LANGUAGE_MAP.keys())

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

    def list_models(self) -> list[ModelInfo]:
        return [
            ModelInfo(
                id="ResembleAI/chatterbox", label="Chatterbox (base)",
                meta={"mode": "multilang"},
            ),
            ModelInfo(
                id="Thomcles/Chatterbox-TTS-French", label="Chatterbox FR fine-tune",
                meta={"mode": "fr_finetune"},
            ),
        ]

    def auto_resolved_keys(self, engine_id: str | None = None) -> list[str]:
        return ["chatterbox_mode"]

    def capabilities(self, engine_id: str | None = None) -> Dict[str, bool | list]:
        return super().capabilities(engine_id)

    def resolve_engine_params(self, engine_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
        mode = self._ENGINE_MODE_MAP.get(engine_id)
        if mode:
            params.setdefault("chatterbox_mode", mode)
        return params

    def supports_ref_for_engine(self, engine_id: str) -> bool:
        return True

    def map_language(self, bcp47: Optional[str]) -> Optional[str]:
        if not bcp47:
            return "fr"
        return CHATTERBOX_LANGUAGE_MAP.get(bcp47, bcp47.split("-")[0])

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
        result = self._run_subprocess(payload)
        meta = {
            "backend_id": self.id,
            "backend_lang": lang,
            "out_path": result.get("out_path") or out_path,
            "duration_s": result.get("duration_s"),
            "retry": bool(result.get("retry")),
        }
        if result.get("logs"):
            meta["runner_logs"] = result["logs"]
        if result.get("stderr"):
            meta["stderr"] = result["stderr"]
        return meta

    def synthesize_chunk(
        self,
        text: str,
        *,
        voice_ref_path: Optional[str] = None,
        lang: Optional[str] = None,
        **params: Any,
    ):
        payload_suffix = _build_runner_suffix(
            voice_ref_path=voice_ref_path,
            lang=lang,
            params=params,
        )
        audio, sr, meta = self._run_subprocess_chunk(
            text, payload_suffix=payload_suffix,
        )
        return audio, sr, meta


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


def _build_runner_suffix(
    *,
    voice_ref_path: Optional[str],
    lang: Optional[str],
    params: dict[str, Any],
) -> dict:
    """Build the extra keys for a chunk-level subprocess payload.

    Returns a dict that does NOT include ``text`` or ``out_wav_path`` —
    these are set by ``_run_subprocess_chunk`` and must not be overridden.
    """
    payload = _build_runner_payload(
        text="__PLACEHOLDER_NOT_OVERWRITTEN__",
        out_wav_path="__PLACEHOLDER_NOT_OVERWRITTEN__",
        voice_ref_path=voice_ref_path,
        lang=lang,
        params=params,
    )
    payload.pop("text", None)
    payload.pop("out_wav_path", None)
    return payload