"""Bark backend wrapper (optional dependency)."""

from __future__ import annotations

from typing import Any, Dict, Optional

from .base import BackendUnavailableError, ParamSpec, TTSBackend


class BarkBackend(TTSBackend):
    id = "bark"
    display_name = "Bark (creative)"
    supports_ref_audio = False
    uses_internal_voices = True

    @classmethod
    def is_available(cls) -> bool:
        from backend_install.status import backend_status
        return backend_status("bark").get("installed", False)

    @classmethod
    def unavailable_reason(cls) -> str | None:
        from backend_install.status import backend_status
        return backend_status("bark").get("reason")

    def supported_languages(self) -> list[str]:
        return ["fr-FR", "en-US", "es-ES", "de-DE", "it-IT", "pt-PT", "nl-NL"]

    def default_language(self) -> str:
        return "fr-FR"

    def params_schema(self) -> dict[str, ParamSpec]:
        return {
            "voice_id": ParamSpec(
                key="voice_id",
                type="select",
                default=None,
                choices=[],
                label="Voix Bark",
                visible_if={"uses_internal_voices": True, "voice_count_min": 2},
            ),
            "temperature": ParamSpec(
                key="temperature",
                type="float",
                default=0.7,
                min=0.0,
                max=1.0,
                step=0.05,
                label="Température",
            ),
            "top_k": ParamSpec(
                key="top_k",
                type="int",
                default=50,
                min=0,
                max=200,
                step=1,
                label="Top-K",
            ),
            "top_p": ParamSpec(
                key="top_p",
                type="float",
                default=0.9,
                min=0.0,
                max=1.0,
                step=0.05,
                label="Top-P",
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
        raise BackendUnavailableError("Bark backend is not wired yet.")
