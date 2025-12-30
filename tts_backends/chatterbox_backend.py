"""Chatterbox TTS backend wrapper."""

from __future__ import annotations

from typing import Any, Dict, Optional

from tts_engine import LANGUAGE_MAP, TTSEngine

from .base import ParamSpec, TTSBackend


class ChatterboxBackend(TTSBackend):
    id = "chatterbox"
    display_name = "Chatterbox (stable long-form)"
    supports_ref_audio = True
    uses_internal_voices = False

    @classmethod
    def is_available(cls) -> bool:
        try:
            from chatterbox.tts import ChatterboxTTS  # noqa: F401
        except Exception:
            return False
        return True

    @classmethod
    def unavailable_reason(cls) -> str | None:
        return "Chatterbox backend requires the chatterbox package."

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
        engine = TTSEngine()
        tts_model_mode = params.pop("tts_model_mode", params.pop("chatterbox_mode", "fr_finetune"))
        multilang_cfg_weight = params.pop("multilang_cfg_weight", 0.5)
        _, _, meta = engine.generate_longform(
            script=script,
            audio_prompt_path=voice_ref_path,
            out_path=out_path,
            tts_model_mode=tts_model_mode,
            tts_language=lang or "fr-FR",
            multilang_cfg_weight=multilang_cfg_weight,
            **params,
        )
        meta.setdefault("warnings", [])
        meta["backend_id"] = self.id
        meta["backend_lang"] = lang
        return meta

    def synthesize_chunk(
        self,
        text: str,
        *,
        voice_ref_path: Optional[str] = None,
        lang: Optional[str] = None,
        **params: Any,
    ):
        engine = TTSEngine()
        tts_model_mode = params.get("tts_model_mode", params.get("chatterbox_mode", "fr_finetune"))
        multilang_cfg_weight = params.get("multilang_cfg_weight", 0.5)
        requested_language = engine._resolve_language(tts_model_mode, lang or "fr-FR")
        backend_language = requested_language
        effective_cfg = float(params.get("cfg_weight", 0.6))
        if tts_model_mode == "multilang":
            backend_language = engine._map_multilang_language(requested_language)
            effective_cfg = float(multilang_cfg_weight)
        tts = engine._get_backend(tts_model_mode)
        audio, _ = engine._synthesize_text(
            tts,
            text,
            voice_ref_path,
            float(params.get("exaggeration", 0.5)),
            float(effective_cfg),
            float(params.get("temperature", 0.5)),
            float(params.get("repetition_penalty", 1.35)),
            backend_language,
            "language_id" if tts_model_mode == "multilang" else None,
            False,
        )
        sr = int(engine.sample_rate or tts.sr)
        retried = False
        duration = len(audio) / sr if sr else 0.0
        if len(text) > 80 and duration < 1.2:
            retried = True
            retry_audio, _ = engine._synthesize_text(
                tts,
                text,
                voice_ref_path,
                float(params.get("exaggeration", 0.5)),
                min(1.5, float(effective_cfg) + 0.05),
                max(0.2, float(params.get("temperature", 0.5)) - 0.05),
                float(params.get("repetition_penalty", 1.35)),
                backend_language,
                "language_id" if tts_model_mode == "multilang" else None,
                False,
            )
            retry_duration = len(retry_audio) / sr if sr else 0.0
            if retry_duration > duration:
                audio = retry_audio
        return audio, sr, {"retry": retried}
