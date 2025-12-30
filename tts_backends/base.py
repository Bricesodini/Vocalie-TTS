"""Backend interface for TTS engines."""

from __future__ import annotations

from dataclasses import dataclass
from abc import ABC, abstractmethod
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class VoiceInfo:
    id: str
    label: str
    lang_codes: List[str] | None = None
    installed: bool = True
    meta: Dict[str, Any] | None = None


@dataclass(frozen=True)
class ParamSpec:
    key: str
    type: str
    default: Any
    min: float | None = None
    max: float | None = None
    step: float | None = None
    choices: List[Any] | None = None
    label: str | None = None
    help: str | None = None
    visible_if: Dict[str, Any] | None = None
    serialize_scope: str = "engine"


class TTSBackend(ABC):
    id: str
    display_name: str
    supports_ref_audio: bool = False
    uses_internal_voices: bool = False

    @classmethod
    def is_available(cls) -> bool:
        return True

    @classmethod
    def unavailable_reason(cls) -> str | None:
        return None

    def validate_config(self, cfg: Dict[str, Any]) -> List[str]:
        return []

    def supported_languages(self) -> List[str]:
        return []

    def default_language(self) -> str:
        return pick_default_language(self.supported_languages(), None)

    def list_voices(self) -> List[VoiceInfo]:
        return []

    def params_schema(self) -> Dict[str, ParamSpec]:
        return {}

    def capabilities(self) -> Dict[str, bool]:
        return {
            "uses_voice_reference": bool(self.supports_ref_audio),
            "uses_internal_voices": bool(self.uses_internal_voices),
        }
    @property
    def supports_multilang(self) -> bool:
        return len(self.supported_languages()) > 1

    @property
    def supports_voice_selector(self) -> bool:
        return self.uses_internal_voices and len(self.list_voices()) > 1

    def map_language(self, bcp47: Optional[str]) -> Optional[str]:
        if not self.supported_languages():
            return None
        if not bcp47:
            return None
        return bcp47

    @abstractmethod
    def synthesize(
        self,
        script: str,
        out_path: str,
        voice_ref_path: Optional[str] = None,
        lang: Optional[str] = None,
        **params: Any,
    ) -> Dict[str, Any]:
        raise NotImplementedError

    def synthesize_chunk(
        self,
        text: str,
        *,
        voice_ref_path: Optional[str] = None,
        lang: Optional[str] = None,
        **params: Any,
    ):
        import numpy as np
        import soundfile as sf

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            meta = self.synthesize(
                script=text,
                out_path=tmp_path,
                voice_ref_path=voice_ref_path,
                lang=lang,
                **params,
            )
            audio, sr = sf.read(tmp_path, dtype="float32")
        finally:
            try:
                Path(tmp_path).unlink(missing_ok=True)
            except OSError:
                pass
        return np.asarray(audio, dtype=np.float32), int(sr), meta or {}


class BackendUnavailableError(RuntimeError):
    """Raised when a backend is selected but not available or not wired."""


def pick_default_language(supported_languages: list[str], default_language: str | None = None) -> str:
    if "fr-FR" in supported_languages:
        return "fr-FR"
    if default_language:
        return default_language
    return supported_languages[0] if supported_languages else "fr-FR"


def coerce_language(requested: str | None, supported_languages: list[str], default_language: str | None = None):
    if requested and requested in supported_languages:
        return requested, False
    return pick_default_language(supported_languages, default_language), True


def validate_param_schema(schema: Dict[str, ParamSpec]) -> List[str]:
    errors: List[str] = []
    for key, spec in schema.items():
        if spec.key != key:
            errors.append(f"{key}: key mismatch ({spec.key})")
        if spec.type not in {"float", "int", "bool", "str", "choice", "select"}:
            errors.append(f"{key}: invalid type {spec.type}")
        if spec.type in {"choice", "select"} and not spec.choices:
            if spec.default is not None:
                errors.append(f"{key}: missing choices")
        if spec.type in {"float", "int"}:
            if spec.min is None or spec.max is None:
                errors.append(f"{key}: min/max required for numeric")
            if spec.step is None:
                errors.append(f"{key}: step required for numeric")
        if spec.serialize_scope not in {"global", "engine"}:
            errors.append(f"{key}: invalid serialize_scope {spec.serialize_scope}")
    return errors
