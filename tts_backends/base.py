"""Backend interface for TTS engines.

Each backend is self-contained: it knows its own capabilities, defaults,
and how to resolve engine-specific parameters.  The caller never needs
if/elif chains to distinguish backends.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class ModelInfo:
    """Describes a model available within a backend."""
    id: str
    label: str
    version: str | None = None
    meta: Dict[str, Any] | None = None


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
    """Abstract base class for TTS backends with auto-registration.

    Subclasses that define ``id`` are automatically registered in
    ``TTSBackend._REGISTRY`` via ``__init_subclass__``.  Import a backend
    module to trigger registration — no manual list maintenance needed.
    """

    _REGISTRY: Dict[str, type[TTSBackend]] = {}

    id: str
    display_name: str
    supports_ref_audio: bool = False
    uses_internal_voices: bool = False
    supports_inter_chunk_gap: bool = False

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        # Only register concrete subclasses that declare an ``id``.
        # ABC itself and intermediate mixins are skipped.
        if getattr(cls, "id", None) and not getattr(cls, "__abstractmethods__", None):
            TTSBackend._REGISTRY[cls.id] = cls

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

    @classmethod
    def engine_variants(cls) -> List[Dict[str, str]]:
        """Return engine variant definitions for the catalog.

        Each dict must contain: ``id``, ``label``.
        Default: single variant with ``id=cls.id``, ``label=cls.display_name``.
        Override in subclasses that expose multiple engine variants
        (e.g. ``chatterbox_native``, ``chatterbox_finetune_fr``).
        """
        if hasattr(cls, 'id') and cls.id:
            return [{"id": cls.id, "label": cls.display_name}]
        return []

    def supports_ref_for_engine(self, engine_id: str) -> bool:
        """Does this engine variant require a reference voice?

        Override when an engine_id under this backend has different
        ref requirements from the backend default.
        """
        return self.supports_ref_audio

    def supports_engine_id(self, engine_id: str) -> bool:
        """Return True if this backend handles the given catalogue engine_id.

        Default: the engine_id must match ``self.id`` OR start with
        ``{self.id}_`` (e.g. "chatterbox_native" → "chatterbox").
        """
        if engine_id == self.id:
            return True
        if engine_id.startswith(f"{self.id}_"):
            return True
        return False

    def resolve_engine_params(self, engine_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Return engine_params dict with backend-specific defaults applied.

        Subclasses should override to inject engine-id-specific defaults
        (e.g. ``chatterbox_mode`` from the engine_id).
        """
        return dict(params)

    def list_models(self) -> List[ModelInfo]:
        """List downloadable/switchable model variants for this backend.

        Override in subclasses that expose multiple model weights
        (e.g. Qwen3 has CustomVoice, VoiceDesign, Base models).
        """
        return []

    def params_schema(self) -> Dict[str, ParamSpec]:
        return {}

    def capabilities(self, engine_id: str | None = None) -> Dict[str, bool | list]:
        """Backend capabilities including keys that are auto-resolved.

        ``auto_resolved_keys`` lists param keys whose values are determined
        by the engine_id — the frontend should hide them from the user.
        """
        ref = self.supports_ref_for_engine(engine_id) if engine_id else self.supports_ref_audio
        return {
            "uses_voice_reference": bool(ref),
            "uses_internal_voices": bool(self.uses_internal_voices),
            "auto_resolved_keys": self.auto_resolved_keys(engine_id),
        }

    def auto_resolved_keys(self, engine_id: str | None = None) -> list[str]:
        """Param keys whose values are set by resolve_engine_params().

        Override in subclasses when specific engine IDs auto-resolve params.
        """
        return []
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


def coerce_bool(value: Any, default: bool) -> bool:
    """Coerce a value to bool, with a fallback default.

    Handles ``None``, ``bool``, ``int``, ``float``, and common
    string representations ("1", "true", "yes", "0", "false", "no").
    """
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return default


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
