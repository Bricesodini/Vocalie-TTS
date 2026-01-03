"""Backend registry."""

from __future__ import annotations

from typing import Dict, List

from .base import TTSBackend
from .bark_backend import BarkBackend
from .chatterbox_backend import ChatterboxBackend
from .piper_backend import PiperBackend
from .xtts_backend import XTTSBackend


BACKENDS: List[type[TTSBackend]] = [
    ChatterboxBackend,
    XTTSBackend,
    PiperBackend,
    BarkBackend,
]


def list_backends() -> List[TTSBackend]:
    return [backend_cls() for backend_cls in BACKENDS]


def get_backend(backend_id: str) -> TTSBackend | None:
    for backend_cls in BACKENDS:
        if backend_cls.id == backend_id:
            return backend_cls()
    return None


def available_backend_ids() -> Dict[str, bool]:
    return {backend_cls.id: backend_cls.is_available() for backend_cls in BACKENDS}
