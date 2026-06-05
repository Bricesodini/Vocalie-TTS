"""Backend registry — auto-discovery via TTSBackend._REGISTRY.

Backends self-register via ``TTSBackend.__init_subclass__`` when they
are imported.  This module imports all backend submodules to trigger
registration, then rebuilds the engine catalog from the registered
backends.

Adding a new backend only requires:
1. Creating a new ``*_backend.py`` that subclasses ``TTSBackend``.
2. Importing it below (one line).
No other list or catalog needs manual editing.
"""

from __future__ import annotations

from typing import Dict, List

from .base import TTSBackend

# Import backend submodules to trigger __init_subclass__ registration.
from .chatterbox_backend import ChatterboxBackend  # noqa: F401
from .cosyvoice_backend import CosyVoiceBackend  # noqa: F401
from .qwen3_backend import Qwen3Backend  # noqa: F401

# Rebuild engine catalog from registered backends.
from .catalog import rebuild_engine_catalog
rebuild_engine_catalog()


def list_backends() -> List[TTSBackend]:
    """Return instantiated backends for all registered classes."""
    return [cls() for cls in TTSBackend._REGISTRY.values()]


def get_backend(engine_id: str) -> TTSBackend | None:
    """Resolve an engine_id to its backend instance.

    Tries exact backend.id match first, then backend.supports_engine_id().
    """
    # Exact match on backend.id
    for cls in TTSBackend._REGISTRY.values():
        if cls.id == engine_id:
            return cls()
    # Fallback: ask each backend if it handles this engine_id
    for cls in TTSBackend._REGISTRY.values():
        instance = cls()
        if instance.supports_engine_id(engine_id):
            return instance
    return None


def available_backend_ids() -> Dict[str, bool]:
    """Return availability status for all registered backends."""
    return {cls.id: cls.is_available() for cls in TTSBackend._REGISTRY.values()}