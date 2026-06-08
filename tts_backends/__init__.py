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

import threading
import time
from typing import Dict, List, Tuple

from .base import TTSBackend

# Import backend submodules to trigger __init_subclass__ registration.
from .chatterbox_backend import ChatterboxBackend  # noqa: F401
from .cosyvoice_backend import CosyVoiceBackend  # noqa: F401
from .qwen3_backend import Qwen3Backend  # noqa: F401

# Rebuild engine catalog from registered backends.
from .catalog import rebuild_engine_catalog
rebuild_engine_catalog()


# ---------------------------------------------------------------------------
# availability cache
# ---------------------------------------------------------------------------
# Each backend's `is_available()` may spawn a subprocess (e.g. spawning
# the venv python to run an import probe), so calling it on every
# /v1/health hit added 3–8s to a route the menu-bar app polls every
# 5s. Cache the result for 30s — long enough that the menu-bar app's
# 5s polling cadence is always a cache hit, short enough that a
# fresh install/uninstall is reflected within a few seconds.
_AVAILABILITY_TTL_S = 30.0
_availability_cache: Dict[str, Tuple[float, Dict[str, bool]]] = {}
_availability_lock = threading.Lock()


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
    """Return availability status for all registered backends.

    Cached for ``_AVAILABILITY_TTL_S`` seconds: calling this from the
    hot path (e.g. /v1/health polled every few seconds) would otherwise
    spawn one subprocess per backend per call.
    """
    now = time.monotonic()
    with _availability_lock:
        cached = _availability_cache.get("all")
        if cached is not None:
            ts, value = cached
            if now - ts < _AVAILABILITY_TTL_S:
                return value
        value = {cls.id: cls.is_available() for cls in TTSBackend._REGISTRY.values()}
        _availability_cache["all"] = (now, value)
        return value


def invalidate_availability_cache() -> None:
    """Drop the cached availability map. Useful after an install/uninstall
    so the next /v1/health reflects the new state without waiting for TTL."""
    with _availability_lock:
        _availability_cache.clear()