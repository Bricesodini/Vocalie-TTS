"""Paths for per-backend virtual environments."""

from __future__ import annotations

import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def venv_dir(engine_id: str) -> Path:
    return ROOT / ".venvs" / engine_id


def _bin_dir(engine_id: str) -> Path:
    if os.name == "nt":
        return venv_dir(engine_id) / "Scripts"
    return venv_dir(engine_id) / "bin"


def python_path(engine_id: str) -> Path:
    if os.name == "nt":
        return _bin_dir(engine_id) / "python.exe"
    return _bin_dir(engine_id) / "python"


def pip_path(engine_id: str) -> Path:
    if os.name == "nt":
        return _bin_dir(engine_id) / "pip.exe"
    return _bin_dir(engine_id) / "pip"
