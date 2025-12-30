"""Paths for per-backend virtual environments."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def venv_dir(engine_id: str) -> Path:
    return ROOT / ".venvs" / engine_id


def python_path(engine_id: str) -> Path:
    return venv_dir(engine_id) / "bin" / "python"


def pip_path(engine_id: str) -> Path:
    return venv_dir(engine_id) / "bin" / "pip"
