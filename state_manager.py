"""Persistence helpers for user state and presets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from output_paths import sanitize_filename

BASE_DIR = Path(__file__).resolve().parent
STATE_DIR = BASE_DIR / ".state"
STATE_FILE = STATE_DIR / "state.json"
PRESET_DIR = BASE_DIR / "presets"


def _read_json(path: Path) -> Dict:
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError:
        return {}


def load_state() -> Dict:
    return _read_json(STATE_FILE)


def save_state(data: Dict) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    with STATE_FILE.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)


def _preset_path(name: str) -> Path:
    slug = sanitize_filename(name)
    if not slug:
        raise ValueError("Nom de preset invalide.")
    return PRESET_DIR / f"{slug}.json"


def list_presets() -> List[str]:
    PRESET_DIR.mkdir(parents=True, exist_ok=True)
    names = []
    for preset_file in PRESET_DIR.glob("*.json"):
        names.append(preset_file.stem)
    return sorted(names)


def load_preset(name: str) -> Dict:
    path = _preset_path(name)
    return _read_json(path)


def save_preset(name: str, data: Dict) -> str:
    PRESET_DIR.mkdir(parents=True, exist_ok=True)
    path = _preset_path(name)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)
    return path.stem


def delete_preset(name: str) -> None:
    path = _preset_path(name)
    if path.exists():
        path.unlink()


__all__ = [
    "delete_preset",
    "list_presets",
    "load_preset",
    "load_state",
    "save_preset",
    "save_state",
]
