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


def ensure_default_presets() -> None:
    PRESET_DIR.mkdir(parents=True, exist_ok=True)
    default_name = "default"
    if not _preset_path(default_name).exists():
        data = {
            "min_words_per_chunk": 16,
            "max_words_without_terminator": 35,
            "max_est_seconds_per_chunk": 10.0,
            "tts_model_mode": "fr_finetune",
            "tts_language": "fr-FR",
            "multilang_cfg_weight": 0.5,
            "comma_pause_ms": 200,
            "period_pause_ms": 350,
            "semicolon_pause_ms": 300,
            "colon_pause_ms": 300,
            "dash_pause_ms": 250,
            "newline_pause_ms": 300,
            "exaggeration": 0.5,
            "cfg_weight": 0.6,
            "temperature": 0.5,
            "repetition_penalty": 1.35,
        }
        save_preset(default_name, data)
    stable_name = "stable-long-form"
    if not _preset_path(stable_name).exists():
        data = {
            "comma_pause_ms": 200,
            "period_pause_ms": 350,
            "semicolon_pause_ms": 300,
            "colon_pause_ms": 300,
            "dash_pause_ms": 250,
            "newline_pause_ms": 250,
            "min_words_per_chunk": 16,
            "max_words_without_terminator": 32,
            "max_est_seconds_per_chunk": 9.0,
            "tts_model_mode": "fr_finetune",
            "tts_language": "fr-FR",
            "multilang_cfg_weight": 0.5,
            "exaggeration": 0.45,
            "cfg_weight": 0.75,
            "temperature": 0.35,
            "repetition_penalty": 1.35,
        }
        save_preset(stable_name, data)
    fidelity_name = "voice-fidelity"
    if not _preset_path(fidelity_name).exists():
        data = {
            "comma_pause_ms": 200,
            "period_pause_ms": 350,
            "semicolon_pause_ms": 300,
            "colon_pause_ms": 300,
            "dash_pause_ms": 250,
            "newline_pause_ms": 300,
            "min_words_per_chunk": 16,
            "max_words_without_terminator": 40,
            "max_est_seconds_per_chunk": 12.0,
            "tts_model_mode": "fr_finetune",
            "tts_language": "fr-FR",
            "multilang_cfg_weight": 0.5,
            "exaggeration": 0.5,
            "cfg_weight": 0.6,
            "temperature": 0.5,
            "repetition_penalty": 1.35,
        }
        save_preset(fidelity_name, data)


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
    "ensure_default_presets",
    "list_presets",
    "load_preset",
    "load_state",
    "save_preset",
    "save_state",
]
