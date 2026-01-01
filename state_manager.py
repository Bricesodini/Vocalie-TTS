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


def migrate_state(data: Dict) -> Dict:
    if not isinstance(data, dict):
        return {}
    if "direction_enabled" not in data:
        data["direction_enabled"] = True
    if "direction_source" not in data:
        data["direction_source"] = "final"
    if "inter_chunk_gap_ms" not in data:
        data["inter_chunk_gap_ms"] = 120
    if "inter_chunk_gap_ms" not in data:
        data["inter_chunk_gap_ms"] = 120
    engines = data.get("engines")
    if not isinstance(engines, dict):
        engines = {}
    engine_id = data.get("last_tts_engine") or "chatterbox"
    engine_cfg = engines.get(engine_id)
    if not isinstance(engine_cfg, dict):
        engine_cfg = {}
    params = engine_cfg.get("params")
    if not isinstance(params, dict):
        params = {}
    if "last_chatterbox_mode" in data and "chatterbox_mode" not in params:
        params["chatterbox_mode"] = data.get("last_chatterbox_mode")
    if "last_multilang_cfg_weight" in data and "multilang_cfg_weight" not in params:
        params["multilang_cfg_weight"] = data.get("last_multilang_cfg_weight")
    language = engine_cfg.get("language") or data.get("last_tts_language") or "fr-FR"
    engine_cfg["language"] = language
    engine_cfg["params"] = params
    engines[engine_id] = engine_cfg
    data["engines"] = engines
    return data


def load_state() -> Dict:
    return migrate_state(_read_json(STATE_FILE))


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
            "tts_engine": "chatterbox",
            "include_model_name": False,
            "direction_enabled": True,
            "direction_source": "final",
            "inter_chunk_gap_ms": 120,
            "engines": {
                "chatterbox": {
                    "language": "fr-FR",
                    "params": {
                        "chatterbox_mode": "fr_finetune",
                        "multilang_cfg_weight": 0.5,
                        "exaggeration": 0.5,
                        "cfg_weight": 0.6,
                        "temperature": 0.5,
                        "repetition_penalty": 1.35,
                    },
                }
            },
        }
        save_preset(default_name, data)
    stable_name = "stable-long-form"
    if not _preset_path(stable_name).exists():
        data = {
            "tts_engine": "chatterbox",
            "include_model_name": False,
            "direction_enabled": True,
            "direction_source": "final",
            "inter_chunk_gap_ms": 120,
            "engines": {
                "chatterbox": {
                    "language": "fr-FR",
                    "params": {
                        "chatterbox_mode": "fr_finetune",
                        "multilang_cfg_weight": 0.5,
                        "exaggeration": 0.45,
                        "cfg_weight": 0.75,
                        "temperature": 0.35,
                        "repetition_penalty": 1.35,
                    },
                }
            },
        }
        save_preset(stable_name, data)
    fidelity_name = "voice-fidelity"
    if not _preset_path(fidelity_name).exists():
        data = {
            "tts_engine": "chatterbox",
            "include_model_name": False,
            "direction_enabled": True,
            "direction_source": "final",
            "inter_chunk_gap_ms": 120,
            "engines": {
                "chatterbox": {
                    "language": "fr-FR",
                    "params": {
                        "chatterbox_mode": "fr_finetune",
                        "multilang_cfg_weight": 0.5,
                        "exaggeration": 0.5,
                        "cfg_weight": 0.6,
                        "temperature": 0.5,
                        "repetition_penalty": 1.35,
                    },
                }
            },
        }
        save_preset(fidelity_name, data)


def load_preset(name: str) -> Dict:
    path = _preset_path(name)
    data = _read_json(path)
    if not data:
        return data
    if "tts_engine" not in data:
        data["tts_engine"] = "chatterbox"
    if "include_model_name" not in data:
        data["include_model_name"] = False
    if "direction_enabled" not in data:
        data["direction_enabled"] = True
    if "direction_source" not in data:
        data["direction_source"] = "final"
    engines = data.get("engines")
    if not isinstance(engines, dict):
        engines = {}
    if "engines" not in data:
        legacy_params = {}
        if "tts_model_mode" in data:
            legacy_params["chatterbox_mode"] = data.get("tts_model_mode")
        if "multilang_cfg_weight" in data:
            legacy_params["multilang_cfg_weight"] = data.get("multilang_cfg_weight")
        if "exaggeration" in data:
            legacy_params["exaggeration"] = data.get("exaggeration")
        if "cfg_weight" in data:
            legacy_params["cfg_weight"] = data.get("cfg_weight")
        if "temperature" in data:
            legacy_params["temperature"] = data.get("temperature")
        if "repetition_penalty" in data:
            legacy_params["repetition_penalty"] = data.get("repetition_penalty")
        engines["chatterbox"] = {
            "language": data.get("tts_language") or "fr-FR",
            "params": legacy_params,
        }
    for engine_id, engine_cfg in engines.items():
        if not isinstance(engine_cfg, dict):
            engines[engine_id] = {"language": "fr-FR", "params": {}}
            continue
        if not engine_cfg.get("language"):
            engine_cfg["language"] = "fr-FR"
        if "params" not in engine_cfg or not isinstance(engine_cfg.get("params"), dict):
            engine_cfg["params"] = {}
        engines[engine_id] = engine_cfg
    data["engines"] = engines
    return data


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
