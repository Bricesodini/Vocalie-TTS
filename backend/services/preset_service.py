from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from backend.config import PRESETS_DIR
from backend.schemas.models import UIState
from pydantic import ValidationError


PRESET_SUFFIX = ".json"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _preset_path(preset_id: str) -> Path:
    safe_id = str(preset_id).strip()
    return PRESETS_DIR / f"{safe_id}{PRESET_SUFFIX}"


def list_presets() -> List[Dict[str, Any]]:
    presets: List[Dict[str, Any]] = []
    for path in sorted(PRESETS_DIR.glob(f"*{PRESET_SUFFIX}")):
        if not path.is_file():
            continue
        preset_id = path.stem
        updated_at = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        name = preset_id
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                name = str(data.get("label") or data.get("name") or data.get("id") or preset_id)
        except json.JSONDecodeError:
            name = preset_id
        presets.append({"id": preset_id, "label": name, "updated_at": updated_at})
    return presets


def _legacy_to_ui_state(data: Dict[str, Any], preset_id: str) -> Dict[str, Any]:
    legacy_engine = str(data.get("tts_engine") or data.get("engine_id") or data.get("engine") or "")
    engine_map = {
        "chatterbox": "chatterbox_finetune_fr",
        "xtts": "xtts_v2",
    }
    engine_id = engine_map.get(legacy_engine, legacy_engine)
    engines = data.get("engines") if isinstance(data.get("engines"), dict) else {}
    engine_cfg = engines.get(legacy_engine) if isinstance(legacy_engine, str) else None
    if not isinstance(engine_cfg, dict):
        engine_cfg = {}
    params = engine_cfg.get("params") if isinstance(engine_cfg.get("params"), dict) else {}
    voice_id = engine_cfg.get("voice_id")
    ref_name = data.get("ref_name")
    if voice_id is None and ref_name:
        voice_id = ref_name
    gap_ms = data.get("inter_chunk_gap_ms")
    if gap_ms is None:
        gap_ms = data.get("chatterbox_gap_ms") or 0
    post_enabled = bool(data.get("post_processing_enabled"))
    return {
        "preset_id": preset_id,
        "preparation": {},
        "direction": {},
        "engine": {
            "engine_id": engine_id,
            "voice_id": voice_id,
            "params": params,
            "chatterbox_gap_ms": int(gap_ms or 0),
        },
        "post": {
            "edit_enabled": post_enabled,
            "trim_enabled": bool(data.get("trim_enabled", False)),
            "normalize_enabled": bool(data.get("normalize_enabled", False)),
            "target_dbfs": float(data.get("target_dbfs") or -1.0),
        },
    }


def _coerce_ui_state(payload: Dict[str, Any], preset_id: str) -> UIState:
    if "state" in payload and isinstance(payload["state"], dict):
        data = payload["state"]
    elif "data" in payload and isinstance(payload["data"], dict):
        data = payload["data"]
    else:
        data = payload
    if not isinstance(data, dict):
        raise ValueError("preset_payload_invalid")
    if "engine" not in data and "preparation" not in data and "direction" not in data:
        data = _legacy_to_ui_state(data, preset_id)
    if "preset_id" not in data:
        data["preset_id"] = preset_id
    try:
        return UIState.model_validate(data)
    except ValidationError as exc:
        raise ValueError("preset_state_invalid") from exc


def get_preset(preset_id: str) -> Optional[Dict[str, Any]]:
    path = _preset_path(preset_id)
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    updated_at = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    label = None
    if isinstance(data, dict):
        label = data.get("label") or data.get("name") or data.get("id")
    state = _coerce_ui_state(data if isinstance(data, dict) else {}, preset_id)
    return {
        "id": str(preset_id),
        "label": str(label) if label is not None else None,
        "state": state.model_dump(),
        "updated_at": updated_at,
    }


def create_preset(preset_id: str, label: Optional[str], data: Dict[str, Any]) -> Dict[str, Any]:
    state = _coerce_ui_state(data, preset_id)
    payload = {"id": preset_id, "label": label, "state": state.model_dump()}
    path = _preset_path(preset_id)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return {"id": preset_id, "status": "created"}


def update_preset(preset_id: str, label: Optional[str], data: Dict[str, Any]) -> Dict[str, Any]:
    state = _coerce_ui_state(data, preset_id)
    payload = {"id": preset_id, "label": label, "state": state.model_dump()}
    path = _preset_path(preset_id)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return {"id": preset_id, "status": "updated"}


def delete_preset(preset_id: str) -> Dict[str, Any]:
    path = _preset_path(preset_id)
    if path.exists():
        path.unlink()
    return {"id": preset_id, "status": "deleted"}
