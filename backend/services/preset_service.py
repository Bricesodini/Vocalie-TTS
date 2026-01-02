from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from backend.config import PRESETS_DIR


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
                name = str(data.get("name") or data.get("id") or preset_id)
        except json.JSONDecodeError:
            name = preset_id
        presets.append({"id": preset_id, "name": name, "updated_at": updated_at})
    return presets


def get_preset(preset_id: str) -> Optional[Dict[str, Any]]:
    path = _preset_path(preset_id)
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    updated_at = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    name = None
    if isinstance(data, dict):
        name = data.get("name") or data.get("id")
    return {
        "id": str(preset_id),
        "name": str(name) if name is not None else None,
        "data": data,
        "updated_at": updated_at,
    }


def create_preset(preset_id: str, name: Optional[str], data: Dict[str, Any]) -> Dict[str, Any]:
    payload = dict(data)
    if name:
        payload["name"] = name
    if "id" not in payload:
        payload["id"] = preset_id
    path = _preset_path(preset_id)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return {"id": preset_id, "status": "created"}


def update_preset(preset_id: str, name: Optional[str], data: Dict[str, Any]) -> Dict[str, Any]:
    payload = dict(data)
    if name is not None:
        payload["name"] = name
    if "id" not in payload:
        payload["id"] = preset_id
    path = _preset_path(preset_id)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return {"id": preset_id, "status": "updated"}


def delete_preset(preset_id: str) -> Dict[str, Any]:
    path = _preset_path(preset_id)
    if path.exists():
        path.unlink()
    return {"id": preset_id, "status": "deleted"}
