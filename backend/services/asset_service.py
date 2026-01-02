from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from backend.config import ASSETS_META_DIR, OUTPUT_DIR


META_SUFFIX = ".json"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _meta_path(asset_id: str) -> Path:
    safe_id = str(asset_id)
    return ASSETS_META_DIR / f"{safe_id}{META_SUFFIX}"


def write_asset_meta(asset_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    meta = dict(payload)
    meta["asset_id"] = asset_id
    meta.setdefault("created_at", _utc_now().isoformat(timespec="seconds"))
    path = _meta_path(asset_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return meta


def get_asset_meta(asset_id: str) -> Optional[Dict[str, Any]]:
    path = _meta_path(asset_id)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_asset_path(meta: Dict[str, Any]) -> Optional[Path]:
    rel = meta.get("relative_path")
    if rel:
        candidate = OUTPUT_DIR / rel
        if candidate.exists():
            return candidate
    file_name = meta.get("file_name")
    if file_name:
        candidate = OUTPUT_DIR / file_name
        if candidate.exists():
            return candidate
    return None
