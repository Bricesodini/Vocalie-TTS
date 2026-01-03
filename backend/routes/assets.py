from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from backend.schemas.models import AssetMetaResponse
from backend.services import asset_service


router = APIRouter(prefix="/v1")


@router.get("/assets/{asset_id}")
def get_asset(asset_id: str):
    meta = asset_service.get_asset_meta(asset_id)
    if not meta:
        raise HTTPException(status_code=404, detail="asset_not_found")
    path = asset_service.resolve_asset_path(meta)
    if not path or not path.exists():
        raise HTTPException(status_code=404, detail="asset_missing")
    return FileResponse(path, media_type="audio/wav", filename=path.name)


@router.get("/assets/{asset_id}/meta", response_model=AssetMetaResponse)
def get_asset_meta(asset_id: str) -> AssetMetaResponse:
    meta = asset_service.get_asset_meta(asset_id)
    if not meta:
        raise HTTPException(status_code=404, detail="asset_not_found")
    return AssetMetaResponse(**meta)
