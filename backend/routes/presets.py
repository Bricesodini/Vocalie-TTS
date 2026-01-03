from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend.schemas.models import (
    PresetCreateRequest,
    PresetListResponse,
    PresetMutationResponse,
    PresetResponse,
    PresetUpdateRequest,
)
from backend.services import preset_service


router = APIRouter(prefix="/v1")


@router.get("/presets", response_model=PresetListResponse)
def list_presets() -> PresetListResponse:
    items = preset_service.list_presets()
    return PresetListResponse(presets=items)


@router.get("/presets/{preset_id}", response_model=PresetResponse)
def get_preset(preset_id: str) -> PresetResponse:
    preset = preset_service.get_preset(preset_id)
    if not preset:
        raise HTTPException(status_code=404, detail="preset_not_found")
    return PresetResponse(**preset)


@router.post("/presets", response_model=PresetMutationResponse)
def create_preset(request: PresetCreateRequest) -> PresetMutationResponse:
    if request.state is None:
        raise HTTPException(status_code=400, detail="preset_state_required")
    try:
        result = preset_service.create_preset(request.id, request.label, request.state)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return PresetMutationResponse(**result)


@router.put("/presets/{preset_id}", response_model=PresetMutationResponse)
def update_preset(preset_id: str, request: PresetUpdateRequest) -> PresetMutationResponse:
    if request.state is None:
        raise HTTPException(status_code=400, detail="preset_state_required")
    try:
        result = preset_service.update_preset(preset_id, request.label, request.state)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return PresetMutationResponse(**result)


@router.delete("/presets/{preset_id}", response_model=PresetMutationResponse)
def delete_preset(preset_id: str) -> PresetMutationResponse:
    result = preset_service.delete_preset(preset_id)
    return PresetMutationResponse(**result)
