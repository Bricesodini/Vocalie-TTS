from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class HealthResponse(BaseModel):
    status: str
    uptime_s: int
    timestamp: datetime


class InfoResponse(BaseModel):
    name: str
    version: str
    commit: Optional[str] = None
    python: str
    os: str
    work_dir: str
    output_dir: str
    presets_dir: str


class CapabilitiesResponse(BaseModel):
    engines: List[str]
    features: Dict[str, Any]


class EngineInfo(BaseModel):
    id: str
    label: str
    available: bool
    supports_ref: bool


class EnginesResponse(BaseModel):
    engines: List[EngineInfo]


class VoiceInfo(BaseModel):
    id: str
    label: str
    language: Optional[str] = None
    gender: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None


class VoicesResponse(BaseModel):
    engine: str
    voices: List[VoiceInfo]


class ModelInfo(BaseModel):
    id: str
    label: str
    version: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None


class ModelsResponse(BaseModel):
    engine: str
    models: List[ModelInfo]


class PresetListItem(BaseModel):
    id: str
    name: Optional[str] = None
    updated_at: Optional[datetime] = None


class PresetListResponse(BaseModel):
    presets: List[PresetListItem]


class PresetResponse(BaseModel):
    id: str
    name: Optional[str] = None
    data: Dict[str, Any]
    updated_at: Optional[datetime] = None


class PresetCreateRequest(BaseModel):
    id: str
    name: Optional[str] = None
    data: Dict[str, Any]

    model_config = ConfigDict(extra="allow")


class PresetUpdateRequest(BaseModel):
    name: Optional[str] = None
    data: Dict[str, Any]

    model_config = ConfigDict(extra="allow")


class PresetMutationResponse(BaseModel):
    id: str
    status: str


class DirectionOptions(BaseModel):
    enabled: bool = False
    chunk_marker: str = "[[CHUNK]]"


class ExportOptions(BaseModel):
    format: Literal["wav"] = "wav"
    filename: Optional[str] = None
    include_timestamp: bool = True
    include_model: bool = False


class EditOptions(BaseModel):
    enabled: bool = False
    trim_silence: bool = True
    normalize: bool = True
    target_dbfs: float = -1.0


class TTSJobRequest(BaseModel):
    text: str
    engine: str
    voice: Optional[str] = None
    model: Optional[str] = None
    language: Optional[str] = None
    preset_id: Optional[str] = None
    direction: Optional[DirectionOptions] = None
    options: Optional[Dict[str, Any]] = None
    export: Optional[ExportOptions] = None
    editing: Optional[EditOptions] = None


class JobCreateResponse(BaseModel):
    job_id: str
    status: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    progress: float
    created_at: datetime
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    asset_id: Optional[str] = None
    error: Optional[str] = None


class JobCancelResponse(BaseModel):
    job_id: str
    status: str


class AssetMetaResponse(BaseModel):
    asset_id: str
    file_name: str
    relative_path: Optional[str] = None
    size_bytes: int
    duration_s: Optional[float] = None
    sample_rate: Optional[int] = None
    engine: Optional[str] = None
    voice: Optional[str] = None
    model: Optional[str] = None
    created_at: Optional[datetime] = None
    job_id: Optional[str] = None
