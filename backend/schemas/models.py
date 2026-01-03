from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class HealthResponse(BaseModel):
    status: str
    api_version: str
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
    label: Optional[str] = Field(default=None, alias="name")
    updated_at: Optional[datetime] = None

    model_config = ConfigDict(populate_by_name=True)


class PresetListResponse(BaseModel):
    presets: List[PresetListItem]


class UIStatePreparation(BaseModel):
    text_raw: str = ""
    text_adjusted: str = ""
    text_interpreted: str = ""
    glossary_enabled: bool = False
    glossary_profile: Optional[str] = None
    glossary_options: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="allow")


class UIStateChunkRange(BaseModel):
    start: int
    end: int


class UIStateChunkPreview(BaseModel):
    index: int
    text: str
    est_duration_s: Optional[float] = None
    word_count: Optional[int] = None


class UIStateDirection(BaseModel):
    snapshot_text: str = ""
    chunk_markers: List[int] = Field(default_factory=list)
    chunk_ranges: List[UIStateChunkRange] = Field(default_factory=list)
    chunks_preview: List[UIStateChunkPreview] = Field(default_factory=list)

    model_config = ConfigDict(extra="allow")


class UIStateEngine(BaseModel):
    engine_id: str = ""
    voice_id: Optional[str] = None
    params: Dict[str, Any] = Field(default_factory=dict)
    chatterbox_gap_ms: int = 0

    model_config = ConfigDict(extra="allow")


class UIStatePost(BaseModel):
    edit_enabled: bool = False
    trim_enabled: bool = False
    normalize_enabled: bool = False
    target_dbfs: float = -1.0

    model_config = ConfigDict(extra="allow")


class UIState(BaseModel):
    preset_id: Optional[str] = None
    preparation: UIStatePreparation = Field(default_factory=UIStatePreparation)
    direction: UIStateDirection = Field(default_factory=UIStateDirection)
    engine: UIStateEngine = Field(default_factory=UIStateEngine)
    post: UIStatePost = Field(default_factory=UIStatePost)

    model_config = ConfigDict(extra="allow")


class PresetResponse(BaseModel):
    id: str
    label: Optional[str] = Field(default=None, alias="name")
    state: UIState
    updated_at: Optional[datetime] = None

    model_config = ConfigDict(populate_by_name=True)


class PresetCreateRequest(BaseModel):
    id: str
    label: Optional[str] = Field(default=None, alias="name")
    state: Optional[Dict[str, Any]] = Field(default=None, alias="data")

    model_config = ConfigDict(extra="allow", populate_by_name=True)


class PresetUpdateRequest(BaseModel):
    label: Optional[str] = Field(default=None, alias="name")
    state: Optional[Dict[str, Any]] = Field(default=None, alias="data")

    model_config = ConfigDict(extra="allow", populate_by_name=True)


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


class PrepAdjustRequest(BaseModel):
    text_raw: str = ""
    options: Dict[str, Any] = Field(default_factory=dict)


class PrepAdjustResponse(BaseModel):
    text_adjusted: str


class PrepInterpretRequest(BaseModel):
    text_adjusted: Optional[str] = None
    text_raw: Optional[str] = None
    glossary_enabled: bool = False
    glossary_profile: Optional[str] = None
    glossary_options: Dict[str, Any] = Field(default_factory=dict)


class PrepInterpretResponse(BaseModel):
    text_interpreted: str
    applied_rules_summary: Optional[List[str]] = None


class ChunkSnapshotRequest(BaseModel):
    text_interpreted: Optional[str] = None
    text_adjusted: Optional[str] = None
    mode: Optional[str] = None
    options: Dict[str, Any] = Field(default_factory=dict)


class ChunkSnapshotResponse(BaseModel):
    snapshot_text: str
    snapshot_id: Optional[str] = None


class ChunkPreviewRequest(BaseModel):
    snapshot_text: str
    markers: Optional[List[int]] = None
    ranges: Optional[List[UIStateChunkRange]] = None
    engine_id: Optional[str] = None
    params: Dict[str, Any] = Field(default_factory=dict)


class ChunkPreviewResponse(BaseModel):
    chunks: List[UIStateChunkPreview]


class ChunkMarkerRequest(BaseModel):
    snapshot_text: str
    action: Literal["insert", "remove"]
    position: int


class ChunkMarkerResponse(BaseModel):
    snapshot_text_updated: str
    markers_updated: List[int]


class EngineSchemaField(BaseModel):
    key: str
    type: str
    label: Optional[str] = None
    help: Optional[str] = None
    min: Optional[float] = None
    max: Optional[float] = None
    step: Optional[float] = None
    default: Optional[Any] = None
    choices: Optional[List[Any]] = None
    visible_if: Optional[Dict[str, Any]] = None
    serialize_scope: str = "engine"


class EngineSchemaResponse(BaseModel):
    engine_id: str
    backend_id: Optional[str] = None
    capabilities: Dict[str, Any]
    fields: List[EngineSchemaField]
    constraints: Dict[str, Any] = Field(default_factory=dict)


class AudioEditRequest(BaseModel):
    input_wav_path: Optional[str] = None
    asset_id: Optional[str] = None
    trim_enabled: bool = False
    normalize_enabled: bool = False
    target_dbfs: float = -1.0


class AudioEditResponse(BaseModel):
    edited_wav_path: str
    asset_id: Optional[str] = None
    metrics: Dict[str, Any]


class TTSJobRequest(BaseModel):
    text: Optional[str] = None
    engine: Optional[str] = None
    voice: Optional[str] = None
    model: Optional[str] = None
    language: Optional[str] = None
    preset_id: Optional[str] = None
    direction: Optional[DirectionOptions] = None
    options: Optional[Dict[str, Any]] = None
    export: Optional[ExportOptions] = None
    editing: Optional[EditOptions] = None
    engine_id: Optional[str] = None
    voice_id: Optional[str] = None
    text_source: Optional[Literal["raw", "adjusted", "interpreted", "snapshot"]] = None
    text_raw: Optional[str] = None
    text_adjusted: Optional[str] = None
    text_interpreted: Optional[str] = None
    text_snapshot: Optional[str] = None
    chunk_markers: Optional[List[int]] = None
    engine_params: Optional[Dict[str, Any]] = None
    post_params: Optional[Dict[str, Any]] = None
    edit_params: Optional[Dict[str, Any]] = None


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
