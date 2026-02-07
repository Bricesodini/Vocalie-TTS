export type EngineInfo = {
  id: string;
  label: string;
  available: boolean;
  supports_ref: boolean;
};

export type EnginesResponse = {
  engines: EngineInfo[];
};

export type AudioSRStatus = {
  enabled: boolean;
  available: boolean;
};

export type CapabilitiesResponse = {
  engines: string[];
  features: Record<string, unknown>;
  audiosr?: AudioSRStatus | null;
};

export type PresetListItem = {
  id: string;
  label?: string | null;
  updated_at?: string | null;
};

export type PresetListResponse = {
  presets: PresetListItem[];
};

export type UIStatePreparation = {
  text_raw: string;
  text_adjusted: string;
  text_interpreted: string;
  glossary_enabled: boolean;
  glossary_profile?: string | null;
  glossary_options?: Record<string, unknown>;
};

export type UIStateChunkRange = {
  start: number;
  end: number;
};

export type UIStateChunkPreview = {
  index: number;
  text: string;
  est_duration_s?: number | null;
  word_count?: number | null;
};

export type UIStateDirection = {
  snapshot_text: string;
  chunk_markers: number[];
  chunk_ranges: UIStateChunkRange[];
  chunks_preview: UIStateChunkPreview[];
};

export type UIStateEngine = {
  engine_id: string;
  voice_id?: string | null;
  language?: string | null;
  params: Record<string, unknown>;
  chunk_gap_ms: number;
};

export type UIStatePost = {
  edit_enabled: boolean;
  trim_enabled: boolean;
  normalize_enabled: boolean;
  target_dbfs: number;
};

export type UIState = {
  preset_id?: string | null;
  preparation: UIStatePreparation;
  direction: UIStateDirection;
  engine: UIStateEngine;
  post: UIStatePost;
};

export type PresetResponse = {
  id: string;
  label?: string | null;
  state: UIState;
  updated_at?: string | null;
};

export type VoiceInfo = {
  id: string;
  label: string;
  meta?: Record<string, unknown> | null;
};

export type VoicesResponse = {
  engine: string;
  voices: VoiceInfo[];
};

export type JobCreateResponse = {
  job_id: string;
  status: string;
};

export type JobStatusResponse = {
  job_id: string;
  status: string;
  progress: number;
  created_at: string;
  started_at?: string | null;
  finished_at?: string | null;
  asset_id?: string | null;
  error?: string | null;
};

export type PrepAdjustResponse = {
  text_adjusted: string;
};

export type PrepInterpretResponse = {
  text_interpreted: string;
  applied_rules_summary?: string[] | null;
};

export type ChunkSnapshotResponse = {
  snapshot_text: string;
  snapshot_id?: string | null;
};

export type ChunkPreviewResponse = {
  chunks: UIStateChunkPreview[];
};

export type ChunkMarkerResponse = {
  snapshot_text_updated: string;
  markers_updated: number[];
};

export type EngineSchemaField = {
  key: string;
  type: string;
  label?: string | null;
  help?: string | null;
  min?: number | null;
  max?: number | null;
  step?: number | null;
  default?: unknown;
  choices?: Array<unknown>;
  visible_if?: Record<string, unknown> | null;
  serialize_scope?: string | null;
};

export type EngineSchemaResponse = {
  engine_id: string;
  backend_id?: string | null;
  capabilities: Record<string, unknown>;
  fields: EngineSchemaField[];
  constraints?: Record<string, unknown>;
};

export type AudioEditResponse = {
  edited_wav_path: string;
  asset_id?: string | null;
  metrics: Record<string, unknown>;
};

export type AudioEnhanceResponse = {
  output_file: string;
  sample_rate: number;
  duration_s: number;
  asset_id?: string | null;
  engine: string;
};

export type AssetMetaResponse = {
  asset_id: string;
  file_name: string;
  relative_path?: string | null;
  size_bytes: number;
  duration_s?: number | null;
  sample_rate?: number | null;
  engine?: string | null;
  voice?: string | null;
  model?: string | null;
  created_at?: string | null;
  job_id?: string | null;
};
