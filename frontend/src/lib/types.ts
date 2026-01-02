export type EngineInfo = {
  id: string;
  label: string;
  available: boolean;
  supports_ref: boolean;
};

export type EnginesResponse = {
  engines: EngineInfo[];
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
