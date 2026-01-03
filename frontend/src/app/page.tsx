"use client";

import { useEffect, useMemo, useRef, useState } from "react";

import { DynamicFields } from "@/components/dynamic-fields";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Textarea } from "@/components/ui/textarea";
import { apiDelete, apiGet, apiPost, apiPut, assetUrl, fetchVoices } from "@/lib/api";
import type {
  AudioEditResponse,
  ChunkMarkerResponse,
  ChunkPreviewResponse,
  ChunkSnapshotResponse,
  EngineInfo,
  EngineSchemaField,
  EngineSchemaResponse,
  EnginesResponse,
  JobCreateResponse,
  JobStatusResponse,
  PrepAdjustResponse,
  PrepInterpretResponse,
  PresetListItem,
  PresetListResponse,
  PresetResponse,
  UIState,
  VoiceInfo,
} from "@/lib/types";

const POLL_INTERVAL_MS = 700;

const EMPTY_STATE: UIState = {
  preparation: {
    text_raw: "",
    text_adjusted: "",
    text_interpreted: "",
    glossary_enabled: false,
    glossary_profile: null,
    glossary_options: {},
  },
  direction: {
    snapshot_text: "",
    chunk_markers: [],
    chunk_ranges: [],
    chunks_preview: [],
  },
  engine: {
    engine_id: "",
    voice_id: null,
    params: {},
    chatterbox_gap_ms: 0,
  },
  post: {
    edit_enabled: false,
    trim_enabled: false,
    normalize_enabled: false,
    target_dbfs: -1,
  },
};

function Waveform({ src }: { src: string }) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    let active = true;

    async function draw() {
      try {
        const canvas = canvasRef.current;
        if (!canvas || !src) return;
        const context = canvas.getContext("2d");
        if (!context) return;
        const resp = await fetch(src);
        const buffer = await resp.arrayBuffer();
        const audioContext = new AudioContext();
        const audioBuffer = await audioContext.decodeAudioData(buffer);
        if (!active) {
          audioContext.close();
          return;
        }
        const data = audioBuffer.getChannelData(0);
        const width = canvas.width;
        const height = canvas.height;
        const step = Math.max(1, Math.floor(data.length / width));
        const amp = height / 2;
        context.clearRect(0, 0, width, height);
        context.fillStyle = "#18181b";
        context.globalAlpha = 0.7;
        for (let x = 0; x < width; x += 1) {
          const start = x * step;
          let min = 1;
          let max = -1;
          for (let i = 0; i < step; i += 1) {
            const value = data[start + i] ?? 0;
            if (value < min) min = value;
            if (value > max) max = value;
          }
          const y = (1 + min) * amp;
          const barHeight = Math.max(1, (max - min) * amp);
          context.fillRect(x, y, 1, barHeight);
        }
        await audioContext.close();
      } catch {
        const canvas = canvasRef.current;
        if (!canvas) return;
        const context = canvas.getContext("2d");
        if (!context) return;
        context.clearRect(0, 0, canvas.width, canvas.height);
      }
    }

    draw();
    return () => {
      active = false;
    };
  }, [src]);

  return <canvas ref={canvasRef} width={560} height={90} className="w-full rounded-md bg-zinc-100" />;
}

function buildDefaults(fields: EngineSchemaField[]) {
  const defaults: Record<string, unknown> = {};
  for (const field of fields) {
    if (field.default !== undefined) {
      defaults[field.key] = field.default;
    }
  }
  return defaults;
}

function engineFields(fields: EngineSchemaField[]) {
  return fields.filter((field) => field.serialize_scope !== "post");
}

function postFields(fields: EngineSchemaField[]) {
  return fields.filter((field) => field.serialize_scope === "post");
}

export default function Home() {
  const [uiState, setUiState] = useState<UIState>(EMPTY_STATE);
  const [engines, setEngines] = useState<EngineInfo[]>([]);
  const [voices, setVoices] = useState<VoiceInfo[]>([]);
  const [engineSchema, setEngineSchema] = useState<EngineSchemaResponse | null>(null);
  const [presets, setPresets] = useState<PresetListItem[]>([]);
  const [selectedPreset, setSelectedPreset] = useState<string>("");
  const [presetId, setPresetId] = useState<string>("");
  const [presetLabel, setPresetLabel] = useState<string>("");
  const [prepTab, setPrepTab] = useState("raw");
  const [snapshotCursor, setSnapshotCursor] = useState(0);
  const [status, setStatus] = useState("idle");
  const [progress, setProgress] = useState(0);
  const [jobId, setJobId] = useState<string | null>(null);
  const [assetId, setAssetId] = useState<string | null>(null);
  const [audioHref, setAudioHref] = useState<string | null>(null);
  const [editedPath, setEditedPath] = useState<string | null>(null);
  const [editedAssetId, setEditedAssetId] = useState<string | null>(null);
  const [editedAudioHref, setEditedAudioHref] = useState<string | null>(null);
  const [editedMeta, setEditedMeta] = useState<Record<string, unknown> | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isGenerating, setIsGenerating] = useState(false);
  const [isEditing, setIsEditing] = useState(false);

  const supportsRef = useMemo(() => {
    const engine = engines.find((item) => item.id === uiState.engine.engine_id);
    return Boolean(engine?.supports_ref);
  }, [engines, uiState.engine.engine_id]);

  const engineAvailable = useMemo(() => {
    const engine = engines.find((item) => item.id === uiState.engine.engine_id);
    return Boolean(engine?.available);
  }, [engines, uiState.engine.engine_id]);

  const canGenerate = useMemo(() => {
    if (!uiState.engine.engine_id) return false;
    if (!engineAvailable) return false;
    if (!uiState.preparation.text_raw.trim()) return false;
    if (supportsRef && !uiState.engine.voice_id) return false;
    return !isGenerating;
  }, [uiState, supportsRef, engineAvailable, isGenerating]);

  const engineFieldList = engineFields(engineSchema?.fields ?? []);
  const postFieldList = postFields(engineSchema?.fields ?? []);

  useEffect(() => {
    let active = true;
    async function loadEngines() {
      try {
        const data = await apiGet<EnginesResponse>("/v1/tts/engines");
        if (!active) return;
        setEngines(data.engines);
        if (!uiState.engine.engine_id && data.engines.length > 0) {
          const firstAvailable = data.engines.find((engine) => engine.available) ?? data.engines[0];
          setUiState((prev) => ({
            ...prev,
            engine: { ...prev.engine, engine_id: firstAvailable.id },
          }));
        }
      } catch (err) {
        if (!active) return;
        setError(err instanceof Error ? err.message : "Impossible de charger les moteurs.");
      }
    }
    loadEngines();
    return () => {
      active = false;
    };
  }, [uiState.engine.engine_id]);

  useEffect(() => {
    let active = true;
    async function loadPresets() {
      try {
        const data = await apiGet<PresetListResponse>("/v1/presets");
        if (active) setPresets(data.presets);
      } catch (err) {
        if (!active) return;
        setError(err instanceof Error ? err.message : "Impossible de charger les presets.");
      }
    }
    loadPresets();
    return () => {
      active = false;
    };
  }, []);

  useEffect(() => {
    let active = true;
    async function loadSchema() {
      if (!uiState.engine.engine_id) return;
      try {
        const schema = await apiGet<EngineSchemaResponse>("/v1/tts/engine_schema", {
          engine: uiState.engine.engine_id,
        });
        if (!active) return;
        setEngineSchema(schema);
        const defaults = buildDefaults(schema.fields ?? []);
        const nextParams: Record<string, unknown> = {};
        const allowed = new Set(engineFields(schema.fields ?? []).map((field) => field.key));
        const gapField = (schema.fields ?? []).find((field) => field.key === "chatterbox_gap_ms");
        const gapDefault = typeof gapField?.default === "number" ? gapField.default : 0;
        setUiState((prev) => ({
          ...prev,
          engine: (() => {
            for (const key of allowed) {
              if (prev.engine.params[key] !== undefined) {
                nextParams[key] = prev.engine.params[key];
              } else if (defaults[key] !== undefined) {
                nextParams[key] = defaults[key];
              }
            }
            return {
              ...prev.engine,
              params: nextParams,
              chatterbox_gap_ms: prev.engine.chatterbox_gap_ms ?? gapDefault,
            };
          })(),
        }));
      } catch (err) {
        if (!active) return;
        setEngineSchema(null);
        setError(err instanceof Error ? err.message : "Schema moteur indisponible.");
      }
    }
    loadSchema();
    return () => {
      active = false;
    };
  }, [uiState.engine.engine_id]);

  useEffect(() => {
    let active = true;
    async function loadVoices() {
      if (!uiState.engine.engine_id || !supportsRef) {
        setVoices([]);
        return;
      }
      try {
        const data = await fetchVoices(uiState.engine.engine_id);
        if (!active) return;
        setVoices(data.voices);
        if (!uiState.engine.voice_id && data.voices.length > 0) {
          setUiState((prev) => ({
            ...prev,
            engine: { ...prev.engine, voice_id: data.voices[0].id },
          }));
        }
      } catch (err) {
        if (!active) return;
        setVoices([]);
        setError(err instanceof Error ? err.message : "Impossible de charger les voix.");
      }
    }
    loadVoices();
    return () => {
      active = false;
    };
  }, [uiState.engine.engine_id, supportsRef, uiState.engine.voice_id]);

  async function refreshPresets() {
    const data = await apiGet<PresetListResponse>("/v1/presets");
    setPresets(data.presets);
  }

  async function handleLoadPreset() {
    const target = selectedPreset || presetId;
    if (!target) return;
    if (!window.confirm(`Charger le preset \"${target}\" ?`)) return;
    const data = await apiGet<PresetResponse>(`/v1/presets/${target}`);
    setUiState(data.state);
    setPresetId(data.id);
    setPresetLabel(data.label ?? "");
    setSelectedPreset(data.id);
  }

  async function handleSavePreset() {
    if (!presetId) return;
    if (!window.confirm(`Sauvegarder le preset \"${presetId}\" ?`)) return;
    const payload = {
      id: presetId,
      label: presetLabel || presetId,
      state: { ...uiState, preset_id: presetId },
    };
    const exists = presets.some((item) => item.id === presetId);
    if (exists) {
      await apiPut(`/v1/presets/${presetId}`, payload);
    } else {
      await apiPost("/v1/presets", payload);
    }
    await refreshPresets();
  }

  async function handleDeletePreset() {
    if (!presetId) return;
    if (!window.confirm(`Supprimer le preset \"${presetId}\" ?`)) return;
    await apiDelete(`/v1/presets/${presetId}`);
    if (selectedPreset === presetId) {
      setSelectedPreset("");
    }
    await refreshPresets();
  }

  async function handlePrepare() {
    setError(null);
    const adjust = await apiPost<PrepAdjustResponse>("/v1/prep/adjust", {
      text_raw: uiState.preparation.text_raw,
    });
    const interpret = await apiPost<PrepInterpretResponse>("/v1/prep/interpret", {
      text_adjusted: adjust.text_adjusted,
      glossary_enabled: uiState.preparation.glossary_enabled,
      glossary_profile: uiState.preparation.glossary_profile,
      glossary_options: uiState.preparation.glossary_options,
    });
    setUiState((prev) => ({
      ...prev,
      preparation: {
        ...prev.preparation,
        text_adjusted: adjust.text_adjusted,
        text_interpreted: interpret.text_interpreted,
      },
    }));
  }

  async function handleSnapshot() {
    const source = uiState.preparation.text_interpreted || uiState.preparation.text_adjusted;
    const response = await apiPost<ChunkSnapshotResponse>("/v1/chunks/snapshot", {
      text_interpreted: source,
    });
    setUiState((prev) => ({
      ...prev,
      direction: {
        ...prev.direction,
        snapshot_text: response.snapshot_text,
        chunk_markers: [],
        chunks_preview: [],
      },
    }));
  }

  async function handleMarker(action: "insert" | "remove") {
    const response = await apiPost<ChunkMarkerResponse>("/v1/chunks/apply_marker", {
      snapshot_text: uiState.direction.snapshot_text,
      action,
      position: snapshotCursor,
    });
    setUiState((prev) => ({
      ...prev,
      direction: {
        ...prev.direction,
        snapshot_text: response.snapshot_text_updated,
        chunk_markers: response.markers_updated,
      },
    }));
  }

  async function handlePreview() {
    const response = await apiPost<ChunkPreviewResponse>("/v1/chunks/preview", {
      snapshot_text: uiState.direction.snapshot_text,
    });
    setUiState((prev) => ({
      ...prev,
      direction: {
        ...prev.direction,
        chunks_preview: response.chunks,
      },
    }));
  }

  async function handleGenerate() {
    if (!canGenerate) return;
    setIsGenerating(true);
    setError(null);
    setStatus("queued");
    setProgress(0);
    setJobId(null);
    setAssetId(null);
    setAudioHref(null);
    setEditedPath(null);
    setEditedAssetId(null);
    setEditedAudioHref(null);
    setEditedMeta(null);

    try {
      const textSnapshot = uiState.direction.snapshot_text.trim();
      const payload = {
        engine_id: uiState.engine.engine_id,
        voice_id: supportsRef ? uiState.engine.voice_id : null,
        text_source: textSnapshot ? "snapshot" : "interpreted",
        text_raw: uiState.preparation.text_raw,
        text_adjusted: uiState.preparation.text_adjusted,
        text_interpreted: uiState.preparation.text_interpreted,
        text_snapshot: textSnapshot || null,
        chunk_markers: uiState.direction.chunk_markers,
        engine_params: uiState.engine.params,
        post_params: {
          chatterbox_gap_ms: uiState.engine.chatterbox_gap_ms,
        },
      };
      const job = await apiPost<JobCreateResponse>("/v1/tts/jobs", payload);
      setJobId(job.job_id);
      setStatus(job.status);

      let done = false;
      let jobStatus: JobStatusResponse | null = null;
      while (!done) {
        jobStatus = await apiGet<JobStatusResponse>(`/v1/jobs/${job.job_id}`);
        setStatus(jobStatus.status);
        setProgress(jobStatus.progress);
        if (["done", "error", "canceled"].includes(jobStatus.status)) {
          done = true;
          break;
        }
        if (!["queued", "running"].includes(jobStatus.status)) {
          done = true;
          break;
        }
        await new Promise((resolve) => setTimeout(resolve, POLL_INTERVAL_MS));
      }

      if (jobStatus?.status === "done" && jobStatus.asset_id) {
        setAssetId(jobStatus.asset_id);
        setAudioHref(assetUrl(jobStatus.asset_id));
      } else if (jobStatus?.status === "error") {
        setError(jobStatus.error || "Erreur pendant la generation.");
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Generation impossible.");
    } finally {
      setIsGenerating(false);
    }
  }

  async function handleCancelJob() {
    if (!jobId) return;
    if (!window.confirm("Annuler la generation en cours ?")) return;
    try {
      await apiDelete(`/v1/jobs/${jobId}`);
      setStatus("canceled");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Annulation impossible.");
    }
  }

  async function handleEdit() {
    if (!assetId || !uiState.post.edit_enabled) return;
    setIsEditing(true);
    setEditedPath(null);
    setEditedAssetId(null);
    setEditedAudioHref(null);
    setEditedMeta(null);
    try {
      const result = await apiPost<AudioEditResponse>("/v1/audio/edit", {
        asset_id: assetId,
        trim_enabled: uiState.post.trim_enabled,
        normalize_enabled: uiState.post.normalize_enabled,
        target_dbfs: uiState.post.target_dbfs,
      });
      setEditedPath(result.edited_wav_path);
      setEditedAssetId(result.asset_id ?? null);
      setEditedAudioHref(result.asset_id ? assetUrl(result.asset_id) : null);
      setEditedMeta(result.metrics);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Edition impossible.");
    } finally {
      setIsEditing(false);
    }
  }

  const internalVoiceCount = engineSchema?.fields?.find((field) => field.key === "voice_id")?.choices?.length ?? 0;
  const context = {
    ...(engineSchema?.capabilities ?? {}),
    voice_count: supportsRef ? voices.length : internalVoiceCount,
    piper_supports_speed: true,
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-amber-50 via-white to-slate-100 text-zinc-900">
      <main className="mx-auto flex w-full max-w-5xl flex-col gap-6 px-6 py-10">
        <header className="flex flex-col gap-2">
          <Badge variant="outline" className="w-fit">API-driven</Badge>
          <h1 className="text-3xl font-semibold tracking-tight">Vocalie-TTS</h1>
          <p className="text-sm text-zinc-500">
            Parite Gradio → API v1 + React. Tous les etats sont serialisables.
          </p>
        </header>

        {error && (
          <div className="rounded-md border border-red-200 bg-red-50 px-4 py-2 text-sm text-red-700">
            {error}
          </div>
        )}

        {!engineAvailable && uiState.engine.engine_id && (
          <div className="rounded-md border border-amber-200 bg-amber-50 px-4 py-2 text-sm text-amber-800">
            Moteur indisponible: {uiState.engine.engine_id}. Installe-le via `./scripts/bootstrap.sh std` ou `./scripts/bootstrap.sh bark`.
          </div>
        )}

        <Card>
          <CardHeader>
            <CardTitle>Presets</CardTitle>
          </CardHeader>
          <CardContent className="grid gap-4">
            <div className="grid gap-3 md:grid-cols-[1fr_1fr]">
              <div className="grid gap-2">
                <label className="text-sm font-medium">Preset disponible</label>
                <Select
                  value={selectedPreset}
                  onValueChange={(value) => {
                    setSelectedPreset(value);
                    setPresetId(value);
                  }}
                >
                  <SelectTrigger>
                    <SelectValue placeholder="Choisir un preset" />
                  </SelectTrigger>
                  <SelectContent>
                    {presets.map((preset) => (
                      <SelectItem key={preset.id} value={preset.id}>
                        {preset.label || preset.id}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              <div className="grid gap-2">
                <label className="text-sm font-medium">Preset ID</label>
                <Input value={presetId} onChange={(event) => setPresetId(event.target.value)} />
              </div>
            </div>
            <div className="grid gap-3 md:grid-cols-[1fr_auto]">
              <div className="grid gap-2">
                <label className="text-sm font-medium">Label</label>
                <Input value={presetLabel} onChange={(event) => setPresetLabel(event.target.value)} />
              </div>
              <div className="flex flex-wrap items-end gap-2">
                <Button variant="outline" onClick={refreshPresets}>Rafraichir</Button>
                <Button variant="outline" onClick={handleLoadPreset}>Charger</Button>
                <Button onClick={handleSavePreset}>Sauvegarder</Button>
                <Button variant="destructive" onClick={handleDeletePreset}>Supprimer</Button>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Preparation</CardTitle>
          </CardHeader>
          <CardContent className="grid gap-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <Switch
                  checked={uiState.preparation.glossary_enabled}
                  onCheckedChange={(checked) =>
                    setUiState((prev) => ({
                      ...prev,
                      preparation: { ...prev.preparation, glossary_enabled: checked },
                    }))
                  }
                />
                <span className="text-sm text-zinc-600">Glossaire active</span>
              </div>
              <Button onClick={handlePrepare}>Preparer</Button>
            </div>
            <Tabs value={prepTab} onValueChange={setPrepTab}>
              <TabsList>
                <TabsTrigger value="raw">Texte</TabsTrigger>
                <TabsTrigger value="adjusted">Ajuste</TabsTrigger>
                <TabsTrigger value="interpreted">Interprete</TabsTrigger>
              </TabsList>
              <TabsContent value="raw">
                <Textarea
                  value={uiState.preparation.text_raw}
                  onChange={(event) =>
                    setUiState((prev) => ({
                      ...prev,
                      preparation: { ...prev.preparation, text_raw: event.target.value },
                    }))
                  }
                  placeholder="Texte brut..."
                />
              </TabsContent>
              <TabsContent value="adjusted">
                <Textarea
                  value={uiState.preparation.text_adjusted}
                  onChange={(event) =>
                    setUiState((prev) => ({
                      ...prev,
                      preparation: { ...prev.preparation, text_adjusted: event.target.value },
                    }))
                  }
                  placeholder="Texte ajuste..."
                />
              </TabsContent>
              <TabsContent value="interpreted">
                <Textarea
                  value={uiState.preparation.text_interpreted}
                  onChange={(event) =>
                    setUiState((prev) => ({
                      ...prev,
                      preparation: { ...prev.preparation, text_interpreted: event.target.value },
                    }))
                  }
                  placeholder="Texte interprete..."
                />
              </TabsContent>
            </Tabs>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Direction (Chunking)</CardTitle>
          </CardHeader>
          <CardContent className="grid gap-4">
            <div className="flex flex-wrap items-center gap-2">
              <Button variant="outline" onClick={handleSnapshot}>Snapshot</Button>
              <Button variant="outline" onClick={() => handleMarker("insert")}>Inserer marqueur</Button>
              <Button variant="outline" onClick={() => handleMarker("remove")}>Retirer marqueur</Button>
              <Button onClick={handlePreview}>Preview chunks</Button>
            </div>
            <Textarea
              value={uiState.direction.snapshot_text}
              onChange={(event) =>
                setUiState((prev) => ({
                  ...prev,
                  direction: { ...prev.direction, snapshot_text: event.target.value },
                }))
              }
              onSelect={(event) => {
                const target = event.currentTarget;
                setSnapshotCursor(target.selectionStart ?? 0);
              }}
              placeholder="Texte snapshot pour placer les chunks"
              rows={6}
            />
            <div className="rounded-md border border-zinc-200 bg-white p-3">
              <p className="text-xs font-semibold uppercase text-zinc-500">Apercu</p>
              <div className="mt-2 max-h-40 overflow-auto text-sm text-zinc-700">
                {uiState.direction.chunks_preview.length === 0 ? (
                  <p className="text-zinc-400">Aucun chunk.</p>
                ) : (
                  uiState.direction.chunks_preview.map((chunk) => (
                    <div key={chunk.index} className="border-b border-zinc-100 py-2 last:border-b-0">
                      <p className="font-medium">Chunk {chunk.index}</p>
                      <p className="text-xs text-zinc-500">
                        {chunk.word_count ?? "-"} mots · {chunk.est_duration_s ? chunk.est_duration_s.toFixed(2) : "--"}s
                      </p>
                      <p className="text-sm text-zinc-700">{chunk.text}</p>
                    </div>
                  ))
                )}
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Moteur</CardTitle>
          </CardHeader>
          <CardContent className="grid gap-4">
            <div className="grid gap-3 md:grid-cols-[1fr_1fr]">
              <div className="grid gap-2">
                <label className="text-sm font-medium">Engine</label>
                <Select
                  value={uiState.engine.engine_id}
                  onValueChange={(value) =>
                    setUiState((prev) => ({
                      ...prev,
                      engine: { ...prev.engine, engine_id: value, params: {}, voice_id: null },
                    }))
                  }
                >
                  <SelectTrigger>
                    <SelectValue placeholder="Choisir un moteur" />
                  </SelectTrigger>
                  <SelectContent>
                    {engines.map((engine) => (
                      <SelectItem key={engine.id} value={engine.id} disabled={!engine.available}>
                        {engine.label}{engine.available ? "" : " (indisponible)"}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              {supportsRef && (
                <div className="grid gap-2">
                  <label className="text-sm font-medium">Reference vocale</label>
                  <Select
                    value={uiState.engine.voice_id ?? ""}
                    onValueChange={(value) =>
                      setUiState((prev) => ({
                        ...prev,
                        engine: { ...prev.engine, voice_id: value },
                      }))
                    }
                  >
                    <SelectTrigger>
                      <SelectValue placeholder="Choisir une voix" />
                    </SelectTrigger>
                    <SelectContent>
                      {voices.map((voice) => (
                        <SelectItem key={voice.id} value={voice.id}>
                          {voice.label}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
              )}
            </div>
            {engineFieldList.length > 0 && (
              <DynamicFields
                fields={engineFieldList}
                values={uiState.engine.params}
                context={context}
                onChange={(key, value, field) => {
                  if (field.serialize_scope === "post" && key === "chatterbox_gap_ms") {
                    setUiState((prev) => ({
                      ...prev,
                      engine: { ...prev.engine, chatterbox_gap_ms: Number(value) },
                    }));
                    return;
                  }
                  setUiState((prev) => ({
                    ...prev,
                    engine: {
                      ...prev.engine,
                      params: { ...prev.engine.params, [key]: value },
                    },
                  }));
                }}
              />
            )}
            {postFieldList.length > 0 && (
              <div className="grid gap-3">
                <p className="text-xs font-semibold uppercase text-zinc-500">Post parametres</p>
                <DynamicFields
                  fields={postFieldList}
                  values={{ chatterbox_gap_ms: uiState.engine.chatterbox_gap_ms }}
                  context={context}
                  onChange={(key, value) => {
                    if (key === "chatterbox_gap_ms") {
                      setUiState((prev) => ({
                        ...prev,
                        engine: { ...prev.engine, chatterbox_gap_ms: Number(value) },
                      }));
                    }
                  }}
                />
              </div>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Generation</CardTitle>
          </CardHeader>
          <CardContent className="grid gap-4">
            <div className="flex flex-wrap items-center gap-3">
              <Button onClick={handleGenerate} disabled={!canGenerate}>
                {isGenerating ? "Generation..." : "Generer"}
              </Button>
              <Button variant="outline" onClick={handleCancelJob} disabled={!jobId || !isGenerating}>
                Annuler
              </Button>
              <div className="text-sm text-zinc-500">{status} · {(progress * 100).toFixed(0)}%</div>
            </div>
            {audioHref && (
              <div className="rounded-md border border-zinc-200 bg-white p-3">
                <Waveform src={audioHref} />
                <audio controls src={audioHref} className="w-full" />
                <p className="mt-2 text-xs text-zinc-500">Asset: {assetId}</p>
              </div>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Edition audio</CardTitle>
          </CardHeader>
          <CardContent className="grid gap-4">
            <label className="flex items-center justify-between rounded-md border border-zinc-200 px-3 py-2">
              <span className="text-sm">Activer edition</span>
              <Switch
                checked={uiState.post.edit_enabled}
                onCheckedChange={(checked) =>
                  setUiState((prev) => ({
                    ...prev,
                    post: { ...prev.post, edit_enabled: checked },
                  }))
                }
              />
            </label>
            <div className="grid gap-3 md:grid-cols-[1fr_1fr]">
              <label className="flex items-center justify-between rounded-md border border-zinc-200 px-3 py-2">
                <span className="text-sm">Trim silence</span>
                <Switch
                  checked={uiState.post.trim_enabled}
                  disabled={!uiState.post.edit_enabled}
                  onCheckedChange={(checked) =>
                    setUiState((prev) => ({
                      ...prev,
                      post: { ...prev.post, trim_enabled: checked },
                    }))
                  }
                />
              </label>
              <label className="flex items-center justify-between rounded-md border border-zinc-200 px-3 py-2">
                <span className="text-sm">Normalize</span>
                <Switch
                  checked={uiState.post.normalize_enabled}
                  disabled={!uiState.post.edit_enabled}
                  onCheckedChange={(checked) =>
                    setUiState((prev) => ({
                      ...prev,
                      post: { ...prev.post, normalize_enabled: checked },
                    }))
                  }
                />
              </label>
            </div>
            <div className="grid gap-2">
              <div className="flex items-center justify-between text-sm">
                <span className="font-medium">Target dBFS</span>
                <span className="font-mono text-xs text-zinc-500">{uiState.post.target_dbfs.toFixed(1)} dB</span>
              </div>
              <Slider
                value={[uiState.post.target_dbfs]}
                min={-12}
                max={0}
                step={0.5}
                disabled={!uiState.post.edit_enabled}
                onValueChange={(vals) =>
                  setUiState((prev) => ({
                    ...prev,
                    post: { ...prev.post, target_dbfs: vals[0] },
                  }))
                }
              />
            </div>
            <div className="flex items-center gap-2">
              <Button onClick={handleEdit} disabled={!assetId || isEditing || !uiState.post.edit_enabled}>
                {isEditing ? "Edition..." : "Editer"}
              </Button>
              {editedPath && <span className="text-xs text-zinc-500">{editedPath}</span>}
            </div>
            {editedAudioHref && (
              <div className="rounded-md border border-zinc-200 bg-white p-3">
                <Waveform src={editedAudioHref} />
                <audio controls src={editedAudioHref} className="w-full" />
                <p className="mt-2 text-xs text-zinc-500">Asset: {editedAssetId}</p>
              </div>
            )}
            {editedMeta && (
              <div className="rounded-md border border-zinc-200 bg-white px-3 py-2 text-xs text-zinc-600">
                <pre className="whitespace-pre-wrap">{JSON.stringify(editedMeta, null, 2)}</pre>
              </div>
            )}
          </CardContent>
        </Card>
      </main>
    </div>
  );
}
