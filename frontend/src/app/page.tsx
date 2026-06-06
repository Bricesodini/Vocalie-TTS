"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { Collapsible } from "@/components/collapsible";
import { DynamicFields } from "@/components/dynamic-fields";
import { Button } from "@/components/ui/button";
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
import { Textarea } from "@/components/ui/textarea";
import { apiDelete, apiGet, apiPost, apiPostForm, apiPut, assetUrl, fetchModels, fetchVoices } from "@/lib/api";
import {
  EMPTY_STATE,
  LANGUAGE_OPTIONS,
  POLL_INTERVAL_MS,
  VOICE_DESIGN_OPTIONS,
  VOICE_DESIGN_PRESETS,
} from "@/lib/page-constants";
import type {
  AssetMetaResponse,
  AudioEditResponse,
  AudioEnhanceResponse,
  CapabilitiesResponse,
  ChunkMarkerResponse,
  ChunkPreviewResponse,
  ChunkSnapshotResponse,
  EngineInfo,
  EngineSchemaField,
  EngineSchemaResponse,
  EnginesResponse,
  GlossaryEntry,
  GlossaryListResponse,
  JobCreateResponse,
  JobStatusResponse,
  ModelsResponse,
  PrepAdjustResponse,
  PrepInterpretResponse,
  PresetListItem,
  PresetListResponse,
  PresetResponse,
  ModelInfo,
  RefDirConfig,
  RefListResponse,
  UIState,
  VoiceInfo,
} from "@/lib/types";

/* ── Helpers ──────────────────────────────────────────────────────────────── */

function buildDefaults(fields: EngineSchemaField[]) {
  const defaults: Record<string, unknown> = {};
  for (const field of fields) { if (field.default !== undefined) defaults[field.key] = field.default; }
  return defaults;
}
function engineFields(fields: EngineSchemaField[]) { return fields.filter((f) => f.serialize_scope !== "post"); }
function postFields(fields: EngineSchemaField[]) { return fields.filter((f) => f.serialize_scope === "post"); }

type FilePickerOptions = { suggestedName?: string; types?: Array<{ description?: string; accept: Record<string, string[]> }> };
type FileWritable = { write: (data: Blob) => Promise<void>; close: () => Promise<void> };
type FileHandle = { createWritable: () => Promise<FileWritable> };
type ShowSaveFilePicker = (options?: FilePickerOptions) => Promise<FileHandle>;

async function saveAudioBlob(blob: Blob, suggestedName: string) {
  if (typeof window !== "undefined" && "showSaveFilePicker" in window) {
    const picker = (window as Window & { showSaveFilePicker?: ShowSaveFilePicker }).showSaveFilePicker;
    if (picker) {
      const handle = await picker({ suggestedName, types: [{ description: "WAV audio", accept: { "audio/wav": [".wav"] } }] });
      const writable = await handle.createWritable(); await writable.write(blob); await writable.close(); return;
    }
  }
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a"); link.href = url; link.download = suggestedName; document.body.appendChild(link); link.click(); link.remove(); URL.revokeObjectURL(url);
}

function buildVoiceDesignInstruction(params: Record<string, unknown>, language?: string | null) {
  const genderMap: Record<string, string> = { masculine: "Voix masculine", feminine: "Voix feminine" };
  const ageMap: Record<string, string> = { teen: "ado", young_adult: "jeune adulte", adult: "adulte", senior: "senior" };
  const pitchMap: Record<string, string> = { low: "pitch bas", mid: "pitch moyen", high: "pitch haut" };
  const speedMap: Record<string, string> = { slow: "debit lent", medium: "debit normal", fast: "debit rapide" };
  const volumeMap: Record<string, string> = { soft: "volume faible", normal: "volume normal", loud: "volume fort" };
  const accentMap: Record<string, string> = { fr_neutral: "accent francais neutre", fr_paris: "accent francais parisien", fr_quebec: "accent francais quebecois", fr_belgium: "accent francais belge", fr_swiss: "accent francais suisse" };
  const emotionMap: Record<string, string> = { happy: "emotion joyeuse", sad: "emotion triste", angry: "emotion colerique", excited: "emotion enthousiaste", calm: "emotion calme" };
  const textureMap: Record<string, string> = { clear: "timbre clair", warm: "timbre chaleureux", raspy: "timbre rauque", nasal: "timbre nasal" };
  const styleMap: Record<string, string> = { conversational: "ton conversationnel", narrative: "ton narratif", authoritative: "ton autoritaire", dramatic: "ton dramatique" };
  const parts: string[] = [];
  if (language && language.startsWith("fr")) parts.push("Parle en francais");
  const picks: Array<[string, Record<string, string>]> = [
    ["design_gender", genderMap], ["design_age", ageMap], ["design_texture", textureMap],
    ["design_pitch", pitchMap], ["design_speed", speedMap], ["design_volume", volumeMap],
    ["design_accent", accentMap], ["design_emotion", emotionMap], ["design_style", styleMap],
  ];
  for (const [key, map] of picks) { const k = String(params[key] || ""); if (k && k !== "none" && map[k]) parts.push(map[k]); }
  if (!parts.some((p) => p.includes("accent")) && language && language.startsWith("fr")) parts.push("accent francais neutre");
  return parts.length ? parts.join(", ") + "." : "";
}

/* ── Waveform ────────────────────────────────────────────────────────────── */

function Waveform({ src }: { src: string }) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  useEffect(() => {
    let active = true;
    async function draw() {
      try {
        const canvas = canvasRef.current; if (!canvas || !src) return;
        const context = canvas.getContext("2d"); if (!context) return;
        const resp = await fetch(src); const buffer = await resp.arrayBuffer();
        const audioContext = new AudioContext(); const audioBuffer = await audioContext.decodeAudioData(buffer);
        if (!active) { audioContext.close(); return; }
        const data = audioBuffer.getChannelData(0); const width = canvas.width; const height = canvas.height;
        const step = Math.max(1, Math.floor(data.length / width)); const amp = height / 2;
        context.clearRect(0, 0, width, height); context.fillStyle = "#18181b"; context.globalAlpha = 0.7;
        for (let x = 0; x < width; x++) { const start = x * step; let min = 1, max = -1; for (let i = 0; i < step; i++) { const v = data[start + i] ?? 0; if (v < min) min = v; if (v > max) max = v; } context.fillRect(x, (1 + min) * amp, 1, Math.max(1, (max - min) * amp)); }
        await audioContext.close();
      } catch { /* ignore */ }
    }
    draw(); return () => { active = false; };
  }, [src]);
  return <canvas ref={canvasRef} width={560} height={90} className="w-full rounded-md bg-zinc-100" />;
}

/* ── Main ────────────────────────────────────────────────────────────────── */

export default function Home() {
  const [uiState, setUiState] = useState<UIState>(EMPTY_STATE);
  const [engines, setEngines] = useState<EngineInfo[]>([]);
  const [voices, setVoices] = useState<VoiceInfo[]>([]);
  const [engineSchema, setEngineSchema] = useState<EngineSchemaResponse | null>(null);
  const [schemaRefreshNonce, setSchemaRefreshNonce] = useState(0);
  const [status, setStatus] = useState("idle");
  const [progress, setProgress] = useState(0);
  const [jobId, setJobId] = useState<string | null>(null);
  const [assetId, setAssetId] = useState<string | null>(null);
  const [audioHref, setAudioHref] = useState<string | null>(null);
  const [editedAssetId, setEditedAssetId] = useState<string | null>(null);
  const [editedAudioHref, setEditedAudioHref] = useState<string | null>(null);
  const [editSummary, setEditSummary] = useState<{ sample_rate?: number | null; duration_s?: number | null } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isGenerating, setIsGenerating] = useState(false);
  const [isEditing, setIsEditing] = useState(false);
  const [editProgress, setEditProgress] = useState(0);
  const [isExporting, setIsExporting] = useState(false);
  const [isExportingEdited, setIsExportingEdited] = useState(false);
  const [audiosrStatus, setAudiosrStatus] = useState<{ enabled: boolean; available: boolean } | null>(null);
  const [audiosrEnabled, setAudiosrEnabled] = useState(false);
  const [audiosrParams, setAudiosrParams] = useState({ ddim_steps: 100, guidance_scale: 2.5, seed: 0, multiband_ensemble: false, chunk_size: 32768, overlap: 1024, input_cutoff: 8000 });
  const [audiosrAdvancedOpen, setAudiosrAdvancedOpen] = useState(false);
  const [glossaryEntries, setGlossaryEntries] = useState<GlossaryEntry[]>([]);
  const [newGlossWord, setNewGlossWord] = useState("");
  const [newGlossPron, setNewGlossPron] = useState("");
  const [showGlossaryEditor, setShowGlossaryEditor] = useState(false);
  const [refDir, setRefDir] = useState("");
  const [refFiles, setRefFiles] = useState<string[]>([]);
  const [refDirInput, setRefDirInput] = useState("");
  const [voiceDesignPresets, setVoiceDesignPresets] = useState<Array<{ id: string; label: string; instruct: string; options: Record<string, unknown> }>>([]);
  const [voiceDesignPresetId, setVoiceDesignPresetId] = useState("none");
  const [voiceDesignPresetName, setVoiceDesignPresetName] = useState("");
  const [presets, setPresets] = useState<PresetListItem[]>([]);
  const [selectedPreset, setSelectedPreset] = useState("");
  const [presetId, setPresetId] = useState<string>("");
  const [presetLabel, setPresetLabel] = useState<string>("");
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [editedPath, setEditedPath] = useState<string | null>(null);
  const [snapshotCursor, setSnapshotCursor] = useState(0);
  const prepareTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  // Track whether user has manually edited the snapshot (to avoid overwriting their separators)
  const [userEditedSnapshot, setUserEditedSnapshot] = useState(false);

  // ── Derived ─────────────────────────────────────────────────────────
  const supportsRef = useMemo(() => Boolean(engines.find((e) => e.id === uiState.engine.engine_id)?.supports_ref), [engines, uiState.engine.engine_id]);
  const engineAvailable = useMemo(() => Boolean(engines.find((e) => e.id === uiState.engine.engine_id)?.available), [engines, uiState.engine.engine_id]);
  const canGenerate = useMemo(() => {
    if (!uiState.engine.engine_id || !engineAvailable) return false;
    if (!uiState.direction.snapshot_text.trim() && !uiState.preparation.text_raw.trim()) return false;
    if (supportsRef && !uiState.engine.voice_id) return false;
    return !isGenerating;
  }, [uiState, supportsRef, engineAvailable, isGenerating]);
  const audiosrAvailable = audiosrStatus?.available ?? false;
  const canApplyEdit = Boolean(assetId && !isEditing);
  const exportTargetId = editedAssetId ?? assetId;
  const isQwen3Custom = uiState.engine.engine_id === "qwen3_custom";
  const isQwen3VoiceDesign = isQwen3Custom && uiState.engine.params?.qwen3_mode === "voice_design";
  const autoResolvedKeys: string[] = Array.isArray(engineSchema?.capabilities?.auto_resolved_keys) ? (engineSchema.capabilities.auto_resolved_keys as string[]) : [];
  const engineFieldList = useMemo(() => {
    const base = engineFields(engineSchema?.fields ?? []);
    if (!uiState.engine.engine_id) return base;
    return base.filter((f) => {
      if (autoResolvedKeys.includes(f.key)) return false;
      if (isQwen3VoiceDesign && f.key === "instruct") return false;
      return true;
    });
  }, [engineSchema, uiState.engine.engine_id, isQwen3VoiceDesign, autoResolvedKeys]);
  const postFieldList = postFields(engineSchema?.fields ?? []);
  const internalVoiceCount = engineSchema?.fields?.find((f) => f.key === "voice_id")?.choices?.length ?? 0;
  const context = { ...(engineSchema?.capabilities ?? {}), voice_count: supportsRef ? voices.length : internalVoiceCount };

  function updateEngineParam(key: string, value: unknown) {
    setUiState((prev) => ({ ...prev, engine: { ...prev.engine, params: { ...prev.engine.params, [key]: value } } }));
  }

  // ── Auto-prepare ──────────────────────────────────────────────────────
  const autoPrepare = useCallback(async (textRaw: string) => {
    if (!textRaw.trim()) return;
    try {
      const adjust = await apiPost<PrepAdjustResponse>("/v1/prep/adjust", { text_raw: textRaw });
      const interpret = await apiPost<PrepInterpretResponse>("/v1/prep/interpret", {
        text_adjusted: adjust.text_adjusted,
        glossary_enabled: uiState.preparation.glossary_enabled,
        glossary_profile: uiState.preparation.glossary_profile,
        glossary_options: uiState.preparation.glossary_options,
      });
      // Only update snapshot if user hasn't manually edited it (preserve their separators)
      setUiState((prev) => {
        const newText = interpret.text_interpreted || adjust.text_adjusted || textRaw;
        // If user has inserted separators, merge the interpreted text but keep separators
        const prevSnapshot = prev.direction.snapshot_text;
        let snapshotText: string;
        if (userEditedSnapshot && prevSnapshot.trim()) {
          // User has manually edited — don't overwrite, just update internal state
          snapshotText = prevSnapshot;
        } else {
          snapshotText = newText;
        }
        return {
          ...prev,
          preparation: { ...prev.preparation, text_adjusted: adjust.text_adjusted, text_interpreted: interpret.text_interpreted },
          direction: { ...prev.direction, snapshot_text: snapshotText },
        };
      });
    } catch (err) { console.error("[Vocalie] Auto-prepare failed:", err); }
  }, [uiState.preparation.glossary_enabled, uiState.preparation.glossary_profile, uiState.preparation.glossary_options, userEditedSnapshot]);

  const scheduleAutoPrepare = useCallback((text: string) => {
    if (prepareTimerRef.current) clearTimeout(prepareTimerRef.current);
    prepareTimerRef.current = setTimeout(() => autoPrepare(text), 600);
  }, [autoPrepare]);

  const autoPreviewChunks = useCallback(async (snapshotText: string) => {
    if (!snapshotText.trim()) { setUiState((prev) => ({ ...prev, direction: { ...prev.direction, chunks_preview: [] } })); return; }
    try {
      const response = await apiPost<ChunkPreviewResponse>("/v1/chunks/preview", { snapshot_text: snapshotText });
      setUiState((prev) => ({ ...prev, direction: { ...prev.direction, chunks_preview: response.chunks } }));
    } catch { /* silent */ }
  }, []);

  // ── Data loading ───────────────────────────────────────────────────────
  useEffect(() => { console.log("[Vocalie] App starting"); }, []);
  useEffect(() => {
    let active = true;
    async function loadEngines() {
      try { const data = await apiGet<EnginesResponse>("/v1/tts/engines"); if (!active) return; setEngines(data.engines); if (!uiState.engine.engine_id && data.engines.length > 0) { const first = data.engines.find((e) => e.available) ?? data.engines[0]; setUiState((prev) => ({ ...prev, engine: { ...prev.engine, engine_id: first.id } })); } }
      catch (err) { if (active) setError(err instanceof Error ? err.message : "Impossible de charger les moteurs."); }
    }
    loadEngines(); return () => { active = false; };
  }, [uiState.engine.engine_id]);

  useEffect(() => {
    let active = true;
    async function loadCapabilities() { try { const data = await apiGet<CapabilitiesResponse>("/v1/capabilities"); if (active) setAudiosrStatus({ enabled: Boolean(data.audiosr?.enabled), available: Boolean(data.audiosr?.available) }); } catch { if (active) setAudiosrStatus({ enabled: false, available: false }); } }
    loadCapabilities(); return () => { active = false; };
  }, []);

  useEffect(() => { if (!audiosrAvailable) setAudiosrEnabled(false); }, [audiosrAvailable]);

  useEffect(() => {
    let active = true;
    async function loadPresets() { try { const data = await apiGet<PresetListResponse>("/v1/presets"); if (active) setPresets(data.presets); } catch {} }
    loadPresets(); return () => { active = false; };
  }, []);

  useEffect(() => {
    let active = true;
    async function loadSchema() {
      if (!uiState.engine.engine_id) return;
      try {
        const schema = await apiGet<EngineSchemaResponse>("/v1/tts/engine_schema", { engine: uiState.engine.engine_id });
        if (!active) return;
        setEngineSchema(schema);
        const defaults = buildDefaults(schema.fields ?? []);
        const nextParams: Record<string, unknown> = {};
        const allowed = new Set(engineFields(schema.fields ?? []).map((f) => f.key));
        const gapField = (schema.fields ?? []).find((f) => f.key === "chunk_gap_ms");
        const gapDefault = typeof gapField?.default === "number" ? gapField.default : 0;
        setUiState((prev) => { for (const key of allowed) { if (prev.engine.params[key] !== undefined) nextParams[key] = prev.engine.params[key]; else if (defaults[key] !== undefined) nextParams[key] = defaults[key]; } return { ...prev, engine: { ...prev.engine, language: prev.engine.language ?? "fr-FR", params: nextParams, chunk_gap_ms: prev.engine.chunk_gap_ms ?? gapDefault } }; });
      } catch (err) { if (!active) { setEngineSchema(null); setError(err instanceof Error ? err.message : "Schema moteur indisponible."); } }
    }
    loadSchema(); return () => { active = false; };
  }, [uiState.engine.engine_id, schemaRefreshNonce]);

  useEffect(() => { setUiState((prev) => prev.engine.language ? prev : { ...prev, engine: { ...prev.engine, language: "fr-FR" } }); }, [uiState.engine.engine_id]);

  useEffect(() => {
    let active = true;
    async function loadVoices() {
      if (!uiState.engine.engine_id || !supportsRef) { setVoices([]); return; }
      try { const data = await fetchVoices(uiState.engine.engine_id); if (active) setVoices(data.voices); if (!uiState.engine.voice_id && data.voices.length > 0) setUiState((prev) => ({ ...prev, engine: { ...prev.engine, voice_id: data.voices[0].id } })); }
      catch { if (active) setVoices([]); }
    }
    loadVoices(); return () => { active = false; };
  }, [uiState.engine.engine_id, supportsRef, uiState.engine.voice_id]);

  useEffect(() => { try { const raw = window.localStorage.getItem("voiceDesignPresets"); if (raw) { const p = JSON.parse(raw); if (Array.isArray(p)) setVoiceDesignPresets(p); } } catch {} }, []);
  useEffect(() => { try { window.localStorage.setItem("voiceDesignPresets", JSON.stringify(voiceDesignPresets)); } catch {} }, [voiceDesignPresets]);
  useEffect(() => { if (!isQwen3VoiceDesign) return; const accent = String(uiState.engine.params?.design_accent ?? ""); if ((uiState.engine.language || "").startsWith("fr") && (!accent || accent === "none")) updateEngineParam("design_accent", "fr_neutral"); }, [isQwen3VoiceDesign, uiState.engine.language]);

  useEffect(() => { let active = true; async function load() { try { const data = await apiGet<GlossaryListResponse>("/v1/glossary"); if (active) setGlossaryEntries(data.entries); } catch {} } load(); return () => { active = false; }; }, []);

  // Load refs (persist dir from localStorage)
  useEffect(() => {
    let active = true;
    async function loadRefs() {
      const savedDir = window.localStorage.getItem("vocalie_ref_dir") || "";
      try { let dir = savedDir; if (dir) { try { const config = await apiPut<RefDirConfig>("/v1/refs/dir", { directory: dir }); dir = config.directory; } catch {} } const data = await apiGet<RefListResponse>("/v1/refs"); if (active) { setRefDir(data.directory); setRefDirInput(data.directory); setRefFiles(data.files); } } catch {}
    }
    loadRefs(); return () => { active = false; };
  }, []);

  useEffect(() => { if (!uiState.engine.engine_id || !supportsRef) return; let active = true; async function reload() { try { const data = await fetchVoices(uiState.engine.engine_id); if (active) setVoices(data.voices); } catch {} } reload(); return () => { active = false; }; }, [refFiles]);

  useEffect(() => {
    let active = true;
    async function loadModels() { try { const data = await fetchModels(uiState.engine.engine_id); if (active) setModels(data.models); } catch { if (active) setModels([]); } }
    loadModels();
    return () => { active = false; };
  }, [uiState.engine.engine_id]);

  function refreshEngineSchema() { setSchemaRefreshNonce((v) => v + 1); }

  // Auto-preview chunks
  useEffect(() => { const text = uiState.direction.snapshot_text; if (!text.trim()) { setUiState((prev) => ({ ...prev, direction: { ...prev.direction, chunks_preview: [] } })); return; } const timer = setTimeout(() => autoPreviewChunks(text), 500); return () => clearTimeout(timer); }, [uiState.direction.snapshot_text, autoPreviewChunks]);

  // ── Handlers ───────────────────────────────────────────────────────────
  async function refreshPresets() { const data = await apiGet<PresetListResponse>("/v1/presets"); setPresets(data.presets); }
  async function handleLoadPreset() { const target = selectedPreset || presetId; if (!target) return; const data = await apiGet<PresetResponse>(`/v1/presets/${target}`); setUiState(data.state); setPresetId(data.id); setPresetLabel(data.label ?? ""); setSelectedPreset(data.id); setUserEditedSnapshot(false); }
  async function handleSavePreset() { if (!presetId) return; const payload = { id: presetId, label: presetLabel || presetId, state: { ...uiState, preset_id: presetId } }; const exists = presets.some((p) => p.id === presetId); if (exists) await apiPut(`/v1/presets/${presetId}`, payload); else await apiPost("/v1/presets", payload); await refreshPresets(); }
  async function handleDeletePreset() { if (!presetId) return; await apiDelete(`/v1/presets/${presetId}`); if (selectedPreset === presetId) setSelectedPreset(""); await refreshPresets(); }

  async function handleMarker(action: "insert" | "remove") {
    setError(null);
    try {
      const resp = await apiPost<ChunkMarkerResponse>("/v1/chunks/apply_marker", { snapshot_text: uiState.direction.snapshot_text, action, position: snapshotCursor });
      setUiState((p) => ({ ...p, direction: { ...p.direction, snapshot_text: resp.snapshot_text_updated, chunk_markers: resp.markers_updated } }));
      setUserEditedSnapshot(true);
    } catch (err) { setError(err instanceof Error ? err.message : "Impossible de modifier le marqueur."); }
  }

  // Handle text change in the unified textarea
  function handleTextChange(text: string) {
    setUiState((prev) => ({
      ...prev,
      preparation: { ...prev.preparation, text_raw: text },
      direction: { ...prev.direction, snapshot_text: text },
    }));
    setUserEditedSnapshot(true);
    scheduleAutoPrepare(text);
  }

  async function handleGenerate() {
    if (!canGenerate) return;
    setIsGenerating(true); setError(null); setStatus("queued"); setProgress(0); setJobId(null); setAssetId(null); setAudioHref(null); setEditedAssetId(null); setEditedAudioHref(null); setEditSummary(null);
    try {
      const textSnapshot = uiState.direction.snapshot_text.trim();
      const payload = {
        engine_id: uiState.engine.engine_id, voice_id: supportsRef ? uiState.engine.voice_id : null, language: uiState.engine.language,
        text_source: textSnapshot ? "snapshot" : "interpreted",
        text_raw: uiState.preparation.text_raw, text_adjusted: uiState.preparation.text_adjusted, text_interpreted: uiState.preparation.text_interpreted,
        text_snapshot: textSnapshot || null, chunk_markers: uiState.direction.chunk_markers,
        engine_params: uiState.engine.params, post_params: { chunk_gap_ms: uiState.engine.chunk_gap_ms },
      };
      const job = await apiPost<JobCreateResponse>("/v1/tts/jobs", payload);
      setJobId(job.job_id); setStatus(job.status);
      let done = false; let jobStatus: JobStatusResponse | null = null;
      while (!done) {
        jobStatus = await apiGet<JobStatusResponse>(`/v1/jobs/${job.job_id}`);
        setStatus(jobStatus.status); setProgress(jobStatus.progress);
        if (["done", "error", "canceled"].includes(jobStatus.status) || !["queued", "running"].includes(jobStatus.status)) { done = true; break; }
        await new Promise((r) => setTimeout(r, POLL_INTERVAL_MS));
      }
      if (jobStatus?.status === "done" && jobStatus.asset_id) { setAssetId(jobStatus.asset_id); setAudioHref(assetUrl(jobStatus.asset_id)); }
      else if (jobStatus?.status === "error") setError(jobStatus.error || "Erreur pendant la generation.");
    } catch (err) { setError(err instanceof Error ? err.message : "Generation impossible."); }
    finally { setIsGenerating(false); }
  }
  async function handleCancelJob() { if (!jobId) return; try { await apiDelete(`/v1/jobs/${jobId}`); setStatus("canceled"); } catch (err) { setError(err instanceof Error ? err.message : "Annulation impossible."); } }

  async function enhanceAsset(sourceAssetId: string): Promise<AudioEnhanceResponse> {
    const resp = await fetch(assetUrl(sourceAssetId)); if (!resp.ok) throw new Error(`${resp.status} ${resp.statusText}`); const blob = await resp.blob();
    const file = new File([blob], `vocalie-${sourceAssetId}.wav`, { type: blob.type || "audio/wav" }); const formData = new FormData(); formData.append("file", file); formData.append("engine", "audiosr");
    formData.append("ddim_steps", String(audiosrParams.ddim_steps)); formData.append("guidance_scale", String(audiosrParams.guidance_scale)); formData.append("seed", String(audiosrParams.seed));
    formData.append("chunk_size", String(audiosrParams.chunk_size)); formData.append("overlap", String(audiosrParams.overlap));
    formData.append("multiband_ensemble", audiosrParams.multiband_ensemble ? "1" : "0"); formData.append("input_cutoff", String(audiosrParams.input_cutoff));
    return apiPostForm("/v1/audio/enhance", formData);
  }
  async function fetchEditSummary(targetAssetId: string) { try { const meta = await apiGet<AssetMetaResponse>(`/v1/assets/${targetAssetId}/meta`); setEditSummary({ sample_rate: meta.sample_rate ?? null, duration_s: meta.duration_s ?? null }); } catch { setEditSummary(null); } }
  async function handleEdit() {
    if (!assetId) return; setIsEditing(true); setEditProgress(0); setEditedAssetId(null); setEditedAudioHref(null); setEditSummary(null);
    let currentAssetId = assetId;
    const totalSteps = (audiosrEnabled ? 1 : 0) + (uiState.post.trim_enabled ? 1 : 0) + (uiState.post.normalize_enabled ? 1 : 0); let completedSteps = 0;
    try {
      if (audiosrEnabled) { const enhanced = await enhanceAsset(currentAssetId); if (enhanced.asset_id) currentAssetId = enhanced.asset_id; completedSteps++; if (totalSteps > 0) setEditProgress(completedSteps / totalSteps); }
      if (uiState.post.trim_enabled) { const trimmed = await apiPost<AudioEditResponse>("/v1/audio/edit", { asset_id: currentAssetId, trim_enabled: true, normalize_enabled: false, target_dbfs: uiState.post.target_dbfs }); if (trimmed.asset_id) currentAssetId = trimmed.asset_id; completedSteps++; if (totalSteps > 0) setEditProgress(completedSteps / totalSteps); }
      if (uiState.post.normalize_enabled) { const normalized = await apiPost<AudioEditResponse>("/v1/audio/edit", { asset_id: currentAssetId, trim_enabled: false, normalize_enabled: true, target_dbfs: uiState.post.target_dbfs }); if (normalized.asset_id) currentAssetId = normalized.asset_id; completedSteps++; if (totalSteps > 0) setEditProgress(completedSteps / totalSteps); }
      setEditedAssetId(currentAssetId); setEditedAudioHref(assetUrl(currentAssetId)); await fetchEditSummary(currentAssetId);
    } catch (err) { setError(err instanceof Error ? err.message : "Edition impossible."); }
    finally { setEditProgress(1); setIsEditing(false); }
  }
  async function handleExport(targetAssetId: string | null, suggestedName: string, setBusy: (v: boolean) => void) {
    if (!targetAssetId) return; setBusy(true); setError(null);
    try { const resp = await fetch(assetUrl(targetAssetId)); if (!resp.ok) throw new Error(`${resp.status} ${resp.statusText}`); const blob = await resp.blob(); await saveAudioBlob(blob, suggestedName); }
    catch (err) { if (err instanceof DOMException && err.name === "AbortError") return; setError(err instanceof Error ? err.message : "Export impossible."); }
    finally { setBusy(false); }
  }

  async function handleAddGlossaryEntry() { const word = newGlossWord.trim(); const pronunciation = newGlossPron.trim(); if (!word || !pronunciation) return; try { await apiPut<GlossaryEntry>("/v1/glossary", { word, pronunciation }); setNewGlossWord(""); setNewGlossPron(""); const data = await apiGet<GlossaryListResponse>("/v1/glossary"); setGlossaryEntries(data.entries); } catch (err) { setError(err instanceof Error ? err.message : "Impossible d'ajouter l'entree glossaire."); } }
  async function handleDeleteGlossaryEntry(word: string) { try { await apiDelete<{ word: string }>(`/v1/glossary?word=${encodeURIComponent(word)}`); const data = await apiGet<GlossaryListResponse>("/v1/glossary"); setGlossaryEntries(data.entries); } catch (err) { setError(err instanceof Error ? err.message : "Impossible de supprimer l'entree glossaire."); } }

  async function handleSetRefDir() { const dir = refDirInput.trim(); if (!dir) return; try { const config = await apiPut<RefDirConfig>("/v1/refs/dir", { directory: dir }); setRefDir(config.directory); setRefDirInput(config.directory); window.localStorage.setItem("vocalie_ref_dir", config.directory); const data = await apiGet<RefListResponse>("/v1/refs"); setRefFiles(data.files); if (uiState.engine.engine_id && supportsRef) { const voiceData = await fetchVoices(uiState.engine.engine_id); setVoices(voiceData.voices); } } catch (err) { setError(err instanceof Error ? err.message : "Impossible de configurer le dossier."); } }
  async function handleUploadRefs(e: React.ChangeEvent<HTMLInputElement>) { const files = e.target.files; if (!files || files.length === 0) return; try { const formData = new FormData(); for (let i = 0; i < files.length; i++) formData.append("files", files[i]); await apiPostForm<RefListResponse>("/v1/refs/upload", formData); const data = await apiGet<RefListResponse>("/v1/refs"); setRefFiles(data.files); if (uiState.engine.engine_id && supportsRef) { const voiceData = await fetchVoices(uiState.engine.engine_id); setVoices(voiceData.voices); } } catch (err) { setError(err instanceof Error ? err.message : "Upload impossible."); } e.target.value = ""; }
  async function handleDeleteRef(filename: string) { try { await apiDelete(`/v1/refs/${encodeURIComponent(filename)}`); const data = await apiGet<RefListResponse>("/v1/refs"); setRefFiles(data.files); if (uiState.engine.engine_id && supportsRef) { const voiceData = await fetchVoices(uiState.engine.engine_id); setVoices(voiceData.voices); } } catch (err) { setError(err instanceof Error ? err.message : "Impossible de supprimer le fichier."); } }

  // ── Render ──────────────────────────────────────────────────────────────
  return (
    <div className="min-h-screen bg-gradient-to-br from-amber-50 via-white to-slate-100 text-zinc-900">
      <main className="mx-auto flex w-full max-w-4xl flex-col gap-4 px-6 py-8">
        <header><h1 className="text-2xl font-semibold tracking-tight">Vocalie-TTS</h1></header>

        {error && (
          <div className="rounded-md border border-red-200 bg-red-50 px-4 py-2 text-sm text-red-700">
            <strong>Erreur : </strong>{error}
            <button className="ml-2 text-xs underline hover:text-red-900" onClick={() => setError(null)}>Fermer</button>
          </div>
        )}
        {(!engineAvailable && uiState.engine.engine_id) && (
          <div className="rounded-md border border-amber-200 bg-amber-50 px-4 py-2 text-sm text-amber-800">
            Moteur indisponible : {uiState.engine.engine_id}
          </div>
        )}

        {/* ── Presets (collapsed by default) ────────────── */}
        <Collapsible title="Presets" badge={`${presets.length}`}>
          <div className="grid gap-3">
            <div className="grid gap-3 md:grid-cols-[1fr_1fr]">
              <div className="grid gap-2">
                <label className="text-sm font-medium">Preset</label>
                <Select value={selectedPreset} onValueChange={(v) => { setSelectedPreset(v); setPresetId(v); }}>
                  <SelectTrigger><SelectValue placeholder="Choisir un preset" /></SelectTrigger>
                  <SelectContent>{presets.map((p) => <SelectItem key={p.id} value={p.id}>{p.label || p.id}</SelectItem>)}</SelectContent>
                </Select>
              </div>
              <div className="grid gap-2">
                <label className="text-sm font-medium">Preset ID</label>
                <Input value={presetId} onChange={(e) => setPresetId(e.target.value)} />
              </div>
            </div>
            <div className="grid gap-3 md:grid-cols-[1fr_auto]">
              <div className="grid gap-2">
                <label className="text-sm font-medium">Label</label>
                <Input value={presetLabel} onChange={(e) => setPresetLabel(e.target.value)} />
              </div>
              <div className="flex flex-wrap items-end gap-2">
                <Button variant="outline" size="sm" onClick={refreshPresets}>Rafraichir</Button>
                <Button variant="outline" size="sm" onClick={handleLoadPreset}>Charger</Button>
                <Button size="sm" onClick={handleSavePreset}>Sauvegarder</Button>
                <Button variant="destructive" size="sm" onClick={handleDeletePreset}>Supprimer</Button>
              </div>
            </div>
          </div>
        </Collapsible>

        {/* ── Preparation — tabs + glossaire ───────────── */}
        <Collapsible title="Texte" defaultOpen>
          <div className="grid gap-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <Switch checked={uiState.preparation.glossary_enabled} onCheckedChange={(v) => setUiState((p) => ({ ...p, preparation: { ...p.preparation, glossary_enabled: v } }))} />
                <span className="text-sm text-zinc-600">{uiState.preparation.glossary_enabled ? "Glossaire actif" : "Glossaire inactif"}</span>
                {uiState.preparation.glossary_enabled && <span className="text-xs text-zinc-400">(auto)</span>}
              </div>
              <div className="flex items-center gap-2">
                <Button variant="outline" size="sm" onClick={() => handleMarker("insert")}>Insérer séparateur</Button>
                <Button variant="outline" size="sm" onClick={() => handleMarker("remove")}>Retirer séparateur</Button>
                {uiState.direction.chunk_markers.length > 0 && (
                  <span className="text-xs text-zinc-500">{uiState.direction.chunk_markers.length} séparateur{uiState.direction.chunk_markers.length > 1 ? "s" : ""}</span>
                )}
              </div>
            </div>
            <Textarea
              value={uiState.direction.snapshot_text}
              onChange={(e) => {
                const v = e.target.value;
                setUiState((p) => ({
                  ...p,
                  direction: { ...p.direction, snapshot_text: v },
                  preparation: { ...p.preparation, text_raw: v },
                }));
                setUserEditedSnapshot(true);
                scheduleAutoPrepare(v);
              }}
              onSelect={(e) => setSnapshotCursor(e.currentTarget.selectionStart ?? 0)}
              onKeyUp={(e) => setSnapshotCursor(e.currentTarget.selectionStart ?? 0)}
              onClick={(e) => setSnapshotCursor(e.currentTarget.selectionStart ?? 0)}
              placeholder="Saisissez votre texte ici. Cliquez dans la zone puis utilisez « Insérer séparateur » pour placer un [[CHUNK]] au curseur."
              rows={8}
            />
            {uiState.preparation.text_adjusted && uiState.preparation.text_adjusted !== uiState.direction.snapshot_text && (
              <div className="grid gap-1">
                <p className="text-xs font-semibold uppercase text-zinc-500">Aperçu normalisé (lexique)</p>
                <Textarea
                  value={uiState.preparation.text_adjusted}
                  readOnly
                  rows={4}
                  className="bg-zinc-50 text-zinc-700"
                />
              </div>
            )}
          </div>
        </Collapsible>

        {/* ── Aperçu des chunks (auto) ─────────── */}
        <Collapsible title="Aperçu des chunks" defaultOpen>
          <div className="grid gap-3">
            {uiState.direction.chunks_preview.length === 0 ? (
              <p className="text-sm text-zinc-500">Saisissez du texte pour voir l’aperçu des chunks.</p>
            ) : (
              <div className="flex flex-wrap gap-2">
                {uiState.direction.chunks_preview.map((chunk) => (
                  <div key={chunk.index} className="flex flex-col items-start rounded-md border border-zinc-200 bg-white px-3 py-2 text-sm">
                    <span className="font-semibold text-zinc-800">#{chunk.index}</span>
                    <span className="text-xs text-zinc-500">{chunk.word_count ?? "-"} mots · {chunk.est_duration_s ? chunk.est_duration_s.toFixed(1) : "--"}s</span>
                    <span className="mt-0.5 line-clamp-3 text-zinc-600">{chunk.text}</span>
                  </div>
                ))}
              </div>
            )}
          </div>
        </Collapsible>

        {/* ── Moteur ─────────────────────────────────────────── */}
        <div className="rounded-md border border-zinc-200 bg-white px-4 py-4">
          <h2 className="text-lg font-semibold">Moteur</h2>
          <div className="mt-3 grid gap-4">
            <div className="grid gap-3 md:grid-cols-[1fr_1fr]">
              <div className="grid gap-2">
                <label className="text-sm font-medium">Engine</label>
                <Select value={uiState.engine.engine_id} onValueChange={(v) => setUiState((p) => ({ ...p, engine: { ...p.engine, engine_id: v, params: {}, voice_id: null, language: p.engine.language ?? "fr-FR" } }))}>
                  <SelectTrigger><SelectValue placeholder="Choisir un moteur" /></SelectTrigger>
                  <SelectContent>{engines.map((e) => <SelectItem key={e.id} value={e.id} disabled={!e.available}>{e.label}{e.available ? "" : " (indisponible)"}</SelectItem>)}</SelectContent>
                </Select>
              </div>
              {supportsRef && (
                <div className="grid gap-2">
                  <label className="text-sm font-medium">Reference vocale</label>
                  <Select value={uiState.engine.voice_id ?? ""} onValueChange={(v) => setUiState((p) => ({ ...p, engine: { ...p.engine, voice_id: v } }))}>
                    <SelectTrigger><SelectValue placeholder="Choisir une voix" /></SelectTrigger>
                    <SelectContent>{voices.map((v) => <SelectItem key={v.id} value={v.id}>{v.label}</SelectItem>)}</SelectContent>
                  </Select>
                </div>
              )}
              <div className="grid gap-2">
                <label className="text-sm font-medium">Langue</label>
                <Select value={uiState.engine.language ?? "fr-FR"} onValueChange={(v) => setUiState((p) => ({ ...p, engine: { ...p.engine, language: v } }))}>
                  <SelectTrigger><SelectValue placeholder="Choisir une langue" /></SelectTrigger>
                  <SelectContent>{LANGUAGE_OPTIONS.map((l) => <SelectItem key={l.value} value={l.value}>{l.label}</SelectItem>)}</SelectContent>
                </Select>
              </div>
            </div>



            {!!engineSchema?.capabilities?.can_refresh_speakers && (
              <div><Button variant="outline" size="sm" onClick={() => setSchemaRefreshNonce((v) => v + 1)}>Mise a jour speakers</Button></div>
            )}
          </div>

          {/* Engine details (collapsed by default) */}
          {(engineFieldList.length > 0 || postFieldList.length > 0) && (
            <Collapsible title="Details moteur" badge={`${engineFieldList.length + postFieldList.length} params`} className="mt-4">
              <div className="grid gap-4">
                {engineFieldList.length > 0 && (
                  <DynamicFields fields={engineFieldList} values={uiState.engine.params} context={context} onChange={(key, value, field) => {
                    if (field.serialize_scope === "post" && key === "chunk_gap_ms") { setUiState((p) => ({ ...p, engine: { ...p.engine, chunk_gap_ms: Number(value) } })); return; }
                    updateEngineParam(key, value);
                  }} />
                )}
                {postFieldList.length > 0 && (
                  <div className="grid gap-3"><p className="text-xs font-semibold uppercase text-zinc-500">Post parametres</p>
                    <DynamicFields fields={postFieldList} values={{ chunk_gap_ms: uiState.engine.chunk_gap_ms }} context={context} onChange={(key, value) => { if (key === "chunk_gap_ms") setUiState((p) => ({ ...p, engine: { ...p.engine, chunk_gap_ms: Number(value) } })); }} />
                  </div>
                )}
                {isQwen3VoiceDesign && (
                  <div className="grid gap-3 rounded-md border border-zinc-200 px-3 py-3">
                    <p className="text-sm font-medium text-zinc-900">Presets VoiceDesign</p>
                    <div className="grid gap-3 md:grid-cols-[1fr_1fr]">
                      <div className="grid gap-2"><label className="text-sm font-medium">Preset</label>
                        <Select value={voiceDesignPresetId} onValueChange={(v) => setVoiceDesignPresetId(v)}><SelectTrigger><SelectValue placeholder="Choisir" /></SelectTrigger>
                          <SelectContent><SelectItem value="none">Aucun</SelectItem>{voiceDesignPresets.map((p) => <SelectItem key={p.id} value={p.id}>{p.label}</SelectItem>)}</SelectContent></Select></div>
                      <div className="grid gap-2"><label className="text-sm font-medium">Nom</label><Input value={voiceDesignPresetName} onChange={(e) => setVoiceDesignPresetName(e.target.value)} placeholder="Nom du preset" /></div>
                    </div>
                    <div className="flex flex-wrap gap-2">
                      <Button variant="outline" size="sm" disabled={voiceDesignPresetId === "none"} onClick={() => { const p = voiceDesignPresets.find((i) => i.id === voiceDesignPresetId); if (!p) return; updateEngineParam("instruct", p.instruct); Object.entries(p.options || {}).forEach(([k, v]) => updateEngineParam(k, v)); }}>Charger</Button>
                      <Button variant="outline" size="sm" onClick={() => { const name = voiceDesignPresetName.trim(); if (name.length < 2) return; const id = name.toLowerCase().replace(/\s+/g, "-"); const options: Record<string, unknown> = {}; ["design_gender", "design_age", "design_pitch", "design_speed", "design_volume", "design_accent", "design_emotion", "design_texture", "design_style"].forEach((k) => { if (uiState.engine.params?.[k] !== undefined) options[k] = uiState.engine.params[k]; }); setVoiceDesignPresets((prev) => [...prev.filter((i) => i.id !== id), { id, label: name, instruct: String(uiState.engine.params?.instruct || ""), options }].sort((a, b) => a.label.localeCompare(b.label))); setVoiceDesignPresetId(id); }}>Enregistrer</Button>
                      <Button variant="outline" size="sm" disabled={voiceDesignPresetId === "none"} onClick={() => { setVoiceDesignPresets((prev) => prev.filter((i) => i.id !== voiceDesignPresetId)); setVoiceDesignPresetId("none"); }}>Supprimer</Button>
                    </div>
                    <div className="grid gap-2"><label className="text-sm font-medium">Instruction</label><Input value={String(uiState.engine.params?.instruct ?? "")} onChange={(e) => updateEngineParam("instruct", e.target.value)} placeholder="Style/intonation (optionnel)." /></div>
                    <div className="flex items-center justify-between"><div><p className="text-sm font-medium">Guidage VoiceDesign</p><p className="text-xs text-zinc-500">Attributs pour generer l'instruction.</p></div><Button variant="outline" size="sm" onClick={() => { const built = buildVoiceDesignInstruction(uiState.engine.params, uiState.engine.language); if (built) updateEngineParam("instruct", built); }}>Generer</Button></div>
                    <div className="grid gap-2"><label className="text-sm font-medium">Preset rapide</label><div className="flex flex-wrap gap-2">{VOICE_DESIGN_PRESETS.map((p) => <Button key={p.id} variant="outline" size="sm" onClick={() => updateEngineParam("instruct", p.instruct)}>{p.label}</Button>)}</div></div>
                    <div className="grid gap-3 md:grid-cols-[1fr_1fr]">
                      {([["design_gender", "Genre", VOICE_DESIGN_OPTIONS.gender], ["design_age", "Age", VOICE_DESIGN_OPTIONS.age], ["design_pitch", "Pitch", VOICE_DESIGN_OPTIONS.pitch], ["design_speed", "Vitesse", VOICE_DESIGN_OPTIONS.speed], ["design_volume", "Volume", VOICE_DESIGN_OPTIONS.volume], ["design_accent", "Accent", VOICE_DESIGN_OPTIONS.accent], ["design_emotion", "Emotion", VOICE_DESIGN_OPTIONS.emotion], ["design_texture", "Texture", VOICE_DESIGN_OPTIONS.texture], ["design_style", "Style", VOICE_DESIGN_OPTIONS.style]] as Array<[string, string, Array<{ value: string; label: string }>]>).map(([key, label, choices]) => (
                        <div key={key} className="grid gap-2"><label className="text-sm font-medium">{label}</label>
                          <Select value={String(uiState.engine.params?.[key] ?? "")} onValueChange={(v) => updateEngineParam(key, v)}><SelectTrigger><SelectValue placeholder="Choisir" /></SelectTrigger>
                            <SelectContent>{choices.map((c) => <SelectItem key={c.value} value={c.value}>{c.label}</SelectItem>)}</SelectContent></Select></div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </Collapsible>
          )}
        </div>

        {/* ── Generation ────────────────────────────────────── */}
        <div className="rounded-md border border-zinc-200 bg-white px-4 py-4">
          <h2 className="text-lg font-semibold">Generation</h2>
          <div className="mt-3 flex flex-wrap items-center gap-3">
            <Button onClick={handleGenerate} disabled={!canGenerate}>{isGenerating ? "Generation..." : "Generer"}</Button>
            <Button variant="outline" onClick={handleCancelJob} disabled={!jobId || !isGenerating}>Annuler</Button>
            <Button variant="outline" onClick={() => handleExport(assetId, `vocalie-generation-${assetId ?? "audio"}.wav`, setIsExporting)} disabled={!assetId || isExporting}>{isExporting ? "Export..." : "Exporter"}</Button>
            {isGenerating && <div className="flex items-center gap-2 text-sm text-zinc-500"><span className="inline-flex h-3 w-3 animate-spin rounded-full border-2 border-zinc-400 border-t-transparent" /><span>{status} · {(progress * 100).toFixed(0)}%</span></div>}
          </div>
          {audioHref && (
            <div className="mt-4 rounded-md border border-zinc-200 bg-white p-3"><Waveform src={audioHref} /><audio controls src={audioHref} className="w-full" /></div>
          )}
        </div>

        {/* ── Edition audio (collapsed by default) ───────── */}
        <Collapsible title="Edition audio">
          <div className="grid gap-4">
            <label className="flex items-center justify-between rounded-md border border-zinc-200 px-3 py-2">
              <div className="flex flex-col"><span className="text-sm">Amelioration audio (AudioSR)</span>{!audiosrAvailable && <span className="text-xs text-zinc-500">Module non installe</span>}</div>
              <Switch checked={audiosrEnabled} disabled={!audiosrAvailable} onCheckedChange={(v) => setAudiosrEnabled(v)} />
            </label>
            <div className={`grid gap-3 rounded-md border border-zinc-200 px-3 py-2 ${audiosrEnabled ? "" : "opacity-60"}`}>
              <div className="flex items-center justify-between text-sm"><span className="font-medium">Steps</span><span className="font-mono text-xs text-zinc-500">{audiosrParams.ddim_steps}</span></div>
              <Slider value={[audiosrParams.ddim_steps]} min={20} max={250} step={1} disabled={!audiosrEnabled || !audiosrAvailable} onValueChange={(v) => setAudiosrParams((p) => ({ ...p, ddim_steps: v[0] }))} />
              <div className="flex items-center justify-between text-sm"><span className="font-medium">Guidance</span><span className="font-mono text-xs text-zinc-500">{audiosrParams.guidance_scale.toFixed(1)}</span></div>
              <Slider value={[audiosrParams.guidance_scale]} min={1} max={4} step={0.1} disabled={!audiosrEnabled || !audiosrAvailable} onValueChange={(v) => setAudiosrParams((p) => ({ ...p, guidance_scale: v[0] }))} />
              <div className="grid gap-2"><label className="text-xs text-zinc-500">Seed</label><Input type="number" value={audiosrParams.seed} disabled={!audiosrEnabled || !audiosrAvailable} onChange={(e) => setAudiosrParams((p) => ({ ...p, seed: Number(e.target.value) }))} /></div>
              <label className="flex items-center justify-between rounded-md border border-zinc-200 px-3 py-2"><span className="text-sm">Multiband ensemble</span><Switch checked={audiosrParams.multiband_ensemble} disabled={!audiosrEnabled || !audiosrAvailable} onCheckedChange={(v) => setAudiosrParams((p) => ({ ...p, multiband_ensemble: v }))} /></label>
              <Button variant="outline" size="sm" onClick={() => setAudiosrAdvancedOpen((v) => !v)} disabled={!audiosrEnabled || !audiosrAvailable}>{audiosrAdvancedOpen ? "Masquer" : "Options avancees"}</Button>
              {audiosrAdvancedOpen && (
                <div className="grid gap-2">
                  <label className="text-xs text-zinc-500">Chunk size</label><Input type="number" value={audiosrParams.chunk_size} disabled={!audiosrEnabled || !audiosrAvailable} onChange={(e) => setAudiosrParams((p) => ({ ...p, chunk_size: Number(e.target.value) }))} />
                  <label className="text-xs text-zinc-500">Overlap</label><Input type="number" value={audiosrParams.overlap} disabled={!audiosrEnabled || !audiosrAvailable} onChange={(e) => setAudiosrParams((p) => ({ ...p, overlap: Number(e.target.value) }))} />
                  <label className="text-xs text-zinc-500">Input cutoff</label><Input type="number" value={audiosrParams.input_cutoff} disabled={!audiosrEnabled || !audiosrAvailable} onChange={(e) => setAudiosrParams((p) => ({ ...p, input_cutoff: Number(e.target.value) }))} />
                </div>
              )}
            </div>
            <label className="flex items-center justify-between rounded-md border border-zinc-200 px-3 py-2"><span className="text-sm">Trim silence</span><Switch checked={uiState.post.trim_enabled} onCheckedChange={(v) => setUiState((p) => ({ ...p, post: { ...p.post, trim_enabled: v } }))} /></label>
            <label className="flex items-center justify-between rounded-md border border-zinc-200 px-3 py-2"><span className="text-sm">Normalize</span><Switch checked={uiState.post.normalize_enabled} onCheckedChange={(v) => setUiState((p) => ({ ...p, post: { ...p.post, normalize_enabled: v } }))} /></label>
            <div className="grid gap-2">
              <div className="flex items-center justify-between text-sm"><span className="font-medium">Target dBFS</span><span className="font-mono text-xs text-zinc-500">{uiState.post.target_dbfs.toFixed(1)} dB</span></div>
              <Slider value={[uiState.post.target_dbfs]} min={-12} max={0} step={0.5} disabled={!uiState.post.normalize_enabled} onValueChange={(v) => setUiState((p) => ({ ...p, post: { ...p.post, target_dbfs: v[0] } }))} />
            </div>
            <div className="flex items-center gap-2">
              <Button onClick={handleEdit} disabled={!canApplyEdit}>{isEditing ? "Edition..." : "Appliquer"}</Button>
              <Button variant="outline" onClick={() => handleExport(exportTargetId, `vocalie-edition-${exportTargetId ?? "audio"}.wav`, setIsExportingEdited)} disabled={!exportTargetId || isExportingEdited}>{isExportingEdited ? "Export..." : "Exporter"}</Button>
              {isEditing && <div className="flex items-center gap-2 text-xs text-zinc-500"><span className="inline-flex h-3 w-3 animate-spin rounded-full border-2 border-zinc-400 border-t-transparent" /><span>{Math.round(editProgress * 100)}%</span></div>}
            </div>
            {editSummary && <div className="rounded-md border border-zinc-200 bg-white px-3 py-2 text-xs text-zinc-600">Sample rate: {editSummary.sample_rate ?? "--"} Hz · Duree: {editSummary.duration_s ? `${editSummary.duration_s.toFixed(2)} s` : "--"}</div>}
            {editedAudioHref && <div className="rounded-md border border-zinc-200 bg-white p-3"><Waveform src={editedAudioHref} /><audio controls src={editedAudioHref} className="w-full" /></div>}
          </div>
        </Collapsible>
        {/* ── Reglages (Glossaire + Refs vocales) ─────── */}
        <Collapsible title="Reglages" badge={`${glossaryEntries.length} gl · ${refFiles.length} rf`}>
          <div className="grid gap-4">
            {/* Glossaire */}
            <div className="grid gap-3">
              <div className="flex items-center justify-between">
                <p className="text-sm font-semibold">Glossaire</p>
                <div className="flex items-center gap-2">
                  <Switch checked={uiState.preparation.glossary_enabled} onCheckedChange={(v) => setUiState((p) => ({ ...p, preparation: { ...p.preparation, glossary_enabled: v } }))} />
                  <span className="text-xs text-zinc-500">{uiState.preparation.glossary_enabled ? "Actif" : "Inactif"}</span>
                </div>
              </div>
              <div className="grid gap-2">
                <div className="flex items-center gap-2">
                  <Input placeholder="Mot (ex: MJC)" value={newGlossWord} onChange={(e) => setNewGlossWord(e.target.value)} className="flex-1" />
                  <Input placeholder="Prononciation (ex: emjice)" value={newGlossPron} onChange={(e) => setNewGlossPron(e.target.value)} className="flex-1" />
                  <Button size="sm" onClick={handleAddGlossaryEntry} disabled={!newGlossWord.trim() || !newGlossPron.trim()}>Ajouter</Button>
                </div>
                <div className="max-h-32 overflow-auto text-sm">
                  {glossaryEntries.length === 0 ? <p className="text-zinc-400">Aucune entree.</p> : glossaryEntries.map((entry) => (
                    <div key={entry.word} className="flex items-center justify-between border-b border-zinc-100 py-1 last:border-b-0">
                      <span><strong>{entry.word}</strong> → {entry.pronunciation}</span>
                      <button className="text-xs text-red-500 hover:text-red-700" onClick={() => handleDeleteGlossaryEntry(entry.word)}>Suppr.</button>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* References vocales */}
            <div className="grid gap-3">
              <p className="text-sm font-semibold">References vocales</p>
              <div className="flex items-center gap-2">
                <Input value={refDirInput} onChange={(e) => setRefDirInput(e.target.value)} placeholder="/chemin/vers/Ref_audio" className="flex-1" />
                <Button variant="outline" size="sm" onClick={handleSetRefDir}>Appliquer</Button>
              </div>
              <div className="flex items-center gap-2">
                <label className="text-sm text-zinc-600 cursor-pointer">
                  <span className="text-blue-600 underline">Ajouter des fichiers</span>
                  <input type="file" accept=".wav,.mp3,.m4a,.aiff,.flac" multiple className="hidden" onChange={handleUploadRefs} />
                </label>
              </div>
              {refFiles.length > 0 && (
                <div className="max-h-28 overflow-auto text-sm">
                  {refFiles.map((f) => (
                    <div key={f} className="flex items-center justify-between border-b border-zinc-100 py-1 last:border-b-0">
                      <span className="text-zinc-700">{f}</span>
                      <button className="text-xs text-red-400 hover:text-red-600" onClick={() => handleDeleteRef(f)}>×</button>
                    </div>
                  ))}
                </div>
              )}
              <p className="text-xs text-zinc-400">{refDir} · {refFiles.length} fichier{refFiles.length !== 1 ? "s" : ""}</p>
            </div>
          </div>
        </Collapsible>
      </main>
    </div>
  );
}