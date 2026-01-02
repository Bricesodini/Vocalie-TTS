"use client";

import { useEffect, useMemo, useState } from "react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import { apiGet, apiPost, assetUrl, fetchVoices } from "@/lib/api";
import type {
  EngineInfo,
  EnginesResponse,
  JobCreateResponse,
  JobStatusResponse,
  VoiceInfo,
  VoicesResponse,
} from "@/lib/types";

const POLL_INTERVAL_MS = 700;

export default function Home() {
  const [engines, setEngines] = useState<EngineInfo[]>([]);
  const [voices, setVoices] = useState<VoiceInfo[]>([]);
  const [engineId, setEngineId] = useState<string>("");
  const [voiceId, setVoiceId] = useState<string>("");
  const [supportsRef, setSupportsRef] = useState(false);
  const [text, setText] = useState("");
  const [status, setStatus] = useState("idle");
  const [progress, setProgress] = useState(0);
  const [assetId, setAssetId] = useState<string | null>(null);
  const [audioHref, setAudioHref] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isGenerating, setIsGenerating] = useState(false);
  const [apiUp, setApiUp] = useState<boolean | null>(null);

  const canGenerate = useMemo(() => {
    if (!engineId || !text.trim()) return false;
    if (supportsRef && !voiceId) return false;
    return !isGenerating;
  }, [engineId, text, supportsRef, voiceId, isGenerating]);

  useEffect(() => {
    let active = true;
    async function loadEngines() {
      try {
        const data = await apiGet<EnginesResponse>("/v1/tts/engines");
        const available = data.engines.filter((engine) => engine.available);
        if (!active) return;
        setEngines(available);
        if (!engineId && available.length > 0) {
          setEngineId(available[0].id);
          if (process.env.NODE_ENV === "development") {
            console.debug("[engines] loaded, selecting", { engineId: available[0].id });
          }
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
  }, []);

  useEffect(() => {
    let active = true;
    async function checkApi() {
      try {
        await apiGet("/v1/health");
        if (active) setApiUp(true);
      } catch {
        if (active) setApiUp(false);
      }
    }
    checkApi();
    return () => {
      active = false;
    };
  }, []);

  useEffect(() => {
    let active = true;
    async function loadVoices() {
      if (!engineId) return;
      if (process.env.NODE_ENV === "development") {
        console.debug("[voices] engine changed, fetching", { engineId });
      }
      const engine = engines.find((item) => item.id === engineId);
      const needsRef = Boolean(engine?.supports_ref);
      setSupportsRef(needsRef);
      if (!needsRef) {
        setVoices([]);
        setVoiceId("");
        return;
      }
      try {
        const data = await fetchVoices(engineId);
        if (!active) return;
        setVoices(data.voices);
        setVoiceId(data.voices[0]?.id ?? "");
      } catch (err) {
        if (!active) return;
        setVoices([]);
        setVoiceId("");
        setError(err instanceof Error ? err.message : "Impossible de charger les voix.");
      }
    }
    loadVoices();
    return () => {
      active = false;
    };
  }, [engineId, engines]);

  async function handleGenerate() {
    if (!canGenerate) return;
    setIsGenerating(true);
    setError(null);
    setStatus("queued");
    setProgress(0);
    setAssetId(null);
    setAudioHref(null);

    try {
      const payload = {
        text,
        engine: engineId,
        voice: supportsRef ? voiceId : null,
        direction: { enabled: false },
      };
      const job = await apiPost<JobCreateResponse>("/v1/tts/jobs", payload);
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

  return (
    <div className="min-h-screen bg-zinc-50 text-zinc-900">
      <main className="mx-auto flex w-full max-w-4xl flex-col gap-6 px-6 py-10">
        <header className="flex flex-col gap-2">
          <Badge variant="outline" className="w-fit">API-driven</Badge>
          <h1 className="text-3xl font-semibold tracking-tight">Chatterbox TTS</h1>
          <p className="text-sm text-zinc-500">
            Interface minimale pour generer un WAV via l&apos;API locale.
          </p>
        </header>

        <Card>
          <CardContent className="flex flex-wrap items-center justify-between gap-3 py-4">
            <div className="flex items-center gap-2 text-sm">
              <span className="font-medium">API status</span>
              <span className={apiUp ? "text-emerald-600" : apiUp === false ? "text-red-600" : "text-zinc-400"}>
                {apiUp ? "up" : apiUp === false ? "down" : "checking..."}
              </span>
            </div>
            <Button
              variant="outline"
              size="sm"
              onClick={async () => {
                setApiUp(null);
                try {
                  await apiGet("/v1/health");
                  setApiUp(true);
                } catch {
                  setApiUp(false);
                }
              }}
            >
              Retry
            </Button>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Generation</CardTitle>
          </CardHeader>
          <CardContent className="flex flex-col gap-4">
            <div className="grid gap-4 md:grid-cols-2">
              <div className="flex flex-col gap-2">
                <span className="text-sm font-medium">Moteur</span>
                <Select value={engineId} onValueChange={setEngineId}>
                  <SelectTrigger>
                    <SelectValue placeholder="Choisir un moteur" />
                  </SelectTrigger>
                  <SelectContent>
                    {engines.map((engine) => (
                      <SelectItem key={engine.id} value={engine.id}>
                        {engine.label}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              {supportsRef ? (
                <div className="flex flex-col gap-2">
                  <span className="text-sm font-medium">Voix (Ref_audio)</span>
                  <Select value={voiceId} onValueChange={setVoiceId}>
                    <SelectTrigger>
                      <SelectValue placeholder="Choisir une reference" />
                    </SelectTrigger>
                    <SelectContent>
                      {voices.map((voice) => (
                        <SelectItem key={voice.id} value={voice.id}>
                          {voice.label}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                  {voices.length === 0 ? (
                    <p className="text-xs text-zinc-500">
                      No reference voices found. Add WAVs in Ref_audio/ then refresh.
                    </p>
                  ) : null}
                </div>
              ) : null}
            </div>

            <div className="flex flex-col gap-2">
              <span className="text-sm font-medium">Texte</span>
              <Textarea
                value={text}
                onChange={(event) => setText(event.target.value)}
                placeholder="Ecrivez votre script..."
                rows={6}
              />
            </div>

            <div className="flex flex-wrap items-center gap-3">
              <Button onClick={handleGenerate} disabled={!canGenerate}>
                {isGenerating ? "Generation..." : "Generer"}
              </Button>
              <div className="text-sm text-zinc-500">
                Statut: <span className="font-medium text-zinc-700">{status}</span>
                {isGenerating ? ` • ${Math.round(progress * 100)}%` : ""}
              </div>
            </div>

            {error ? <p className="text-sm text-red-600">{error}</p> : null}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Sortie</CardTitle>
          </CardHeader>
          <CardContent className="flex flex-col gap-4">
            {audioHref ? (
              <>
                <audio controls src={audioHref} className="w-full" />
                <div className="flex flex-wrap items-center gap-3">
                  <Button asChild>
                    <a href={audioHref} download>
                      Telecharger WAV
                    </a>
                  </Button>
                  {assetId ? (
                    <span className="text-xs text-zinc-500">asset_id: {assetId}</span>
                  ) : null}
                </div>
              </>
            ) : (
              <p className="text-sm text-zinc-500">Aucun audio disponible pour le moment.</p>
            )}
          </CardContent>
        </Card>
      </main>
    </div>
  );
}
