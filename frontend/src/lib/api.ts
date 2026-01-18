import type { VoicesResponse } from "@/lib/types";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "";

function resolveBase() {
  if (API_BASE) return API_BASE;
  return "";
}

function withQuery(path: string, params?: Record<string, string | number | boolean | undefined>) {
  const basePath = path.startsWith("/") ? path : `/${path}`;
  if (!params || Object.keys(params).length === 0) {
    return basePath;
  }
  const url = new URL(basePath, typeof window !== "undefined" ? window.location.origin : "http://127.0.0.1");
  for (const [key, value] of Object.entries(params)) {
    if (value === undefined || value === null || value === "") continue;
    url.searchParams.set(key, String(value));
  }
  return `${url.pathname}${url.search}`;
}

function buildUrl(path: string, params?: Record<string, string | number | boolean | undefined>) {
  const base = resolveBase();
  const withParams = withQuery(path, params);
  if (!base) {
    return withParams;
  }
  const url = new URL(withParams, base);
  return url.toString();
}

export async function apiGet<T>(path: string, params?: Record<string, string | number | boolean | undefined>): Promise<T> {
  const url = buildUrl(path, params);
  if (process.env.NODE_ENV === "development") {
    console.debug("[api] GET", { url });
  }
  const resp = await fetch(url, { cache: "no-store" });
  if (!resp.ok) {
    throw new Error(`${resp.status} ${resp.statusText}`);
  }
  return (await resp.json()) as T;
}

export async function apiPost<T>(path: string, body: unknown): Promise<T> {
  const resp = await fetch(buildUrl(path), {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(body),
  });
  if (!resp.ok) {
    const text = await resp.text();
    throw new Error(text || `${resp.status} ${resp.statusText}`);
  }
  return (await resp.json()) as T;
}

export async function apiPostForm<T>(path: string, body: FormData): Promise<T> {
  const resp = await fetch(buildUrl(path), {
    method: "POST",
    body,
  });
  if (!resp.ok) {
    const text = await resp.text();
    throw new Error(text || `${resp.status} ${resp.statusText}`);
  }
  return (await resp.json()) as T;
}

export async function apiPut<T>(path: string, body: unknown): Promise<T> {
  const resp = await fetch(buildUrl(path), {
    method: "PUT",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(body),
  });
  if (!resp.ok) {
    const text = await resp.text();
    throw new Error(text || `${resp.status} ${resp.statusText}`);
  }
  return (await resp.json()) as T;
}

export async function apiDelete<T>(path: string): Promise<T> {
  const resp = await fetch(buildUrl(path), { method: "DELETE" });
  if (!resp.ok) {
    const text = await resp.text();
    throw new Error(text || `${resp.status} ${resp.statusText}`);
  }
  return (await resp.json()) as T;
}

export class MissingEngineIdError extends Error {
  constructor() {
    super("engine_id_required");
    this.name = "MissingEngineIdError";
  }
}

export async function fetchVoices(engineId: string): Promise<VoicesResponse> {
  if (!engineId) {
    if (process.env.NODE_ENV === "development") {
      console.warn("[voices] missing engineId", { engineId });
      console.trace("[voices] callsite");
    }
    return { engine: "", voices: [] };
  }
  if (process.env.NODE_ENV === "development") {
    console.debug("[voices] fetch", { engineId, url: `/v1/tts/voices?engine=${engineId}` });
  }
  return apiGet("/v1/tts/voices", { engine: engineId });
}

export function assetUrl(assetId: string): string {
  return buildUrl(`/v1/assets/${assetId}`);
}
