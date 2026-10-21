/**
 * lib/api.ts
 * ──────────
 * Cliente del backend. Todas las llamadas van a /api en el mismo origen:
 * en desarrollo lo proxea Vite, en producción lo proxea server.ts.
 */

export type WebMap = {
  id: string;
  titulo: string;
  snippet: string;
  propietario: string;
  tags: string[];
  url_item: string;
  vectorizado: boolean;
  embedding: string;
};

export type Source = {
  capa: string;
  url: string;
  registros: number;
  geometria: string;
  similitud: number;
};

export type Step = { tool: string; input: string; output: string };

export type ChatReply = {
  success: boolean;
  answer: string;
  agent_used: string;
  steps_count: number;
  steps: Step[];
  duration_ms: number;
  geojson: GeoJSON.FeatureCollection | null;
  sources: Source[];
  map_action: { type: string; url?: string; title?: string; features?: number } | null;
  plan?: { agente: string; tarea: string }[];
  metrics?: {
    llm_calls?: number; prompt_tokens?: number; completion_tokens?: number;
    duration_ms?: number; features?: number;
  };
};

export type Status = {
  arcgis: { ok: boolean; usuario?: string; error?: string };
  llm: {
    ok: boolean; provider?: string; label?: string; model?: string;
    full?: string; local?: boolean; source?: string; error?: string;
  };
  database: { ok: boolean; postgres?: string; error?: string };
  vectors: {
    ok: boolean; modo?: string; colecciones?: number; documentos?: number;
    embeddings?: { provider?: string; model?: string }; error?: string;
  };
};

export type Provider = {
  name: string; label: string; env_key: string; signup_url: string;
  default_model: string; local: boolean; example: string; ready: boolean;
};

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`/api${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...init,
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail ?? `${res.status} ${res.statusText}`);
  }
  return res.json() as Promise<T>;
}

export const api = {
  status: () => req<Status>('/status'),

  providers: () => req<{ providers: Provider[] }>('/llm-providers'),

  ollamaModels: () => req<{ models: string[] }>('/ollama-models'),

  selectModel: (model: string) =>
    req<{ ok: boolean; full: string }>('/llm-select', {
      method: 'POST',
      body: JSON.stringify({ model }),
    }),

  webmaps: () => req<{ total: number; webmaps: WebMap[] }>('/webmaps'),

  vectorize: (webmap_id: string, deep = true) =>
    req<{ ok: boolean; titulo: string; capas: number; documentos: number; embedding: string }>(
      '/ingest-webmap',
      { method: 'POST', body: JSON.stringify({ webmap_id, deep }) },
    ),

  chat: (message: string, webmap_id: string | null) =>
    req<ChatReply>('/chat', {
      method: 'POST',
      body: JSON.stringify({ message, webmap_id }),
    }),
};
