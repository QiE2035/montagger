/** Typed surface of the montagger API. */

export interface Tag {
  name: string;
  category: string;
  confidence: number;
}

export type JobStatus = "queued" | "running" | "done" | "error" | "lost" | "canceled";

export interface Job {
  id: string;
  filename: string;
  source: string;
  model: string;
  provider?: string;
  status: JobStatus;
  error?: string;
  sha256?: string;
  md5?: string;
  bytes?: number;
  width?: number;
  height?: number;
  elapsed_ms?: number;
  tags?: Tag[];
  rating?: string | null;
  monbooru_id?: number | null;
  created_at?: string;
  finished_at?: string | null;
  dedup?: boolean;
}

export interface Thresholds {
  global: number;
  categories: Record<string, number>;
  top_k: Record<string, number>;
  disabled: string[];
}

export interface ModelStatus {
  name: string;
  description: string;
  gated: boolean;
  in_catalog: boolean;
  discovered: boolean;
  available: boolean;
  reason: string;
  size_bytes: number;
  default_threshold: number;
  default_thresholds: Record<string, number>;
  default_top_k: Record<string, number>;
  emitted_categories: string[];
  effective: Thresholds;
}

export interface ModelsResponse {
  default: string;
  provider: string;
  provider_short: string;
  available_providers: string[];
  default_models: string[];
  models: ModelStatus[];
}

export interface JobsResponse {
  jobs: Job[];
  total: number;
  offset: number;
  limit: number;
}

export interface Settings {
  server: { bind_address: string; base_url: string };
  auth: { has_password: boolean; session_days: number };
  models: {
    path: string;
    default: string;
    default_models: string[];
    execution_provider: string;
    device_id: number;
    intra_op_threads: number;
    max_upload_mb: number;
    disabled_categories: string[];
    idle_unload_min: number;
    max_loaded: number;
    isolated: boolean;
    available_providers: string[];
    running_provider: string;
    model_root: string;
    loaded: { name: string; idle_s: number | null }[];
  };
  queue: { max_pending: number; history_days: number };
  hf: { endpoint: string; has_token: boolean };
  monbooru: {
    api_url: string;
    web_url: string;
    paired: boolean;
    push_tags: boolean;
    push_images: boolean;
    gallery: string;
  };
  log: { debug: boolean };
}

export interface Token {
  id: number;
  name: string;
  token: string;
  created_at: string;
}

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

async function handle<T>(resp: Response): Promise<T> {
  if (!resp.ok) {
    let detail = `${resp.status}`;
    try {
      const body = await resp.json();
      detail = body.detail ?? detail;
    } catch {
      /* non-JSON error body */
    }
    throw new ApiError(resp.status, detail);
  }
  return resp.json() as Promise<T>;
}

export const api = {
  get<T>(url: string): Promise<T> {
    return fetch(url).then((r) => handle<T>(r));
  },
  post<T>(url: string, body?: unknown): Promise<T> {
    return fetch(url, {
      method: "POST",
      headers: body !== undefined ? { "Content-Type": "application/json" } : undefined,
      body: body !== undefined ? JSON.stringify(body) : undefined,
    }).then((r) => handle<T>(r));
  },
  del<T>(url: string): Promise<T> {
    return fetch(url, { method: "DELETE" }).then((r) => handle<T>(r));
  },

  models: () => api.get<ModelsResponse>("/api/v1/models"),
  jobs: (limit = 50, offset = 0, search = "", status = "") => {
    const p = new URLSearchParams({ limit: String(limit), offset: String(offset) });
    if (search) p.set("q", search);
    if (status) p.set("status", status);
    return api.get<JobsResponse>(`/api/v1/jobs?${p}`);
  },
  job: (id: string) => api.get<Job>(`/api/v1/jobs/${id}`),
  deleteJob: (id: string) => api.del<{ ok: boolean }>(`/api/v1/jobs/${id}`),
  cancelJob: (id: string) => api.post<{ ok: boolean }>(`/api/v1/jobs/${id}/cancel`),
  batchJobs: (body: {
    action: "delete" | "cancel";
    ids?: string[];
    all_done?: boolean;
    all_queued?: boolean;
  }) => api.post<{ ok: boolean; affected: number }>("/api/v1/jobs/batch", body),
  clearFinished: () => api.post<{ deleted: number }>("/api/v1/jobs/clear"),
  purgeHistory: () => api.post<{ deleted: number }>("/api/v1/history/purge"),
  releaseMemory: (models?: string[]) =>
    api.post<{
      ok: boolean;
      unloaded: string[];
      jobs: { jobs_forgotten: number; jobs_kept: number };
      rss_mb: number;
      malloc_trim: boolean;
    }>("/api/v1/memory/release", models && models.length ? { models } : {}),
  removeModel: (name: string) => api.del<{ ok: boolean }>(`/api/v1/models/${name}`),
  installModel: (name: string) => api.post<{ ok: boolean; path: string }>(`/api/v1/models/${name}/install`),
  settings: () => api.get<Settings>("/api/v1/settings"),
  saveSettings: (patch: object) => api.post<{ ok: boolean }>("/api/v1/settings", patch),
  tokens: () => api.get<Token[]>("/api/v1/tokens"),
  createToken: (name: string) => api.post<{ token: string }>("/api/v1/tokens", { name }),
  deleteToken: (id: number) => api.del<{ ok: boolean }>(`/api/v1/tokens/${id}`),
  authState: () => api.get<{ guard_active: boolean; authenticated: boolean; open_lan: boolean }>("/api/v1/auth/state"),
  login: (password: string) => api.post<{ ok: boolean }>("/api/v1/auth/login", { password }),
  logout: () => api.post<{ ok: boolean }>("/api/v1/auth/logout"),
  monbooruStatus: () =>
    api.get<{
      api_url: string;
      paired: boolean;
      waiting: boolean;
      push_tags: boolean;
      push_images: boolean;
      gallery: string;
    }>("/api/v1/monbooru/status"),
  monbooruPair: (apiUrl: string) => api.post<{ ok: boolean }>("/api/v1/monbooru/pair", { api_url: apiUrl }),
  monbooruUnpair: () => api.post<{ ok: boolean }>("/api/v1/monbooru/unpair"),
  monbooruGalleries: () => api.get<{ galleries: { name: string; images: number; active: boolean }[] }>(
    "/api/v1/monbooru/galleries",
  ),
};

/** Upload one file as a raw byte stream; onprogress reports network bytes.
 * Multi-model uploads answer with one job per model (array). */
export function uploadFile(
  file: File,
  model: string | null,
  onprogress: (sent: number, total: number) => void,
): Promise<Job | Job[]> {
  return new Promise((resolve, reject) => {
    const params = new URLSearchParams({ filename: file.name });
    if (model) params.set("model", model);
    const xhr = new XMLHttpRequest();
    xhr.open("POST", `/api/v1/tag?${params}`);
    xhr.setRequestHeader("Content-Type", file.type || "application/octet-stream");
    xhr.upload.onprogress = (e) => onprogress(e.loaded, e.total);
    xhr.onload = () => {
      try {
        const body = JSON.parse(xhr.responseText);
        if (xhr.status === 202 || xhr.status === 200) resolve(body as Job | Job[]);
        else reject(new ApiError(xhr.status, body.detail ?? `${xhr.status}`));
      } catch {
        reject(new ApiError(xhr.status, `${xhr.status}`));
      }
    };
    xhr.onerror = () => reject(new ApiError(0, "网络错误"));
    xhr.send(file);
  });
}

/** Single shared SSE connection; Naive UI message toasts surface drops. */
export function connectEvents(onJob: (job: Job) => void, onDown?: () => void): EventSource {
  const es = new EventSource("/api/v1/events");
  es.addEventListener("open", () => undefined);
  es.addEventListener("error", () => onDown?.());
  es.addEventListener("message", (ev) => {
    try {
      const payload = JSON.parse((ev as MessageEvent).data);
      if (payload.event === "job") onJob(payload.job as Job);
    } catch {
      /* malformed frame */
    }
  });
  return es;
}
