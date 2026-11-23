// Typed fetch wrapper with one function per backend endpoint.
import type {
  ApiErrorBody,
  DocumentDetail,
  DocumentsResponse,
  EvaluationDataset,
  EvaluationRequest,
  EvaluationResultsResponse,
  EvaluationRunDetail,
  GnnStatus,
  GraphNodeType,
  GraphResponse,
  HealthResponse,
  Job,
  NeighborsResponse,
  OverviewResponse,
  PerformanceResponse,
  QAHistoryResponse,
  QARequest,
  QAResult,
  ReprocessAllResponse,
  RetrievalAnalyticsResponse,
  RuntimeSettings,
  SettingsResponse,
  SettingsUpdateResponse,
  UploadResponse,
} from '../types/api'

/** Base URL of the backend. Empty string = same origin (Vite proxy forwards /api). */
export const API_BASE: string = (import.meta.env.VITE_API_BASE ?? '').replace(/\/+$/, '')

export function apiUrl(path: string): string {
  return `${API_BASE}${path}`
}

export class ApiError extends Error {
  readonly status: number
  readonly detail: string

  constructor(status: number, detail: string) {
    super(detail)
    this.name = 'ApiError'
    this.status = status
    this.detail = detail
  }
}

/** Turn a FastAPI error body into a human-readable message. */
export function formatDetail(body: unknown, fallback: string): string {
  if (body && typeof body === 'object' && 'detail' in body) {
    const detail = (body as ApiErrorBody).detail
    if (typeof detail === 'string') return detail
    if (Array.isArray(detail)) {
      return detail
        .map((d) => {
          const loc = d.loc ? d.loc.filter((l) => l !== 'body').join('.') : ''
          return loc ? `${loc}: ${d.msg ?? 'invalid'}` : (d.msg ?? 'invalid value')
        })
        .join('; ')
    }
  }
  return fallback
}

export function errorMessage(err: unknown): string {
  if (err instanceof ApiError) return err.detail
  if (err instanceof Error) return err.message
  return String(err)
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  let res: Response
  try {
    res = await fetch(apiUrl(path), {
      ...init,
      headers: {
        Accept: 'application/json',
        ...(init.body && !(init.body instanceof FormData) ? { 'Content-Type': 'application/json' } : {}),
        ...init.headers,
      },
    })
  } catch (err) {
    if (err instanceof DOMException && err.name === 'AbortError') throw err
    throw new ApiError(0, 'Cannot reach the GraphRAG-QA backend. Is the API server running on port 8000?')
  }
  const text = await res.text()
  let body: unknown = null
  if (text) {
    try {
      body = JSON.parse(text)
    } catch {
      body = text
    }
  }
  if (!res.ok) {
    const fallback = typeof body === 'string' && body ? body.slice(0, 300) : `${res.status} ${res.statusText}`
    throw new ApiError(res.status, formatDetail(body, fallback))
  }
  return body as T
}

const json = (data: unknown): RequestInit => ({ body: JSON.stringify(data) })

function qs(params: Record<string, string | number | undefined | null>): string {
  const sp = new URLSearchParams()
  for (const [k, v] of Object.entries(params)) {
    if (v !== undefined && v !== null && v !== '') sp.set(k, String(v))
  }
  const s = sp.toString()
  return s ? `?${s}` : ''
}

// ------------------------------------------------------------------- system
export const getHealth = (signal?: AbortSignal) => request<HealthResponse>('/api/health', { signal })
export const getSettings = (signal?: AbortSignal) => request<SettingsResponse>('/api/settings', { signal })
export const updateSettings = (changes: Partial<RuntimeSettings>) =>
  request<SettingsUpdateResponse>('/api/settings', { method: 'PUT', ...json(changes) })
export const reprocessAll = () => request<ReprocessAllResponse>('/api/settings/reprocess-all', { method: 'POST' })
export const getJob = (jobId: string, signal?: AbortSignal) =>
  request<Job>(`/api/jobs/${encodeURIComponent(jobId)}`, { signal })

// ---------------------------------------------------------------- documents
export const listDocuments = (signal?: AbortSignal) => request<DocumentsResponse>('/api/documents', { signal })
export const getDocument = (id: string, signal?: AbortSignal) =>
  request<DocumentDetail>(`/api/documents/${encodeURIComponent(id)}`, { signal })
export const deleteDocument = (id: string) =>
  request<{ deleted: string }>(`/api/documents/${encodeURIComponent(id)}`, { method: 'DELETE' })
export const processDocument = (id: string) =>
  request<{ job_id: string }>(`/api/documents/${encodeURIComponent(id)}/process`, { method: 'POST' })

/**
 * Upload files with XMLHttpRequest so real upload progress can be reported.
 * Multipart field name is `files` (repeated).
 */
export function uploadDocuments(
  files: File[],
  onProgress: (fraction: number) => void,
): { promise: Promise<UploadResponse>; abort: () => void } {
  const xhr = new XMLHttpRequest()
  const promise = new Promise<UploadResponse>((resolve, reject) => {
    const form = new FormData()
    for (const f of files) form.append('files', f, f.name)
    xhr.open('POST', apiUrl('/api/documents/upload'))
    xhr.responseType = 'text'
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable) onProgress(e.loaded / e.total)
    }
    xhr.onload = () => {
      let body: unknown = null
      try {
        body = xhr.responseText ? JSON.parse(xhr.responseText) : null
      } catch {
        body = xhr.responseText
      }
      if (xhr.status >= 200 && xhr.status < 300) {
        onProgress(1)
        resolve(body as UploadResponse)
      } else {
        reject(new ApiError(xhr.status, formatDetail(body, `${xhr.status} ${xhr.statusText}`)))
      }
    }
    xhr.onerror = () => reject(new ApiError(0, 'Network error while uploading. Is the backend running?'))
    xhr.onabort = () => reject(new ApiError(0, 'Upload cancelled'))
    xhr.send(form)
  })
  return { promise, abort: () => xhr.abort() }
}

// ----------------------------------------------------------------------- QA
export const askMultiDocument = (req: QARequest, signal?: AbortSignal) =>
  request<QAResult>('/api/qa/ask', { method: 'POST', signal, ...json({ ...req, mode: 'multi_document' }) })
export const askOpenDomain = (req: QARequest, signal?: AbortSignal) => {
  const { question, strategy, top_k, max_hops, backend } = req
  return request<QAResult>('/api/qa/open-domain', {
    method: 'POST',
    signal,
    ...json({ question, strategy, top_k, max_hops, backend }),
  })
}
export const getQAHistory = (signal?: AbortSignal) => request<QAHistoryResponse>('/api/qa/history', { signal })
export const getQuery = (queryId: string, signal?: AbortSignal) =>
  request<QAResult>(`/api/qa/${encodeURIComponent(queryId)}`, { signal })

// -------------------------------------------------------------------- graph
export interface GraphParams {
  limit?: number
  search?: string
  node_types?: GraphNodeType[]
}
export const getGraph = (p: GraphParams, signal?: AbortSignal) =>
  request<GraphResponse>(
    `/api/graph${qs({ limit: p.limit, search: p.search, node_types: p.node_types?.join(',') })}`,
    { signal },
  )
export const queryGraph = (
  body: { query_id?: string; question?: string; view: 'subgraph' | 'chain' },
  signal?: AbortSignal,
) => request<GraphResponse>('/api/graph/query', { method: 'POST', signal, ...json(body) })
export const getNeighbors = (nodeId: string, signal?: AbortSignal) =>
  request<NeighborsResponse>(`/api/graph/node/${encodeURIComponent(nodeId)}/neighbors`, { signal })

// ---------------------------------------------------------------- analytics
export const getOverview = (signal?: AbortSignal) => request<OverviewResponse>('/api/analytics/overview', { signal })
export const getRetrievalAnalytics = (queryId?: string | null, signal?: AbortSignal) =>
  request<RetrievalAnalyticsResponse>(`/api/analytics/retrieval${qs({ query_id: queryId ?? undefined })}`, { signal })
export const getPerformance = (signal?: AbortSignal) =>
  request<PerformanceResponse>('/api/analytics/performance', { signal })

// --------------------------------------------------------------- evaluation
export const getEvaluationDatasets = (signal?: AbortSignal) =>
  request<{ datasets: EvaluationDataset[] }>('/api/evaluation/datasets', { signal })
export const getEvaluationResults = (signal?: AbortSignal) =>
  request<EvaluationResultsResponse>('/api/evaluation/results', { signal })
export const getEvaluationRun = (runId: string, signal?: AbortSignal) =>
  request<EvaluationRunDetail>(`/api/evaluation/results${qs({ run_id: runId })}`, { signal })
export const startEvaluation = (req: EvaluationRequest) =>
  request<{ run_id: string; job_id: string }>('/api/evaluation/run', { method: 'POST', ...json(req) })
export const evaluationCsvUrl = (runId: string) =>
  apiUrl(`/api/evaluation/results/${encodeURIComponent(runId)}/csv`)

// ---------------------------------------------------------------------- gnn
export const getGnnStatus = (signal?: AbortSignal) => request<GnnStatus>('/api/gnn/status', { signal })
export const trainGnn = (source: string) =>
  request<{ job_id: string }>('/api/gnn/train', { method: 'POST', ...json({ source }) })
