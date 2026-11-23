import { useEffect, useMemo, useState, type FormEvent } from 'react'
import { BrainCircuit, Cpu, HardDrive, RefreshCw, RotateCcw, Save, SlidersHorizontal, Sparkles } from 'lucide-react'
import { useApi } from '../hooks/useApi'
import { useJobTracker } from '../hooks/useJobs'
import { errorMessage, getSettings, reprocessAll, updateSettings } from '../services/api'
import type { RuntimeSettings, SettingsResponse, SettingsUpdateResponse } from '../types/api'
import { PageHeader, KeyValue } from '../components/PageHeader'
import { Card } from '../components/Card'
import { Badge, StatusBadge } from '../components/Badge'
import { EmptyState, ErrorBanner, ProgressBar, SkeletonBlock, Spinner, WarningPanel } from '../components/Feedback'
import { BACKEND_LABELS } from '../lib/constants'
import { fmtBytes, fmtNumber, fmtValue, humanize } from '../lib/format'

const NUMERIC_FIELDS: Array<{ key: keyof RuntimeSettings; label: string; hint: string; step?: number; min?: number; max?: number }> = [
  { key: 'chunk_size', label: 'Chunk size (tokens)', hint: 'Token budget per passage. Changing it requires reprocessing documents.', min: 50, step: 10 },
  { key: 'chunk_overlap', label: 'Chunk overlap (tokens)', hint: 'Tokens shared between consecutive passages.', min: 0, step: 10 },
  { key: 'top_k', label: 'Top-K passages', hint: 'Default number of passages retrieved per question.', min: 1, max: 50 },
  { key: 'max_hops', label: 'Max hops', hint: 'Maximum reasoning-chain length in the evidence graph.', min: 1, max: 4 },
  { key: 'hybrid_alpha', label: 'Hybrid α', hint: 'Weight of semantic vs BM25 scores in hybrid fusion (1 = semantic only).', min: 0, max: 1, step: 0.05 },
  { key: 'max_evidence_passages', label: 'Max evidence passages', hint: 'Passages passed to the generator as evidence.', min: 1, max: 20 },
]

export default function SettingsPage() {
  const settings = useApi((s) => getSettings(s), [])
  const s = settings.data
  return (
    <>
      <PageHeader
        title="Settings"
        description="Runtime configuration, model availability and local storage."
        actions={
          <button type="button" className="btn-secondary" onClick={() => void settings.refetch()} disabled={settings.loading}>
            <RefreshCw size={14} className={settings.loading ? 'animate-spin' : ''} /> Refresh
          </button>
        }
      />
      {settings.error && <ErrorBanner message={settings.error} onRetry={() => void settings.refetch()} className="mb-6" />}
      {settings.loading && !s ? (
        <div className="grid grid-cols-1 gap-6 xl:grid-cols-2">
          {Array.from({ length: 4 }, (_, i) => (
            <Card key={i} title=" ">
              <SkeletonBlock rows={5} />
            </Card>
          ))}
        </div>
      ) : s ? (
        <div className="grid grid-cols-1 gap-6 xl:grid-cols-2">
          <RuntimeForm settings={s} onSaved={() => void settings.refetch(true)} />
          <div className="space-y-6">
            <GeneratorStatus settings={s} />
            <GnnStatusCard settings={s} />
          </div>
          <ModelsCard settings={s} />
          <StorageCard settings={s} />
        </div>
      ) : null}
    </>
  )
}

function RuntimeForm({ settings, onSaved }: { settings: SettingsResponse; onSaved: () => void }) {
  const [draft, setDraft] = useState<RuntimeSettings>(settings.runtime)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<SettingsUpdateResponse | null>(null)
  const [reprocessing, setReprocessing] = useState(false)
  const [reprocessError, setReprocessError] = useState<string | null>(null)
  const tracker = useJobTracker()

  useEffect(() => setDraft(settings.runtime), [settings.runtime])

  const changed = useMemo(() => {
    const out: Partial<RuntimeSettings> = {}
    for (const k of Object.keys(draft) as Array<keyof RuntimeSettings>) {
      if (draft[k] !== settings.runtime[k]) (out as Record<string, unknown>)[k] = draft[k]
    }
    return out
  }, [draft, settings.runtime])
  const dirty = Object.keys(changed).length > 0

  const save = async (e: FormEvent) => {
    e.preventDefault()
    if (!dirty) return
    setSaving(true)
    setError(null)
    setResult(null)
    try {
      const res = await updateSettings(changed)
      setResult(res)
      if (res.reindex_job) tracker.track(res.reindex_job, 'Rebuild FAISS index')
      onSaved()
    } catch (err) {
      setError(errorMessage(err))
    } finally {
      setSaving(false)
    }
  }

  const reprocess = async () => {
    setReprocessing(true)
    setReprocessError(null)
    try {
      const res = await reprocessAll()
      res.jobs.forEach((id, i) => tracker.track(id, `Reprocess document ${i + 1}/${res.jobs.length}`))
    } catch (err) {
      setReprocessError(errorMessage(err))
    } finally {
      setReprocessing(false)
    }
  }

  const set = <K extends keyof RuntimeSettings>(k: K, v: RuntimeSettings[K]) => setDraft((d) => ({ ...d, [k]: v }))

  return (
    <Card title="Runtime settings" subtitle="Only changed fields are sent (PUT /api/settings)" icon={<SlidersHorizontal size={16} />}>
      <form onSubmit={save} className="space-y-4">
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
          {NUMERIC_FIELDS.map((f) => {
            const isChanged = f.key in changed
            const def = settings.defaults[f.key]
            return (
              <div key={f.key}>
                <label className="label flex items-center justify-between" htmlFor={f.key}>
                  <span>
                    {f.label} {isChanged && <span className="text-indigo-300">• modified</span>}
                  </span>
                  <span className="text-subtle">default {String(def)}</span>
                </label>
                <input
                  id={f.key}
                  type="number"
                  className="input"
                  min={f.min}
                  max={f.max}
                  step={f.step ?? 1}
                  value={String(draft[f.key])}
                  onChange={(e) => set(f.key, (e.target.value === '' ? 0 : Number(e.target.value)) as never)}
                />
                <p className="mt-1 text-[11px] text-subtle">{f.hint}</p>
              </div>
            )
          })}
          <div>
            <label className="label" htmlFor="generator_backend">
              Default generator backend {'generator_backend' in changed && <span className="text-indigo-300">• modified</span>}
            </label>
            <select id="generator_backend" className="input" value={draft.generator_backend} onChange={(e) => set('generator_backend', e.target.value as RuntimeSettings['generator_backend'])}>
              {settings.options.generator_backends.map((b) => {
                const unavailable = b !== 'auto' && settings.generator_status[b] !== undefined && !settings.generator_status[b].available
                return (
                  <option key={b} value={b} disabled={unavailable}>
                    {BACKEND_LABELS[b] ?? b}
                    {unavailable ? ' — unavailable' : ''}
                  </option>
                )
              })}
            </select>
          </div>
          <div>
            <label className="label" htmlFor="embedding_model">
              Embedding model {'embedding_model' in changed && <span className="text-indigo-300">• modified</span>}
            </label>
            <select id="embedding_model" className="input" value={draft.embedding_model} onChange={(e) => set('embedding_model', e.target.value)}>
              {settings.options.embedding_models.map((m) => (
                <option key={m} value={m}>
                  {m}
                </option>
              ))}
            </select>
            <p className="mt-1 text-[11px] text-subtle">Changing the embedding model rebuilds the FAISS index and invalidates the GNN checkpoint.</p>
          </div>
        </div>

        {error && <ErrorBanner title="Settings rejected" message={error} onDismiss={() => setError(null)} />}

        <div className="flex flex-wrap items-center justify-end gap-2 border-t border-line pt-4">
          <button type="button" className="btn-ghost" disabled={!dirty || saving} onClick={() => setDraft(settings.runtime)}>
            Discard
          </button>
          <button type="button" className="btn-secondary" disabled={saving} onClick={() => setDraft(settings.defaults)} title="Fill the form with default values (not saved until you click Save)">
            <RotateCcw size={14} /> Defaults
          </button>
          <button type="submit" className="btn-primary" disabled={!dirty || saving}>
            {saving ? <Spinner size={14} className="text-white" /> : <Save size={14} />} Save {dirty ? `(${Object.keys(changed).length})` : ''}
          </button>
        </div>

        {result && (
          <div className="space-y-3">
            <p className="text-sm text-emerald-300">
              Saved{result.changed.length ? `: ${result.changed.map(humanize).join(', ')}` : ' — no effective changes'}.
            </p>
            {result.requires_reprocessing && (
              <WarningPanel title="Documents must be reprocessed">
                <p>Chunking settings changed. Existing passages, entities and graph edges were built with the old settings.</p>
                <button type="button" className="btn-primary btn-sm mt-2" onClick={() => void reprocess()} disabled={reprocessing}>
                  {reprocessing ? <Spinner size={12} className="text-white" /> : <RefreshCw size={12} />} Reprocess all documents
                </button>
                {reprocessError && <p className="mt-2 text-xs text-rose-300">{reprocessError}</p>}
              </WarningPanel>
            )}
            {result.requires_reindex && <p className="text-xs text-muted">The FAISS index is being rebuilt with the new embedding model.</p>}
          </div>
        )}

        {tracker.jobs.length > 0 && (
          <ul className="space-y-2">
            {tracker.jobs.map((j) => (
              <li key={j.id}>
                <div className="mb-1 flex justify-between text-xs">
                  <span className="text-muted">{j.label}</span>
                  <StatusBadge status={j.error ? 'error' : (j.job?.status ?? 'queued')} />
                </div>
                <ProgressBar value={j.job?.progress} indeterminate={!j.job} />
                {(j.error || j.job?.message) && <p className="mt-0.5 text-[11px] text-subtle">{j.error ?? j.job?.message}</p>}
              </li>
            ))}
          </ul>
        )}
      </form>
    </Card>
  )
}

function GeneratorStatus({ settings }: { settings: SettingsResponse }) {
  const entries = Object.entries(settings.generator_status)
  return (
    <Card title="Generator backends" icon={<Sparkles size={16} />}>
      {entries.length === 0 ? (
        <EmptyState title="No generator information" />
      ) : (
        <ul className="divide-y divide-line">
          {entries.map(([name, st]) => (
            <li key={name} className="flex items-start gap-3 py-2.5 first:pt-0 last:pb-0">
              <div className="min-w-0 flex-1">
                <p className="text-sm font-medium text-fg">{BACKEND_LABELS[name] ?? name}</p>
                <p className="truncate text-xs text-subtle">
                  {[st.model, st.description, st.url].filter(Boolean).join(' · ') || '—'}
                  {st.loaded !== undefined && <> · {st.loaded ? 'loaded' : 'not loaded'}</>}
                </p>
                {st.error && <p className="text-xs text-rose-300">{st.error}</p>}
              </div>
              <Badge tone={st.available ? 'success' : 'neutral'} dot>
                {st.available ? 'available' : 'unavailable'}
              </Badge>
            </li>
          ))}
        </ul>
      )}
    </Card>
  )
}

function GnnStatusCard({ settings }: { settings: SettingsResponse }) {
  const g = settings.gnn_status
  const meta = Object.entries(g.meta ?? {}).filter(([, v]) => typeof v !== 'object' || v === null)
  return (
    <Card title="GNN reranker" icon={<BrainCircuit size={16} />} actions={<StatusBadge status={g.available ? 'ready' : g.checkpoint_exists ? 'degraded' : 'untrained'} />}>
      <KeyValue
        items={[
          ['Status', g.reason ?? '—'],
          ['Architecture', g.architecture ?? '—'],
          ['Parameters', fmtNumber(g.num_parameters)],
          ['Feature spec', g.feature_spec_version ?? '—'],
          ['Checkpoint', g.checkpoint_path ? <span className="mono break-all">{g.checkpoint_path}</span> : '—'],
          ['Checkpoint exists', g.checkpoint_exists ? 'yes' : 'no'],
          ...meta.map(([k, v]) => [humanize(k), fmtValue(v)] as [string, string]),
        ]}
      />
    </Card>
  )
}

function ModelsCard({ settings }: { settings: SettingsResponse }) {
  return (
    <Card title="Models" icon={<Cpu size={16} />}>
      <KeyValue items={Object.entries(settings.models).map(([k, v]) => [humanize(k), v ? <span className="mono">{v}</span> : '—'])} />
      <div className="mt-4 border-t border-line pt-3">
        <p className="label">Upload limits</p>
        <p className="text-sm text-fg">
          {settings.limits.max_upload_mb} MB per file · {settings.limits.allowed_extensions.join(', ')}
        </p>
      </div>
    </Card>
  )
}

function StorageCard({ settings }: { settings: SettingsResponse }) {
  const bytes = Object.entries(settings.storage.bytes)
  const total = bytes.reduce((a, [, v]) => a + v, 0)
  const max = Math.max(1, ...bytes.map(([, v]) => v))
  return (
    <Card title="Local storage" subtitle={`${fmtBytes(total)} total`} icon={<HardDrive size={16} />}>
      <div className="space-y-2">
        {bytes
          .sort((a, b) => b[1] - a[1])
          .map(([k, v]) => (
            <div key={k} className="flex items-center gap-3 text-xs">
              <span className="w-20 shrink-0 text-muted">{humanize(k)}</span>
              <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-surface-3">
                <div className="h-full rounded-full bg-indigo-400/80" style={{ width: `${v > 0 ? Math.max(1, (v / max) * 100) : 0}%` }} />
              </div>
              <span className="w-16 shrink-0 text-right text-fg tabular-nums">{fmtBytes(v)}</span>
            </div>
          ))}
      </div>
      <div className="mt-4 border-t border-line pt-3">
        <p className="label">Paths</p>
        <KeyValue items={Object.entries(settings.storage.paths).map(([k, v]) => [humanize(k), <span className="mono break-all text-muted">{v}</span>])} />
      </div>
    </Card>
  )
}
