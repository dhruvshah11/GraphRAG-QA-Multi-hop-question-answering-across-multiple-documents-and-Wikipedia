import { useState } from 'react'
import { useSearchParams } from 'react-router-dom'
import { Database, FlaskConical, History, RefreshCw } from 'lucide-react'
import { useApi } from '../hooks/useApi'
import { useJob } from '../hooks/useJobs'
import { usePolling } from '../hooks/usePolling'
import { errorMessage, getEvaluationDatasets, getEvaluationResults, getEvaluationRun, getSettings, startEvaluation } from '../services/api'
import type { EvaluationRequest } from '../types/api'
import { PageHeader } from '../components/PageHeader'
import { Card } from '../components/Card'
import { Badge, StatusBadge } from '../components/Badge'
import { EmptyState, ErrorBanner, ProgressBar, SkeletonBlock } from '../components/Feedback'
import { RunForm } from '../components/evaluation/RunForm'
import { RunDetail } from '../components/evaluation/RunDetail'
import { STRATEGY_SHORT } from '../lib/constants'
import { cx, fmtDate, fmtPct, fmtScore } from '../lib/format'

export default function EvaluationPage() {
  const [params, setParams] = useSearchParams()
  const runParam = params.get('run')
  const datasets = useApi((s) => getEvaluationDatasets(s), [])
  const results = useApi((s) => getEvaluationResults(s), [])
  const settings = useApi((s) => getSettings(s), [])
  const [dataset, setDataset] = useState('demo')
  const [starting, setStarting] = useState(false)
  const [startError, setStartError] = useState<string | null>(null)
  const [jobId, setJobId] = useState<string | null>(null)
  const [startedRunId, setStartedRunId] = useState<string | null>(null)

  const runs = results.data?.runs ?? []
  const selectedRunId = runParam ?? runs[0]?.id ?? null
  const detail = useApi((s) => getEvaluationRun(selectedRunId as string, s), [selectedRunId], selectedRunId !== null)

  // Runs started elsewhere (CLI or another tab) are tracked through the results endpoint's progress field;
  // only runs started from this page are polled through GET /api/jobs/{job_id}.
  const running = runs.find((r) => r.status === 'running' || r.status === 'queued')

  const { job } = useJob(jobId, () => {
    void results.refetch(true)
    void detail.refetch(true)
  })
  const active = job ? job.status === 'running' || job.status === 'queued' : !!running
  usePolling(
    () => {
      void results.refetch(true)
      if (selectedRunId && (detail.data?.run.status === 'running' || detail.data?.run.status === 'queued')) void detail.refetch(true)
    },
    4000,
    active,
  )

  const start = async (req: EvaluationRequest) => {
    setStarting(true)
    setStartError(null)
    try {
      const res = await startEvaluation(req)
      setJobId(res.job_id)
      setStartedRunId(res.run_id)
      setParams({ run: res.run_id })
      void results.refetch(true)
    } catch (e) {
      setStartError(errorMessage(e))
    } finally {
      setStarting(false)
    }
  }

  const dsList = datasets.data?.datasets ?? results.data?.datasets ?? []

  return (
    <>
      <PageHeader
        title="Evaluation Lab"
        description="Benchmark FAISS, BM25, hybrid, graph-heuristic and GNN reranking on answer quality (EM, F1), retrieval (P@K, R@K, MRR), evidence recall, citations and abstention."
        actions={
          <button type="button" className="btn-secondary" onClick={() => void results.refetch()} disabled={results.loading}>
            <RefreshCw size={14} className={results.loading ? 'animate-spin' : ''} /> Refresh
          </button>
        }
      />

      {datasets.error && <ErrorBanner message={datasets.error} onRetry={() => void datasets.refetch()} className="mb-6" />}

      {/* dataset cards */}
      <div className="mb-6 grid gap-3 md:grid-cols-3">
        {datasets.loading && !datasets.data
          ? Array.from({ length: 3 }, (_, i) => (
              <div key={i} className="rounded-xl border border-line bg-surface p-4">
                <SkeletonBlock rows={3} />
              </div>
            ))
          : dsList.map((d) => (
              <button
                key={d.id}
                type="button"
                disabled={!d.available}
                onClick={() => setDataset(d.id)}
                className={cx(
                  'rounded-xl border bg-surface p-4 text-left transition-colors disabled:opacity-50',
                  dataset === d.id ? 'border-indigo-500/60 ring-1 ring-indigo-500/30' : 'border-line hover:border-line-strong',
                )}
              >
                <div className="flex items-start justify-between gap-2">
                  <div className="flex items-center gap-2">
                    <Database size={15} className="text-accent-strong" />
                    <span className="mono text-muted">{d.id}</span>
                  </div>
                  <Badge tone={d.available ? 'success' : 'neutral'}>{d.available ? `${d.questions} questions` : 'unavailable'}</Badge>
                </div>
                <p className="mt-2 text-sm font-semibold text-fg">{d.name}</p>
                <p className="mt-1 line-clamp-4 text-xs leading-relaxed text-muted" title={d.description}>
                  {d.description}
                </p>
              </button>
            ))}
      </div>

      <div className="grid grid-cols-1 gap-6 xl:grid-cols-[360px_minmax(0,1fr)]">
        <div className="space-y-6">
          <Card title="Run configuration" icon={<FlaskConical size={16} />}>
            {dsList.length ? (
              <RunForm datasets={dsList} dataset={dataset} onDatasetChange={setDataset} settings={settings.data} busy={starting || active} onSubmit={(r) => void start(r)} />
            ) : (
              <SkeletonBlock rows={6} />
            )}
            {active && <p className="mt-3 text-xs text-amber-300/90">An evaluation is already running; wait for it to finish before starting another.</p>}
            {startError && <ErrorBanner message={startError} onDismiss={() => setStartError(null)} className="mt-3" />}
          </Card>

          <Card title="Runs" icon={<History size={16} />} flush>
            {results.error ? (
              <ErrorBanner message={results.error} onRetry={() => void results.refetch()} className="m-3" />
            ) : results.loading && !results.data ? (
              <div className="p-4">
                <SkeletonBlock rows={4} />
              </div>
            ) : runs.length === 0 ? (
              <EmptyState title="No evaluation runs yet — start one" description="Configure a run above. The demo dataset (12 questions) finishes in about a minute per strategy." />
            ) : (
              <ul className="max-h-[520px] divide-y divide-line overflow-y-auto">
                {runs.map((r) => {
                  const bs = r.summary?.by_strategy ?? {}
                  const bestF1 = Math.max(-1, ...Object.values(bs).map((m) => m.f1 ?? -1))
                  return (
                    <li key={r.id}>
                      <button
                        type="button"
                        onClick={() => setParams({ run: r.id })}
                        className={cx('w-full px-4 py-3 text-left transition-colors hover:bg-surface-2/60', selectedRunId === r.id && 'bg-accent-soft')}
                      >
                        <div className="flex items-center justify-between gap-2">
                          <span className="mono truncate text-fg">{r.id}</span>
                          <StatusBadge status={r.status} />
                        </div>
                        <div className="mt-1 flex flex-wrap items-center gap-x-2 text-[11px] text-subtle">
                          <span className="text-muted">{r.dataset}</span>
                          <span>{fmtDate(r.created_at)}</span>
                          <span>{(r.config.strategies ?? []).map((s) => STRATEGY_SHORT[s] ?? s).join(', ')}</span>
                          {bestF1 >= 0 && <span>best F1 {fmtScore(bestF1, 2)}</span>}
                        </div>
                        {(r.status === 'running' || r.status === 'queued') && (
                          <div className="mt-2">
                            <ProgressBar value={job && r.id === startedRunId ? job.progress : r.progress} />
                            <p className="mt-1 text-[11px] text-subtle">{job && r.id === startedRunId && job.message ? job.message : `${fmtPct(r.progress, 0)} complete`}</p>
                          </div>
                        )}
                        {r.error && <p className="mt-1 line-clamp-2 text-[11px] text-rose-300">{r.error}</p>}
                      </button>
                    </li>
                  )
                })}
              </ul>
            )}
          </Card>
        </div>

        <div className="min-w-0">
          {detail.error ? (
            <ErrorBanner message={detail.error} onRetry={() => void detail.refetch()} />
          ) : detail.loading && !detail.data ? (
            <Card>
              <SkeletonBlock rows={8} />
            </Card>
          ) : detail.data && detail.data.run.id === selectedRunId ? (
            <RunDetail detail={detail.data} />
          ) : !selectedRunId ? (
            <Card>
              <EmptyState icon={<FlaskConical size={18} />} title="No evaluation runs yet — start one" description="Results, comparison charts and per-question tables appear here." />
            </Card>
          ) : (
            <Card>
              <SkeletonBlock rows={8} />
            </Card>
          )}
        </div>
      </div>
    </>
  )
}
