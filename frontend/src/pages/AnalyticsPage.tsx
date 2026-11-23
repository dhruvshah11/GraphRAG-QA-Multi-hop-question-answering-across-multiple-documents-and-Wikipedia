import { useMemo, useState } from 'react'
import { Link, useSearchParams } from 'react-router-dom'
import { BarChart3, GitCommitVertical, ListOrdered, RefreshCw, Scale } from 'lucide-react'
import { useApi } from '../hooks/useApi'
import { getGnnStatus, getPerformance, getRetrievalAnalytics } from '../services/api'
import type { EvaluationRunRef } from '../types/api'
import { PageHeader } from '../components/PageHeader'
import { Card } from '../components/Card'
import { EmptyState, ErrorBanner, SkeletonBlock } from '../components/Feedback'
import { Badge } from '../components/Badge'
import { Tabs } from '../components/Tabs'
import { StrategyMetricsChart } from '../components/charts/StrategyMetricsChart'
import { ChainRanking, PassageScoreChart, RankComparison } from '../components/analytics/RetrievalCharts'
import { GnnPanel } from '../components/analytics/GnnPanel'
import { PerformancePanel } from '../components/analytics/PerformancePanel'
import { METRIC_LABELS, STRATEGY_LABELS } from '../lib/constants'
import { fmtDate, fmtMs, fmtScore } from '../lib/format'

export default function AnalyticsPage() {
  const [params, setParams] = useSearchParams()
  const queryId = params.get('query')
  const retrieval = useApi((s) => getRetrievalAnalytics(queryId, s), [queryId])
  const perf = useApi((s) => getPerformance(s), [])
  const gnn = useApi((s) => getGnnStatus(s), [])
  const r = retrieval.data

  const refresh = () => {
    void retrieval.refetch()
    void perf.refetch()
    void gnn.refetch()
  }

  return (
    <>
      <PageHeader
        title="Retrieval & GNN Analytics"
        description="How each retrieval strategy scores and ranks passages, which reasoning chains were built, and how the GAT reranker was trained."
        actions={
          <button type="button" className="btn-secondary" onClick={refresh} disabled={retrieval.loading}>
            <RefreshCw size={14} className={retrieval.loading ? 'animate-spin' : ''} /> Refresh
          </button>
        }
      />
      {retrieval.error && <ErrorBanner message={retrieval.error} onRetry={() => void retrieval.refetch()} className="mb-6" />}

      <div className="space-y-6">
        <Card title="Query" subtitle="Per-passage analytics are computed for one answered question">
          {retrieval.loading && !r ? (
            <SkeletonBlock rows={2} />
          ) : r && r.recent_queries.length ? (
            <div className="flex flex-wrap items-end gap-4">
              <div className="min-w-[260px] flex-1">
                <label htmlFor="aq" className="label">
                  Question
                </label>
                <select id="aq" className="input" value={r.query_id ?? ''} onChange={(e) => setParams({ query: e.target.value })}>
                  {r.query_id && !r.recent_queries.some((q) => q.id === r.query_id) && <option value={r.query_id}>{r.question}</option>}
                  {r.recent_queries.map((q) => (
                    <option key={q.id} value={q.id}>
                      {q.question}
                    </option>
                  ))}
                </select>
              </div>
              <div className="flex flex-wrap gap-1.5 pb-1">
                {r.effective_strategy && <Badge tone="accent">effective: {STRATEGY_LABELS[r.effective_strategy] ?? r.effective_strategy}</Badge>}
                {r.query_id && (
                  <>
                    <Link to={`/qa?query=${r.query_id}`} className="btn-ghost btn-sm">
                      Open answer
                    </Link>
                    <Link to={`/graph?query=${r.query_id}&view=chain`} className="btn-ghost btn-sm">
                      View chain graph
                    </Link>
                  </>
                )}
              </div>
            </div>
          ) : (
            <EmptyState
              title="No questions answered yet"
              description="Retrieval analytics need at least one answered question."
              action={
                <Link to="/qa" className="btn-primary">
                  Ask a question
                </Link>
              }
            />
          )}
        </Card>

        {r && r.query_id && (
          <>
            <Card title="Passage scores by strategy" subtitle="Semantic (FAISS), BM25, hybrid fusion, graph heuristic and GNN scores per retrieved passage" icon={<BarChart3 size={16} />}>
              <PassageScoreChart rows={r.score_table} />
            </Card>
            <div className="grid grid-cols-1 gap-6 2xl:grid-cols-5">
              <Card className="2xl:col-span-3" title="Rank comparison" subtitle="Top-K per strategy side by side" icon={<ListOrdered size={16} />} flush>
                <RankComparison rankings={r.strategy_rankings} rows={r.score_table} />
              </Card>
              <Card className="2xl:col-span-2" title="Evidence-chain ranking" subtitle="Candidate multi-hop chains ordered by chain score" icon={<GitCommitVertical size={16} />}>
                <ChainRanking chains={r.chains} rows={r.score_table} />
              </Card>
            </div>
          </>
        )}

        {r && <StrategyComparison runs={r.evaluation_runs} />}

        {gnn.error ? (
          <ErrorBanner message={gnn.error} onRetry={() => void gnn.refetch()} />
        ) : gnn.data ? (
          <GnnPanel gnn={gnn.data} onChanged={() => void gnn.refetch(true)} />
        ) : (
          <Card title="GNN reranker">
            <SkeletonBlock rows={4} />
          </Card>
        )}

        {perf.error ? (
          <ErrorBanner message={perf.error} onRetry={() => void perf.refetch()} />
        ) : perf.data ? (
          <PerformancePanel perf={perf.data} />
        ) : (
          <Card title="Performance">
            <SkeletonBlock rows={4} />
          </Card>
        )}
      </div>
    </>
  )
}

function StrategyComparison({ runs }: { runs: EvaluationRunRef[] }) {
  const byDataset = useMemo(() => {
    const m = new Map<string, EvaluationRunRef[]>()
    for (const run of runs) {
      const arr = m.get(run.dataset) ?? []
      arr.push(run)
      m.set(run.dataset, arr)
    }
    for (const arr of m.values()) arr.sort((a, b) => (b.created_at ?? '').localeCompare(a.created_at ?? ''))
    return m
  }, [runs])
  const datasets = [...byDataset.keys()]
  const [ds, setDs] = useState<string>(datasets[0] ?? '')
  const [runIdx, setRunIdx] = useState(0)
  const dsRuns = byDataset.get(ds) ?? byDataset.get(datasets[0] ?? '') ?? []
  const run = dsRuns[Math.min(runIdx, dsRuns.length - 1)]
  const byStrategy = run?.summary?.by_strategy ?? {}

  return (
    <Card
      title="Strategy comparison"
      subtitle="FAISS vs BM25 vs Hybrid vs Graph heuristic vs GNN on completed evaluation runs"
      icon={<Scale size={16} />}
      actions={
        datasets.length > 0 && (
          <Tabs
            variant="pill"
            value={byDataset.has(ds) ? ds : datasets[0]}
            onChange={(d) => {
              setDs(d)
              setRunIdx(0)
            }}
            items={datasets.map((d) => ({ key: d, label: d, count: byDataset.get(d)?.length }))}
          />
        )
      }
    >
      {!run ? (
        <EmptyState
          title="No evaluation runs yet — start one"
          action={
            <Link to="/evaluation" className="btn-primary">
              Open Evaluation Lab
            </Link>
          }
        />
      ) : (
        <div className="space-y-4">
          <div className="flex flex-wrap items-center gap-3 text-xs text-muted">
            <select className="input w-auto py-1 text-xs" value={runIdx} onChange={(e) => setRunIdx(Number(e.target.value))}>
              {dsRuns.map((x, i) => (
                <option key={x.run_id} value={i}>
                  {x.run_id} · {fmtDate(x.created_at)}
                </option>
              ))}
            </select>
            <span>
              K={String(run.config.k ?? '—')} · top_k={String(run.config.top_k ?? '—')} · hops={String(run.config.max_hops ?? '—')} · {run.summary.num_questions ?? '—'} questions
            </span>
            {(run.summary.skipped ?? []).map((s) => (
              <Badge key={s} tone="warning">
                skipped: {s}
              </Badge>
            ))}
          </div>
          {Object.keys(byStrategy).length ? (
            <>
              <StrategyMetricsChart byStrategy={byStrategy} />
              <div className="overflow-x-auto">
                <table className="table-base">
                  <thead>
                    <tr>
                      <th>Strategy</th>
                      {['exact_match', 'f1', 'precision_at_k', 'recall_at_k', 'mrr', 'evidence_recall'].map((m) => (
                        <th key={m} className="text-right">
                          {METRIC_LABELS[m]}
                        </th>
                      ))}
                      <th className="text-right">Answer latency</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(byStrategy).map(([s, m]) => (
                      <tr key={s}>
                        <td className="font-medium">{STRATEGY_LABELS[s] ?? s}</td>
                        {(['exact_match', 'f1', 'precision_at_k', 'recall_at_k', 'mrr', 'evidence_recall'] as const).map((k) => (
                          <td key={k} className="text-right tabular-nums">
                            {fmtScore(m[k])}
                          </td>
                        ))}
                        <td className="text-right text-muted tabular-nums">{fmtMs(m.answer_latency_ms)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </>
          ) : (
            <EmptyState title="This run has no per-strategy summary" />
          )}
        </div>
      )}
    </Card>
  )
}
