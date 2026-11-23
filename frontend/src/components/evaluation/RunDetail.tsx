import { useMemo, useState } from 'react'
import { Check, Download, X } from 'lucide-react'
import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import type { EvaluationRow, EvaluationRunDetail, StrategyMetrics } from '../../types/api'
import { evaluationCsvUrl } from '../../services/api'
import { CHART, METRIC_LABELS, STRATEGY_COLORS, STRATEGY_LABELS, STRATEGY_SHORT } from '../../lib/constants'
import { cx, fmtDate, fmtMs, fmtScore, humanize, truthy } from '../../lib/format'
import { Card } from '../Card'
import { Badge, StatusBadge } from '../Badge'
import { EmptyState, ErrorBanner, ProgressBar } from '../Feedback'
import { Tabs } from '../Tabs'
import { StrategyMetricsChart } from '../charts/StrategyMetricsChart'

const SUMMARY_COLS: Array<keyof StrategyMetrics> = [
  'questions',
  'exact_match',
  'f1',
  'precision_at_k',
  'recall_at_k',
  'mrr',
  'evidence_recall',
  'citation_coverage',
  'citation_precision',
  'abstention_accuracy',
  'answer_latency_ms',
  'graph_latency_ms',
]

function cell(k: keyof StrategyMetrics, v: number | null | undefined) {
  if (k === 'answer_latency_ms' || k === 'graph_latency_ms') return fmtMs(v)
  if (k === 'questions') return v ?? '—'
  return fmtScore(v)
}

export function RunDetail({ detail }: { detail: EvaluationRunDetail }) {
  const { run, rows, job } = detail
  const byStrategy = run.summary?.by_strategy ?? {}
  const byType = run.summary?.by_type ?? {}
  const strategies = Object.keys(byStrategy)
  const best = useMemo(() => {
    const m: Partial<Record<keyof StrategyMetrics, number>> = {}
    for (const k of SUMMARY_COLS) {
      if (k === 'questions') continue
      const vals = strategies.map((s) => byStrategy[s][k]).filter((v): v is number => typeof v === 'number')
      if (vals.length) m[k] = k.endsWith('latency_ms') ? Math.min(...vals) : Math.max(...vals)
    }
    return m
  }, [byStrategy, strategies])
  const inProgress = run.status === 'running' || run.status === 'queued'

  return (
    <div className="space-y-6">
      <Card
        title={
          <span className="flex flex-wrap items-center gap-2">
            Run <span className="mono">{run.id}</span> <StatusBadge status={run.status} />
          </span>
        }
        subtitle={`${run.dataset} · started ${fmtDate(run.created_at)}${run.finished_at ? ` · finished ${fmtDate(run.finished_at)}` : ''}`}
        actions={
          run.status === 'completed' && (
            <a href={evaluationCsvUrl(run.id)} className="btn-secondary btn-sm" download>
              <Download size={13} /> Export CSV
            </a>
          )
        }
      >
        <div className="flex flex-wrap gap-1.5 text-xs">
          <Badge tone="neutral">K = {String(run.config.k ?? '—')}</Badge>
          <Badge tone="neutral">top_k = {String(run.config.top_k ?? '—')}</Badge>
          <Badge tone="neutral">hops = {String(run.config.max_hops ?? '—')}</Badge>
          <Badge tone="neutral">limit = {String(run.config.limit ?? 'all')}</Badge>
          <Badge tone="neutral">backend = {String(run.config.backend ?? 'default')}</Badge>
          {typeof run.config.hybrid_alpha === 'number' && <Badge tone="neutral">α = {run.config.hybrid_alpha}</Badge>}
          {typeof run.config.chunk_size === 'number' && <Badge tone="neutral">chunk {run.config.chunk_size}/{String(run.config.chunk_overlap ?? '')}</Badge>}
          {run.summary?.num_questions !== undefined && <Badge tone="accent">{run.summary.num_questions} questions</Badge>}
          {(run.summary?.skipped ?? []).map((s) => (
            <Badge key={s} tone="warning">
              skipped: {s}
            </Badge>
          ))}
        </div>
        {inProgress && (
          <div className="mt-4">
            <ProgressBar value={job?.progress ?? run.progress} />
            <p className="mt-1 text-xs text-subtle">{job?.message ?? 'running…'}</p>
          </div>
        )}
        {run.error && <ErrorBanner title="Run failed" message={run.error} className="mt-4" />}
      </Card>

      {strategies.length > 0 ? (
        <>
          <Card title="Metric summary per strategy" subtitle="Best value per column in bold" flush>
            <div className="overflow-x-auto">
              <table className="table-base">
                <thead>
                  <tr>
                    <th>Strategy</th>
                    {SUMMARY_COLS.map((k) => (
                      <th key={k} className="text-right">
                        {k === 'questions' ? 'N' : (METRIC_LABELS[k] ?? k)}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {strategies.map((s) => (
                    <tr key={s}>
                      <td className="font-medium whitespace-nowrap">
                        <span className="mr-2 inline-block h-2 w-2 rounded-full" style={{ background: STRATEGY_COLORS[s] }} />
                        {STRATEGY_LABELS[s] ?? s}
                      </td>
                      {SUMMARY_COLS.map((k) => {
                        const v = byStrategy[s][k]
                        const isBest = typeof v === 'number' && best[k] === v && strategies.length > 1
                        return (
                          <td key={k} className={cx('text-right tabular-nums', isBest ? 'font-semibold text-emerald-300' : 'text-fg/85')}>
                            {cell(k, v)}
                          </td>
                        )
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          <div className="grid grid-cols-1 gap-6 2xl:grid-cols-2">
            <Card title="Baselines vs graph / GNN reranking" subtitle="Answer quality and retrieval metrics per strategy">
              <StrategyMetricsChart byStrategy={byStrategy} />
            </Card>
            <Card title="Evidence & citation quality">
              <StrategyMetricsChart byStrategy={byStrategy} metrics={['evidence_recall', 'citation_coverage', 'citation_precision', 'abstention_accuracy']} />
            </Card>
          </div>

          {Object.keys(byType).length > 0 && <TypeBreakdown byType={byType} />}
        </>
      ) : (
        !inProgress && (
          <Card>
            <EmptyState title="No summary available" description="This run has not produced per-strategy metrics." />
          </Card>
        )
      )}

      <QuestionTable rows={rows} />
    </div>
  )
}

function TypeBreakdown({ byType }: { byType: Record<string, Record<string, StrategyMetrics>> }) {
  const metrics = ['exact_match', 'f1', 'recall_at_k', 'mrr'] as const
  const [metric, setMetric] = useState<(typeof metrics)[number]>('f1')
  const types = Object.keys(byType)
  const strategies = [...new Set(types.flatMap((t) => Object.keys(byType[t])))]
  const data = types.map((t) => {
    const row: Record<string, string | number | null> = { type: humanize(t) }
    for (const s of strategies) {
      const v = byType[t][s]?.[metric]
      row[s] = typeof v === 'number' ? v : null
    }
    const n = Object.values(byType[t])[0]?.questions
    row.type = `${humanize(t)}${n !== undefined ? ` (${n})` : ''}`
    return row
  })
  return (
    <Card
      title="Per-question-type breakdown"
      subtitle="summary.by_type — bridge, comparison, single-hop and unanswerable questions"
      actions={<Tabs variant="pill" value={metric} onChange={setMetric} items={metrics.map((m) => ({ key: m, label: METRIC_LABELS[m] }))} />}
    >
      <ResponsiveContainer width="100%" height={280}>
        <BarChart data={data} margin={{ top: 8, right: 8, left: -12, bottom: 0 }} barGap={2} barCategoryGap="24%">
          <CartesianGrid stroke={CHART.grid} vertical={false} />
          <XAxis dataKey="type" tick={CHART.tick} axisLine={{ stroke: CHART.grid }} tickLine={false} />
          <YAxis domain={[0, 1]} tick={CHART.tick} axisLine={false} tickLine={false} />
          <Tooltip {...CHART.tooltip} formatter={(v) => (typeof v === 'number' ? v.toFixed(3) : 'n/a (no gold evidence)')} />
          <Legend itemSorter={null} wrapperStyle={{ fontSize: 12 }} iconType="circle" iconSize={8} />
          {strategies.map((s) => (
            <Bar key={s} dataKey={s} name={STRATEGY_SHORT[s] ?? s} fill={STRATEGY_COLORS[s] ?? '#94a3b8'} radius={[3, 3, 0, 0]} maxBarSize={22} />
          ))}
        </BarChart>
      </ResponsiveContainer>
      <p className="mt-1 text-[11px] text-subtle">A zero value draws no bar; the tooltip shows “n/a” where the metric is undefined for that type (e.g. retrieval metrics on unanswerable questions).</p>
    </Card>
  )
}

function isCorrect(r: EvaluationRow): boolean {
  return (r.exact_match ?? 0) >= 1 || (r.f1 ?? 0) >= 0.999
}

function QuestionTable({ rows }: { rows: EvaluationRow[] }) {
  const strategies = [...new Set(rows.map((r) => r.strategy))]
  const [filter, setFilter] = useState<string>('all')
  const [onlyWrong, setOnlyWrong] = useState(false)
  const visible = rows.filter((r) => (filter === 'all' || r.strategy === filter) && (!onlyWrong || !isCorrect(r)))
  const correct = visible.filter(isCorrect).length
  return (
    <Card
      title="Per-question results"
      subtitle={rows.length ? `${visible.length} rows · ${correct} correct (EM) · ${visible.length - correct} incorrect` : undefined}
      actions={
        rows.length > 0 && (
          <div className="flex flex-wrap items-center gap-3">
            <label className="flex items-center gap-1.5 text-xs text-muted">
              <input type="checkbox" className="accent-indigo-500" checked={onlyWrong} onChange={(e) => setOnlyWrong(e.target.checked)} />
              Incorrect only
            </label>
            <Tabs
              variant="pill"
              value={filter}
              onChange={setFilter}
              items={[{ key: 'all', label: 'All' }, ...strategies.map((s) => ({ key: s, label: STRATEGY_SHORT[s] ?? s }))]}
            />
          </div>
        )
      }
      flush
    >
      {rows.length === 0 ? (
        <EmptyState title="No per-question rows yet" />
      ) : (
        <div className="max-h-[640px] overflow-auto">
          <table className="table-base">
            <thead className="sticky top-0 z-10 bg-surface">
              <tr>
                <th></th>
                <th>Question</th>
                <th>Type</th>
                <th>Strategy</th>
                <th>Expected</th>
                <th>Predicted</th>
                <th className="text-right">F1</th>
                <th className="text-right">P@K</th>
                <th className="text-right">R@K</th>
                <th className="text-right">MRR</th>
                <th className="text-right">Ev. recall</th>
                <th className="text-right">Cit. cov.</th>
                <th className="text-right">Conf.</th>
                <th className="text-right">Latency</th>
              </tr>
            </thead>
            <tbody>
              {visible.map((r, i) => {
                const ok = isCorrect(r)
                const eff = r.details?.effective_strategy
                return (
                  <tr key={`${r.question_id}-${r.strategy}-${i}`} className={ok ? '' : 'bg-rose-500/[0.04]'}>
                    <td>
                      <span className={cx('flex h-5 w-5 items-center justify-center rounded-full', ok ? 'bg-emerald-500/15 text-emerald-400' : 'bg-rose-500/15 text-rose-400')}>
                        {ok ? <Check size={12} /> : <X size={12} />}
                      </span>
                    </td>
                    <td className="max-w-[320px] min-w-[220px]">
                      <p className="line-clamp-2 text-xs text-fg" title={r.question}>
                        {r.question}
                      </p>
                      <span className="mono text-subtle">{r.question_id}</span>
                    </td>
                    <td>
                      <Badge tone={r.question_type === 'unanswerable' ? 'warning' : 'neutral'}>{r.question_type}</Badge>
                    </td>
                    <td className="text-xs whitespace-nowrap">
                      {STRATEGY_SHORT[r.strategy] ?? r.strategy}
                      {eff && eff !== r.strategy && <div className="text-[10px] text-amber-300">→ {eff}</div>}
                    </td>
                    <td className="max-w-[160px] text-xs text-muted">{r.expected_answer || <span className="text-subtle italic">(abstain)</span>}</td>
                    <td className={cx('max-w-[160px] text-xs', ok ? 'text-emerald-300' : 'text-rose-300')}>
                      {truthy(r.insufficient) ? <span className="italic">insufficient evidence</span> : r.predicted_answer || '—'}
                      {r.details?.generator_backend && <div className="text-[10px] text-subtle">{r.details.generator_backend}</div>}
                    </td>
                    <td className="text-right text-xs tabular-nums">{fmtScore(r.f1, 2)}</td>
                    <td className="text-right text-xs tabular-nums">{fmtScore(r.precision_at_k, 2)}</td>
                    <td className="text-right text-xs tabular-nums">{fmtScore(r.recall_at_k, 2)}</td>
                    <td className="text-right text-xs tabular-nums">{fmtScore(r.mrr, 2)}</td>
                    <td className="text-right text-xs tabular-nums">{fmtScore(r.evidence_recall, 2)}</td>
                    <td className="text-right text-xs tabular-nums">{fmtScore(r.citation_coverage, 2)}</td>
                    <td className="text-right text-xs tabular-nums">{fmtScore(r.details?.confidence, 2)}</td>
                    <td className="text-right text-xs text-muted tabular-nums">{fmtMs(r.answer_latency_ms)}</td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      )}
    </Card>
  )
}
