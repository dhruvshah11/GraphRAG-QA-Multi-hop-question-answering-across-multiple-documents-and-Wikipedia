import { useMemo, useState, type ReactNode } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import {
  Activity,
  BookOpen,
  Boxes,
  FileText,
  GitBranch,
  Layers,
  MessageSquareText,
  Network,
  RefreshCw,
  ShieldCheck,
  Timer,
} from 'lucide-react'
import { useApi } from '../hooks/useApi'
import { getOverview } from '../services/api'
import { PageHeader } from '../components/PageHeader'
import { Card } from '../components/Card'
import { StatCard } from '../components/StatCard'
import { EmptyState, ErrorBanner, SkeletonBlock, SkeletonCards } from '../components/Feedback'
import { PipelineStepper } from '../components/PipelineStepper'
import { Badge, StatusBadge } from '../components/Badge'
import { Tabs } from '../components/Tabs'
import { StrategyMetricsChart } from '../components/charts/StrategyMetricsChart'
import { CHART, EDGE_TYPE_STYLES, NODE_TYPE_COLORS, STRATEGY_SHORT } from '../lib/constants'
import { fmtDate, fmtMs, fmtNumber, fmtPct, fmtRelative, humanize, truthy } from '../lib/format'
import type { OverviewResponse } from '../types/api'

export default function DashboardPage() {
  const { data, error, loading, refetch } = useApi((s) => getOverview(s), [])

  return (
    <>
      <PageHeader
        title={data?.project.name ?? 'Dashboard'}
        description={data?.project.description ?? 'Overview of the GraphRAG-QA pipeline, corpus and recent activity.'}
        actions={
          <button type="button" className="btn-secondary" onClick={() => void refetch()} disabled={loading}>
            <RefreshCw size={14} className={loading ? 'animate-spin' : ''} /> Refresh
          </button>
        }
      />
      {error && <ErrorBanner message={error} onRetry={() => void refetch()} className="mb-6" />}
      {loading && !data ? (
        <div className="space-y-6">
          <SkeletonCards count={6} className="grid-cols-2 md:grid-cols-3 xl:grid-cols-6" />
          <Card title="Pipeline status">
            <SkeletonBlock rows={3} />
          </Card>
        </div>
      ) : data ? (
        <DashboardContent data={data} />
      ) : null}
    </>
  )
}

function DashboardContent({ data }: { data: OverviewResponse }) {
  const c = data.counts
  return (
    <div className="space-y-6">
      <div className="grid grid-cols-2 gap-3 md:grid-cols-3 xl:grid-cols-6">
        <StatCard label="Documents" value={fmtNumber(c.documents)} hint={`${fmtNumber(c.processed_documents)} processed · ${fmtNumber(c.pages)} pages`} icon={<FileText size={15} />} />
        <StatCard label="Passages" value={fmtNumber(c.passages)} hint={`avg ${fmtNumber(c.avg_chunk_tokens, 1)} tokens`} icon={<Layers size={15} />} />
        <StatCard label="Entities" value={fmtNumber(c.entities)} hint={`${fmtNumber(c.entity_mentions)} mentions`} icon={<Boxes size={15} />} />
        <StatCard label="Relations" value={fmtNumber(c.relations)} hint="dependency-parsed" icon={<GitBranch size={15} />} />
        <StatCard label="Graph" value={fmtNumber(c.graph_nodes)} hint={`${fmtNumber(c.graph_edges)} edges`} icon={<Network size={15} />} />
        <StatCard
          label="Questions answered"
          value={fmtNumber(c.questions_answered)}
          hint={`${fmtNumber(c.wikipedia_cache_entries)} Wikipedia cache entries`}
          icon={<MessageSquareText size={15} />}
        />
      </div>

      <Card title="Pipeline status" subtitle="Nine stages from ingestion to grounded answer generation" icon={<Activity size={16} />}>
        {data.pipeline.length ? <PipelineStepper stages={data.pipeline} /> : <EmptyState title="No pipeline information" />}
      </Card>

      <div className="grid grid-cols-1 gap-6 xl:grid-cols-5">
        <Card className="xl:col-span-3" title="Recent queries" subtitle="Click a query to open its full answer" icon={<MessageSquareText size={16} />} flush>
          <RecentQueries data={data} />
        </Card>
        <Card className="xl:col-span-2" title="Graph composition" subtitle={`${fmtNumber(data.graph.nodes)} nodes · ${fmtNumber(data.graph.edges)} edges`} icon={<Network size={16} />}>
          <GraphComposition data={data} />
        </Card>
      </div>

      <div className="grid grid-cols-1 gap-6 xl:grid-cols-5">
        <Card className="xl:col-span-3" title="Retrieval performance" subtitle="Latest evaluation run per dataset" icon={<BookOpen size={16} />}>
          <RetrievalPerformance data={data} />
        </Card>
        <Card className="xl:col-span-2" title="System health" icon={<ShieldCheck size={16} />} actions={<StatusBadge status={data.health.status} />}>
          <SystemHealth data={data} />
        </Card>
      </div>
    </div>
  )
}

function RecentQueries({ data }: { data: OverviewResponse }) {
  const navigate = useNavigate()
  if (!data.recent_queries.length) {
    return (
      <EmptyState
        title="No questions asked yet"
        description="Ask a multi-hop question to see it here."
        action={
          <Link to="/qa" className="btn-primary">
            Ask a question
          </Link>
        }
      />
    )
  }
  return (
    <ul className="divide-y divide-line">
      {data.recent_queries.map((q) => (
        <li key={q.id}>
          <button
            type="button"
            onClick={() => navigate(`/qa?query=${encodeURIComponent(q.id)}`)}
            className="w-full px-4 py-3 text-left transition-colors hover:bg-surface-2/60 sm:px-5"
          >
            <div className="flex items-start gap-3">
              <div className="min-w-0 flex-1">
                <p className="truncate text-sm font-medium text-fg">{q.question}</p>
                <p className="mt-0.5 truncate text-sm text-muted">
                  {truthy(q.insufficient_evidence) ? (
                    <span className="text-amber-300">Insufficient evidence</span>
                  ) : (
                    <>
                      <span className="text-subtle">Answer: </span>
                      <span className="text-indigo-200">{q.answer || '—'}</span>
                    </>
                  )}
                </p>
              </div>
              <div className="flex shrink-0 flex-col items-end gap-1">
                <div className="flex gap-1">
                  <Badge tone="accent">{STRATEGY_SHORT[q.strategy] ?? q.strategy}</Badge>
                  <Badge tone="neutral">{q.mode === 'open_domain' ? 'open-domain' : 'multi-doc'}</Badge>
                </div>
                <span className="text-[11px] text-subtle" title={fmtDate(q.created_at)}>
                  {fmtMs(q.latency_ms)} · conf {fmtPct(q.confidence, 0)} · {fmtRelative(q.created_at)}
                </span>
              </div>
            </div>
          </button>
        </li>
      ))}
    </ul>
  )
}

function GraphComposition({ data }: { data: OverviewResponse }) {
  const [tab, setTab] = useState<'nodes' | 'edges'>('nodes')
  const rows = useMemo(() => {
    if (tab === 'nodes') {
      return Object.entries(data.graph.node_types).map(([k, v]) => ({ name: humanize(k), value: v, color: NODE_TYPE_COLORS[k] ?? '#94a3b8' }))
    }
    return Object.entries(data.graph.edge_types).map(([k, v]) => ({ name: k, value: v, color: EDGE_TYPE_STYLES[k]?.color ?? '#94a3b8' }))
  }, [data, tab])
  const total = rows.reduce((a, r) => a + r.value, 0)
  return (
    <div>
      <Tabs
        variant="pill"
        value={tab}
        onChange={setTab}
        items={[
          { key: 'nodes', label: 'Node types' },
          { key: 'edges', label: 'Edge types' },
        ]}
      />
      {total === 0 ? (
        <EmptyState title="Graph is empty" description="Upload and process documents to build the evidence graph." />
      ) : (
        <div className="mt-3">
          <ResponsiveContainer width="100%" height={Math.max(160, rows.length * 34)}>
            <BarChart data={rows} layout="vertical" margin={{ top: 4, right: 32, left: 8, bottom: 0 }}>
              <CartesianGrid stroke={CHART.grid} horizontal={false} />
              <XAxis type="number" tick={CHART.tick} axisLine={false} tickLine={false} allowDecimals={false} />
              <YAxis type="category" dataKey="name" tick={CHART.tick} axisLine={false} tickLine={false} width={112} />
              <Tooltip {...CHART.tooltip} formatter={(v) => [fmtNumber(Number(v)), 'count']} />
              <Bar dataKey="value" radius={[0, 4, 4, 0]} maxBarSize={18} label={{ position: 'right', fill: '#94a3b8', fontSize: 11 }}>
                {rows.map((r) => (
                  <Cell key={r.name} fill={r.color} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  )
}

function RetrievalPerformance({ data }: { data: OverviewResponse }) {
  const datasets = Object.keys(data.latest_evaluations)
  const [ds, setDs] = useState(datasets[0] ?? '')
  const active = data.latest_evaluations[ds]
  if (!datasets.length || !active) {
    return (
      <EmptyState
        title="No evaluation runs yet — start one"
        description="Run an evaluation to compare FAISS, BM25, hybrid, graph and GNN retrieval."
        action={
          <Link to="/evaluation" className="btn-primary">
            Open Evaluation Lab
          </Link>
        }
      />
    )
  }
  const strategies = Object.keys(active.summary)
  return (
    <div>
      <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
        <Tabs variant="pill" value={ds} onChange={setDs} items={datasets.map((d) => ({ key: d, label: d }))} />
        <span className="text-xs text-subtle">
          run <span className="mono">{active.run_id}</span> · {fmtDate(active.created_at)}
        </span>
      </div>
      {strategies.length ? (
        <StrategyMetricsChart byStrategy={active.summary} height={280} />
      ) : (
        <EmptyState title="This run has no strategy metrics" />
      )}
      <div className="mt-2 text-right">
        <Link to="/evaluation" className="text-xs text-indigo-300 hover:underline">
          Full results in Evaluation Lab →
        </Link>
      </div>
    </div>
  )
}

function SystemHealth({ data }: { data: OverviewResponse }) {
  const comps = Object.entries(data.health.components)
  return (
    <div className="space-y-4">
      <div className="grid grid-cols-3 gap-2">
        <MiniStat icon={<Timer size={13} />} label="Avg query" value={fmtMs(data.latency.avg_query_ms)} />
        <MiniStat icon={<Timer size={13} />} label="p95 query" value={fmtMs(data.latency.p95_query_ms)} />
        <MiniStat icon={<Activity size={13} />} label="Measured" value={fmtNumber(data.latency.queries_measured)} />
      </div>
      <ul className="divide-y divide-line rounded-lg border border-line">
        {comps.map(([name, comp]) => {
          const detail =
            (typeof comp.model === 'string' && comp.model) ||
            (typeof comp.backend === 'string' && comp.backend) ||
            (typeof comp.vectors === 'number' && `${comp.vectors} vectors`) ||
            (typeof comp.reason === 'string' && comp.reason) ||
            (comp.backends && typeof comp.backends === 'object'
              ? Object.entries(comp.backends as Record<string, boolean>)
                  .filter(([, ok]) => ok)
                  .map(([k]) => k)
                  .join(', ')
              : '') ||
            ''
          const err = typeof comp.error === 'string' && comp.error ? comp.error : null
          return (
            <li key={name} className="flex items-start gap-3 px-3 py-2.5">
              <div className="min-w-0 flex-1">
                <p className="text-sm text-fg">{humanize(name)}</p>
                {detail && <p className="truncate text-xs text-subtle" title={String(detail)}>{String(detail)}</p>}
                {err && <p className="text-xs text-rose-300">{err}</p>}
              </div>
              <StatusBadge status={comp.status} />
            </li>
          )
        })}
      </ul>
    </div>
  )
}

function MiniStat({ icon, label, value }: { icon: ReactNode; label: string; value: string }) {
  return (
    <div className="rounded-lg border border-line bg-surface-2/50 px-3 py-2">
      <div className="flex items-center gap-1.5 text-[11px] text-muted">
        {icon}
        {label}
      </div>
      <div className="mt-0.5 text-sm font-semibold text-fg tabular-nums">{value}</div>
    </div>
  )
}
