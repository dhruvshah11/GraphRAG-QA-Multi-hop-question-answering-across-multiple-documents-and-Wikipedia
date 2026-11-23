import { useState } from 'react'
import { Bar, BarChart, CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { Gauge, Timer } from 'lucide-react'
import type { PerformanceResponse, StageLatency } from '../../types/api'
import { CHART, STAGE_LABELS } from '../../lib/constants'
import { fmtDate, fmtMs, fmtValue, humanize } from '../../lib/format'
import { Card } from '../Card'
import { EmptyState } from '../Feedback'
import { Tabs } from '../Tabs'

function StageChart({ stages }: { stages: StageLatency[] }) {
  const [scale, setScale] = useState<'linear' | 'log'>('log')
  if (!stages.length) return <EmptyState title="No stage measurements yet" />
  const data = stages.map((s) => ({ ...s, name: STAGE_LABELS[s.stage] ?? humanize(s.stage), mean_ms: Math.max(s.mean_ms, 0.01), p95_ms: Math.max(s.p95_ms, 0.01) }))
  return (
    <div>
      <div className="mb-2 flex justify-end">
        <Tabs
          variant="pill"
          value={scale}
          onChange={setScale}
          items={[
            { key: 'log', label: 'Log scale' },
            { key: 'linear', label: 'Linear' },
          ]}
        />
      </div>
      <ResponsiveContainer width="100%" height={Math.max(220, data.length * 30)}>
        <BarChart data={data} layout="vertical" margin={{ top: 0, right: 16, left: 8, bottom: 0 }} barGap={1}>
          <CartesianGrid stroke={CHART.grid} horizontal={false} />
          <XAxis
            type="number"
            scale={scale}
            domain={scale === 'log' ? [0.01, 'auto'] : [0, 'auto']}
            allowDataOverflow
            tick={CHART.tick}
            axisLine={false}
            tickLine={false}
            tickFormatter={(v: number) => fmtMs(v)}
          />
          <YAxis type="category" dataKey="name" width={130} tick={CHART.tick} axisLine={false} tickLine={false} />
          <Tooltip {...CHART.tooltip} formatter={(v) => (typeof v === 'number' ? fmtMs(v) : 'n/a')} />
          <Legend itemSorter={null} wrapperStyle={{ fontSize: 12 }} iconType="circle" iconSize={8} />
          <Bar dataKey="mean_ms" name="Mean" fill="#818cf8" radius={[0, 3, 3, 0]} maxBarSize={10} />
          <Bar dataKey="p95_ms" name="p95" fill="#f472b6" radius={[0, 3, 3, 0]} maxBarSize={10} />
        </BarChart>
      </ResponsiveContainer>
      <p className="mt-1 text-[11px] text-subtle">Counts: {stages.map((s) => `${STAGE_LABELS[s.stage] ?? s.stage} ×${s.count}`).join(' · ')}</p>
    </div>
  )
}

export function PerformancePanel({ perf }: { perf: PerformanceResponse }) {
  const [stageTab, setStageTab] = useState<'query' | 'ingestion'>('query')
  const series = [...perf.recent_queries]
    .reverse()
    .map((q, i) => ({ i: i + 1, latency_ms: q.latency_ms, question: q.question, strategy: q.strategy, at: fmtDate(q.created_at) }))
  const loads = Object.entries(perf.model_load_ms ?? {}).filter(([, v]) => typeof v === 'number') as Array<[string, number]>
  return (
    <div className="grid grid-cols-1 gap-6 xl:grid-cols-2">
      <Card
        title="Stage latency"
        subtitle="Measured per pipeline stage across all recorded runs"
        icon={<Timer size={16} />}
        actions={
          <Tabs
            variant="pill"
            value={stageTab}
            onChange={setStageTab}
            items={[
              { key: 'query', label: 'Query', count: perf.query_stages.length },
              { key: 'ingestion', label: 'Ingestion', count: perf.ingestion_stages.length },
            ]}
          />
        }
      >
        <StageChart stages={stageTab === 'query' ? perf.query_stages : perf.ingestion_stages} />
      </Card>
      <div className="space-y-6">
        <Card title="Latency over queries" subtitle="End-to-end latency of recent questions (oldest → newest)" icon={<Gauge size={16} />}>
          {series.length ? (
            <ResponsiveContainer width="100%" height={220}>
              <LineChart data={series} margin={{ top: 4, right: 12, left: -6, bottom: 0 }}>
                <CartesianGrid stroke={CHART.grid} vertical={false} />
                <XAxis dataKey="i" tick={CHART.tick} axisLine={{ stroke: CHART.grid }} tickLine={false} />
                <YAxis tick={CHART.tick} axisLine={false} tickLine={false} tickFormatter={(v: number) => fmtMs(v)} width={60} />
                <Tooltip
                  {...CHART.tooltip}
                  formatter={(v) => [typeof v === 'number' ? fmtMs(v) : 'n/a', 'latency']}
                  labelFormatter={(_, p) => {
                    const d = p?.[0]?.payload as { question?: string; strategy?: string; at?: string } | undefined
                    return d ? `${d.question ?? ''} (${d.strategy ?? ''}, ${d.at ?? ''})` : ''
                  }}
                />
                <Line type="monotone" dataKey="latency_ms" stroke="#818cf8" strokeWidth={2} dot={{ r: 3, fill: '#818cf8' }} />
              </LineChart>
            </ResponsiveContainer>
          ) : (
            <EmptyState title="No queries measured yet" />
          )}
        </Card>
        <Card title="Model load time">
          <div>
            <div>
              {loads.length ? (
                <ul className="space-y-1 text-sm">
                  {loads.map(([k, v]) => (
                    <li key={k} className="flex justify-between">
                      <span className="text-muted">{humanize(k)}</span>
                      <span className="text-fg tabular-nums">{fmtMs(v)}</span>
                    </li>
                  ))}
                </ul>
              ) : (
                <p className="text-xs text-subtle">No models loaded yet.</p>
              )}
            </div>
          </div>
        </Card>
      </div>
      <Card className="xl:col-span-2" title="Latency benchmark" subtitle="results/latency_results.json (offline benchmark script)">
        {perf.benchmark ? <BenchmarkView bench={perf.benchmark} /> : <EmptyState title="No benchmark results" description="results/latency_results.json was not found on the server." />}
      </Card>
    </div>
  )
}

type Rec = Record<string, unknown>
const isRec = (v: unknown): v is Rec => typeof v === 'object' && v !== null && !Array.isArray(v)
const num = (v: unknown): number | null => (typeof v === 'number' ? v : null)

/** Defensive rendering of the benchmark JSON: environment, config and a per-stage latency table. */
function BenchmarkView({ bench }: { bench: Rec }) {
  const env = isRec(bench.environment) ? bench.environment : {}
  const cfg = isRec(bench.config) ? bench.config : {}
  const stages = isRec(bench.stages) ? Object.entries(bench.stages).filter(([, v]) => isRec(v)) as Array<[string, Rec]> : []
  const META = new Set(['ms', 'mean_ms', 'median_ms', 'p95_ms', 'min_ms', 'max_ms', 'n', 'total_ms', 'files', 'node_types', 'edge_types'])
  return (
    <div className="space-y-4">
      <div className="flex flex-wrap gap-x-6 gap-y-1 text-xs text-muted">
        {typeof bench.generated_at === 'string' && <span>generated {fmtDate(bench.generated_at)}</span>}
        {Object.entries(env).map(([k, v]) => (
          <span key={k}>
            {humanize(k)}: <span className="text-fg">{fmtValue(v)}</span>
          </span>
        ))}
      </div>
      {Object.keys(cfg).length > 0 && (
        <div className="flex flex-wrap gap-x-4 gap-y-1 text-[11px] text-subtle">
          {Object.entries(cfg).map(([k, v]) => (
            <span key={k}>
              {k}=<span className="text-muted">{fmtValue(v)}</span>
            </span>
          ))}
        </div>
      )}
      {stages.length > 0 && (
        <div className="overflow-x-auto">
          <table className="table-base">
            <thead>
              <tr>
                <th>Stage</th>
                <th className="text-right">Time / mean</th>
                <th className="text-right">Median</th>
                <th className="text-right">p95</th>
                <th className="text-right">n</th>
                <th>Details</th>
              </tr>
            </thead>
            <tbody>
              {stages.map(([name, v]) => {
                const main = num(v.mean_ms) ?? num(v.ms) ?? num(v.total_ms)
                const details = Object.entries(v)
                  .filter(([k, x]) => !META.has(k) && (typeof x !== 'object' || x === null))
                  .map(([k, x]) => `${k.replace(/_/g, ' ')}: ${fmtValue(x)}`)
                  .join(' · ')
                return (
                  <tr key={name}>
                    <td className="whitespace-nowrap">{humanize(name.replace(/^\d+_/, ''))}</td>
                    <td className="text-right tabular-nums">{fmtMs(main)}</td>
                    <td className="text-right text-muted tabular-nums">{fmtMs(num(v.median_ms))}</td>
                    <td className="text-right text-muted tabular-nums">{fmtMs(num(v.p95_ms))}</td>
                    <td className="text-right text-muted tabular-nums">{num(v.n) ?? (Array.isArray(v.files) ? v.files.length : '—')}</td>
                    <td className="max-w-[420px] text-xs text-subtle">{details || '—'}</td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}
