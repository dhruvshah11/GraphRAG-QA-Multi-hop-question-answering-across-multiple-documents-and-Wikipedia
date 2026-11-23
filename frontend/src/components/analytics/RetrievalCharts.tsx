import { Fragment, useMemo } from 'react'
import { ArrowDown, ArrowUp, Link2, Minus } from 'lucide-react'
import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import type { ReasoningChain, ScoreRow } from '../../types/api'
import { CHART, SCORE_SERIES, STRATEGY_COLORS, STRATEGY_LABELS, STRATEGY_SHORT } from '../../lib/constants'
import { fmtScore, humanize } from '../../lib/format'
import { Badge } from '../Badge'
import { EmptyState } from '../Feedback'

/** Short display label for a chunk id, e.g. "geoffrey_hinton_profile #1". */
export function chunkLabel(chunkId: string, rows: ScoreRow[]): string {
  const row = rows.find((r) => r.chunk_id === chunkId)
  const m = /:c(\d+)$/.exec(chunkId)
  const idx = m ? `#${Number(m[1]) + 1}` : ''
  const name = row?.document_name ?? chunkId.split(':')[0]
  return `${name.replace(/\.(pdf|txt|docx)$/i, '')} ${idx}`.trim()
}

const PALETTE = ['#60a5fa', '#f59e0b', '#2dd4bf', '#a78bfa', '#f472b6', '#34d399', '#fb7185', '#facc15', '#22d3ee', '#c084fc']

export function useDocColors(rows: ScoreRow[]) {
  return useMemo(() => {
    const m = new Map<string, string>()
    for (const r of rows) if (!m.has(r.document_name)) m.set(r.document_name, PALETTE[m.size % PALETTE.length])
    return m
  }, [rows])
}

export function PassageScoreChart({ rows }: { rows: ScoreRow[] }) {
  const present = SCORE_SERIES.filter((s) => rows.some((r) => typeof r[s.key] === 'number'))
  const data = rows.map((r) => {
    const o: Record<string, string | number | null> = { name: chunkLabel(r.chunk_id, rows) }
    for (const s of present) o[s.key] = r[s.key]
    return o
  })
  if (!rows.length) return <EmptyState title="No passages scored for this query" />
  return (
    <ResponsiveContainer width="100%" height={320}>
      <BarChart data={data} margin={{ top: 8, right: 8, left: -12, bottom: 48 }} barGap={1} barCategoryGap="18%">
        <CartesianGrid stroke={CHART.grid} vertical={false} />
        <XAxis dataKey="name" tick={{ ...CHART.tick, fontSize: 10 }} angle={-25} textAnchor="end" interval={0} axisLine={{ stroke: CHART.grid }} tickLine={false} />
        <YAxis domain={[0, 1]} tick={CHART.tick} axisLine={false} tickLine={false} />
        <Tooltip {...CHART.tooltip} formatter={(v) => (typeof v === 'number' ? v.toFixed(3) : 'n/a')} />
        <Legend itemSorter={null} verticalAlign="top" height={28} wrapperStyle={{ fontSize: 12 }} iconType="circle" iconSize={8} />
        {present.map((s) => (
          <Bar key={s.key} dataKey={s.key} name={s.label} fill={s.color} radius={[2, 2, 0, 0]} maxBarSize={14} />
        ))}
      </BarChart>
    </ResponsiveContainer>
  )
}

/** Side-by-side top-K per strategy; deltas are relative to the hybrid (first-stage) ranking. */
export function RankComparison({ rankings, rows, topK = 8 }: { rankings: Record<string, string[]>; rows: ScoreRow[]; topK?: number }) {
  const strategies = Object.keys(rankings)
  const docColors = useDocColors(rows)
  const depth = Math.min(topK, Math.max(0, ...strategies.map((s) => rankings[s].length)))
  if (!strategies.length || depth === 0) return <EmptyState title="No strategy rankings" />
  const base = rankings.hybrid ?? rankings[strategies[0]]
  const baseName = rankings.hybrid ? 'hybrid' : strategies[0]
  return (
    <div className="overflow-x-auto">
      <table className="table-base">
        <thead>
          <tr>
            <th className="w-10">Rank</th>
            {strategies.map((s) => (
              <th key={s}>
                <span className="inline-flex items-center gap-1.5">
                  <span className="h-2 w-2 rounded-full" style={{ background: STRATEGY_COLORS[s] }} />
                  {STRATEGY_LABELS[s] ?? s}
                </span>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {Array.from({ length: depth }, (_, i) => (
            <tr key={i}>
              <td className="font-mono text-xs text-muted">{i + 1}</td>
              {strategies.map((s) => {
                const id = rankings[s][i]
                if (!id) return <td key={s} className="text-subtle">—</td>
                const row = rows.find((r) => r.chunk_id === id)
                const baseRank = base.indexOf(id)
                const delta = s === baseName || baseRank === -1 ? null : baseRank - i
                return (
                  <td key={s} className="min-w-[170px]">
                    <div className="flex items-center gap-1.5">
                      <span className="h-2 w-2 shrink-0 rounded-sm" style={{ background: docColors.get(row?.document_name ?? '') ?? '#64748b' }} />
                      <span className="truncate text-xs text-fg" title={id}>
                        {chunkLabel(id, rows)}
                      </span>
                      {delta !== null && (
                        <span
                          className={`ml-auto inline-flex shrink-0 items-center text-[10px] font-semibold tabular-nums ${delta > 0 ? 'text-emerald-400' : delta < 0 ? 'text-rose-400' : 'text-subtle'}`}
                          title={`${delta > 0 ? 'up' : delta < 0 ? 'down' : 'same'} vs ${STRATEGY_SHORT[baseName]}`}
                        >
                          {delta > 0 ? <ArrowUp size={10} /> : delta < 0 ? <ArrowDown size={10} /> : <Minus size={10} />}
                          {delta !== 0 && Math.abs(delta)}
                        </span>
                      )}
                      {delta === null && s !== baseName && baseRank === -1 && <span className="ml-auto text-[10px] text-sky-300">new</span>}
                    </div>
                  </td>
                )
              })}
            </tr>
          ))}
        </tbody>
      </table>
      <p className="mt-2 px-3 text-[11px] text-subtle">Arrows show movement relative to the {STRATEGY_SHORT[baseName]} ranking; “new” marks passages it did not rank.</p>
    </div>
  )
}

export function ChainRanking({ chains, rows }: { chains: ReasoningChain[]; rows: ScoreRow[] }) {
  if (!chains.length) return <EmptyState title="No reasoning chains" description="No multi-hop chains were constructed for this query." />
  return (
    <ol className="space-y-3">
      {chains.map((c, i) => (
        <li key={i} className={`rounded-lg border p-3.5 ${i === 0 ? 'border-indigo-500/40 bg-indigo-500/[0.05]' : 'border-line'}`}>
          <div className="flex flex-wrap items-center gap-2">
            <span className="font-mono text-xs font-semibold text-muted">#{i + 1}</span>
            {i === 0 && <Badge tone="accent">selected</Badge>}
            <span className="text-xs text-muted">
              score <span className="font-semibold text-fg tabular-nums">{fmtScore(c.score)}</span>
            </span>
            <span className="ml-auto flex flex-wrap gap-1">
              {Object.entries(c.components).map(([k, v]) => (
                <Badge key={k} tone="neutral">
                  {humanize(k)} {k === 'length' ? v : fmtScore(v, 2)}
                </Badge>
              ))}
            </span>
          </div>
          <div className="mt-2.5 flex flex-wrap items-center gap-1.5 text-xs">
            {c.steps.map((s, j) => (
              <Fragment key={`${s.chunk_id}-${j}`}>
                {j > 0 && (
                  <span className="inline-flex items-center gap-1 rounded-full border border-emerald-500/35 bg-emerald-500/10 px-2 py-0.5 text-[11px] text-emerald-200">
                    <Link2 size={10} /> {s.via_entity_name ?? 'link'}
                  </span>
                )}
                <span className="rounded-md border border-line bg-surface-2 px-2 py-0.5 text-fg" title={s.reason ?? undefined}>
                  {chunkLabel(s.chunk_id, rows)}
                  <span className="ml-1 text-subtle tabular-nums">{fmtScore(s.relevance, 2)}</span>
                </span>
              </Fragment>
            ))}
          </div>
          {(c.missing_terms?.length ?? 0) > 0 && <p className="mt-2 text-[11px] text-subtle">missing terms: {c.missing_terms?.join(', ')}</p>}
        </li>
      ))}
    </ol>
  )
}
