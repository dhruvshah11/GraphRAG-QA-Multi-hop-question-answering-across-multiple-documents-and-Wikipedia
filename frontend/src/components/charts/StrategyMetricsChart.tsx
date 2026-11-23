import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import type { StrategyMetrics } from '../../types/api'
import { CHART, METRIC_LABELS, STRATEGY_COLORS, STRATEGY_SHORT } from '../../lib/constants'

export const DEFAULT_METRICS: Array<keyof StrategyMetrics> = [
  'exact_match',
  'f1',
  'precision_at_k',
  'recall_at_k',
  'mrr',
  'evidence_recall',
]

interface Props {
  byStrategy: Record<string, StrategyMetrics>
  metrics?: Array<keyof StrategyMetrics>
  height?: number
}

/** Grouped bars: one group per metric, one bar per strategy (fixed strategy colors). */
export function StrategyMetricsChart({ byStrategy, metrics = DEFAULT_METRICS, height = 300 }: Props) {
  const strategies = Object.keys(byStrategy)
  const data = metrics.map((m) => {
    const row: Record<string, string | number | null> = { metric: METRIC_LABELS[m] ?? m }
    for (const s of strategies) {
      const v = byStrategy[s]?.[m]
      row[s] = typeof v === 'number' ? Number(v.toFixed(4)) : null
    }
    return row
  })
  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={data} margin={{ top: 8, right: 8, left: -12, bottom: 0 }} barGap={2} barCategoryGap="22%">
        <CartesianGrid stroke={CHART.grid} vertical={false} />
        <XAxis dataKey="metric" tick={CHART.tick} axisLine={{ stroke: CHART.grid }} tickLine={false} interval={0} />
        <YAxis domain={[0, 1]} tick={CHART.tick} axisLine={false} tickLine={false} tickCount={6} />
        <Tooltip {...CHART.tooltip} formatter={(v) => (typeof v === 'number' ? v.toFixed(3) : 'n/a')} />
        <Legend itemSorter={null} wrapperStyle={{ fontSize: 12, color: '#94a3b8' }} iconType="circle" iconSize={8} />
        {strategies.map((s) => (
          <Bar
            key={s}
            dataKey={s}
            name={STRATEGY_SHORT[s] ?? s}
            fill={STRATEGY_COLORS[s] ?? '#94a3b8'}
            radius={[3, 3, 0, 0]}
            maxBarSize={22}
          />
        ))}
      </BarChart>
    </ResponsiveContainer>
  )
}
