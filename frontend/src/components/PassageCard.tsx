import { useState } from 'react'
import { ChevronDown, ChevronUp } from 'lucide-react'
import type { RetrievedPassage } from '../types/api'
import { SCORE_SERIES } from '../lib/constants'
import { cx, fmtScore } from '../lib/format'
import { Badge } from './Badge'
import { ScoreBar } from './ScoreBar'
import { sourceLabel } from './CitationCard'

interface PassageCardProps {
  passage: RetrievedPassage
  cited?: boolean
  inChain?: boolean
}

export function PassageCard({ passage, cited, inChain }: PassageCardProps) {
  const [expanded, setExpanded] = useState(false)
  return (
    <div className={cx('rounded-lg border bg-surface-2/40 p-3.5', inChain ? 'border-indigo-500/35' : 'border-line')}>
      <div className="flex flex-wrap items-center gap-2">
        <span className="font-mono text-xs font-semibold text-muted">#{passage.rank}</span>
        <span className="min-w-0 flex-1 truncate text-sm font-medium text-fg">{sourceLabel(passage)}</span>
        <Badge tone={passage.origin === 'retrieval' ? 'neutral' : 'info'}>{passage.origin}</Badge>
        {passage.hop !== null && <Badge tone="neutral">hop {passage.hop}</Badge>}
        {inChain && <Badge tone="accent">evidence chain</Badge>}
        {cited && <Badge tone="success">cited</Badge>}
        <span className="text-xs text-muted">
          final <span className="font-semibold text-fg tabular-nums">{fmtScore(passage.final_score)}</span>
        </span>
      </div>
      <div className="mt-3 grid gap-x-6 gap-y-1.5 sm:grid-cols-2">
        {SCORE_SERIES.map((s) => (
          <ScoreBar key={s.key} label={s.label} value={passage[s.key]} color={s.color} compact />
        ))}
      </div>
      <p className={cx('mt-3 text-sm leading-relaxed whitespace-pre-wrap text-fg/75', !expanded && 'line-clamp-3')}>
        {passage.text}
      </p>
      <button type="button" className="mt-1 inline-flex items-center gap-1 text-xs text-indigo-300 hover:text-indigo-200" onClick={() => setExpanded((v) => !v)}>
        {expanded ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
        {expanded ? 'Show less' : 'Show full passage'}
      </button>
    </div>
  )
}
