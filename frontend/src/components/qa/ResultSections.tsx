import { Fragment } from 'react'
import { Link } from 'react-router-dom'
import { ArrowDown, Brain, Info, Link2, Network, Quote } from 'lucide-react'
import type { EvidenceChainStep, QAResult, ReasoningStep } from '../../types/api'
import { STRATEGY_COLORS, STRATEGY_LABELS } from '../../lib/constants'
import { clamp01, cx, fmtPct, fmtScore, humanize } from '../../lib/format'
import { Badge } from '../Badge'
import { CitationChip, sourceLabel } from '../CitationCard'
import { ScoreBar } from '../ScoreBar'
import { EmptyState } from '../Feedback'

type OpenCitation = (index: number) => void

export function Chips({ ids, onOpen }: { ids: number[] | undefined; onOpen: OpenCitation }) {
  if (!ids?.length) return null
  return (
    <span className="whitespace-nowrap">
      {ids.map((n) => (
        <CitationChip key={n} index={n} onClick={onOpen} />
      ))}
    </span>
  )
}

/** Render "[n]" markers inside generated text as clickable citation chips. */
export function TextWithChips({ text, onOpen }: { text: string; onOpen: OpenCitation }) {
  const parts = text.split(/(\[\d+\])/g)
  return (
    <>
      {parts.map((p, i) => {
        const m = /^\[(\d+)\]$/.exec(p)
        return m ? <CitationChip key={i} index={Number(m[1])} onClick={onOpen} /> : <Fragment key={i}>{p}</Fragment>
      })}
    </>
  )
}

export function ReasoningSteps({ steps, onOpen }: { steps: ReasoningStep[]; onOpen: OpenCitation }) {
  if (!steps.length) return <EmptyState title="No reasoning steps" description="The generator did not return a reasoning trace." />
  return (
    <ol className="space-y-2.5">
      {steps.map((s) => {
        const isFact = s.kind === 'fact'
        return (
          <li
            key={s.step}
            className={cx(
              'flex gap-3 rounded-lg border px-3.5 py-3',
              isFact ? 'border-line bg-surface-2/40' : 'border-dashed border-violet-500/40 bg-violet-500/[0.06]',
            )}
          >
            <span className="mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-surface-3 text-[11px] font-semibold text-muted">{s.step}</span>
            <div className="min-w-0 flex-1">
              <div className="mb-1 flex items-center gap-1.5 text-[11px] font-semibold tracking-wide">
                {isFact ? (
                  <span className="inline-flex items-center gap-1 text-sky-300">
                    <Quote size={11} /> RETRIEVED FACT
                  </span>
                ) : (
                  <span className="inline-flex items-center gap-1 text-violet-300">
                    <Brain size={11} /> Inference
                  </span>
                )}
              </div>
              <p className={cx('text-sm leading-relaxed', isFact ? 'text-fg/90' : 'text-violet-100/90 italic')}>
                {isFact ? (
                  <>“{s.text}”</>
                ) : (
                  <TextWithChips text={s.text} onOpen={onOpen} />
                )}
                <Chips ids={isFact ? s.citations : s.citations.filter((n) => !s.text.includes(`[${n}]`))} onOpen={onOpen} />
              </p>
            </div>
          </li>
        )
      })}
    </ol>
  )
}

export function EvidenceChain({ chain, queryId, onOpen }: { chain: EvidenceChainStep[]; queryId: string; onOpen: OpenCitation }) {
  if (!chain.length) return <EmptyState title="No evidence chain" description="No multi-hop chain was built for this answer." />
  return (
    <div>
      <ol className="relative">
        {chain.map((s, i) => (
          <Fragment key={`${s.step}-${s.chunk_id}`}>
            {i > 0 && (
              <li className="flex items-center gap-2 py-2 pl-3" aria-hidden={!s.via_entity_name}>
                <ArrowDown size={14} className="text-subtle" />
                {s.via_entity_name ? (
                  <span className="inline-flex items-center gap-1.5 rounded-full border border-emerald-500/40 bg-emerald-500/10 px-2.5 py-0.5 text-xs font-medium text-emerald-200">
                    <Link2 size={11} /> bridge entity: {s.via_entity_name}
                  </span>
                ) : (
                  <span className="text-xs text-subtle">next hop</span>
                )}
              </li>
            )}
            <li className="rounded-lg border border-indigo-500/25 bg-surface-2/40 p-3.5">
              <div className="flex flex-wrap items-center gap-2">
                <span className="flex h-6 w-6 items-center justify-center rounded-md bg-indigo-500/20 text-xs font-semibold text-indigo-200">{s.step}</span>
                <span className="min-w-0 flex-1 truncate text-sm font-medium text-fg">{sourceLabel(s)}</span>
                {s.evidence_index !== null && <CitationChip index={s.evidence_index} onClick={onOpen} />}
                <span className="text-xs text-muted">
                  relevance <span className="font-semibold text-fg tabular-nums">{fmtScore(s.relevance)}</span>
                </span>
              </div>
              {s.reason && <p className="mt-2 text-xs text-muted">{s.reason}</p>}
              <p className="mt-2 line-clamp-3 text-sm leading-relaxed text-fg/75" title={s.text}>
                {s.text}
              </p>
            </li>
          </Fragment>
        ))}
      </ol>
      <div className="mt-3 flex justify-end">
        <Link to={`/graph?query=${encodeURIComponent(queryId)}&view=chain`} className="btn-secondary btn-sm">
          <Network size={13} /> Open in Graph Explorer
        </Link>
      </div>
    </div>
  )
}

function Gauge({ value }: { value: number }) {
  const v = clamp01(value)
  const r = 52
  const c = Math.PI * r
  const color = v >= 0.66 ? '#34d399' : v >= 0.4 ? '#fbbf24' : '#fb7185'
  return (
    <svg viewBox="0 0 128 74" className="w-40" role="img" aria-label={`Confidence ${fmtPct(v, 0)}`}>
      <path d="M 12 66 A 52 52 0 0 1 116 66" fill="none" stroke="#1a2231" strokeWidth="10" strokeLinecap="round" />
      <path
        d="M 12 66 A 52 52 0 0 1 116 66"
        fill="none"
        stroke={color}
        strokeWidth="10"
        strokeLinecap="round"
        strokeDasharray={`${c * v} ${c}`}
        style={{ transition: 'stroke-dasharray 600ms ease' }}
      />
      <text x="64" y="58" textAnchor="middle" fill="#e5e9f0" fontSize="20" fontWeight="600">
        {Math.round(v * 100)}%
      </text>
    </svg>
  )
}

export function ConfidencePanel({ result }: { result: QAResult }) {
  const cov = result.evidence_coverage
  const covered = new Set(cov?.covered ?? [])
  return (
    <div className="space-y-5">
      <div className="flex flex-col items-center gap-1">
        <Gauge value={result.confidence} />
        <span className="text-xs text-muted">Answer confidence</span>
      </div>
      <div className="space-y-2">
        {Object.entries(result.confidence_components ?? {}).map(([k, v]) => (
          <ScoreBar key={k} label={humanize(k)} value={v} color="#818cf8" digits={2} />
        ))}
      </div>
      {cov && (
        <div>
          <div className="mb-2 flex items-center justify-between">
            <span className="label mb-0">Evidence coverage of question terms</span>
            <span className="text-xs font-semibold text-fg tabular-nums">{fmtPct(cov.ratio, 0)}</span>
          </div>
          <div className="flex flex-wrap gap-1.5">
            {cov.question_terms.map((t) => (
              <Badge key={t} tone={covered.has(t) ? 'success' : 'danger'} title={covered.has(t) ? 'covered by evidence' : 'missing from evidence'}>
                {t}
              </Badge>
            ))}
            {cov.question_terms.length === 0 && <span className="text-xs text-subtle">No content terms detected.</span>}
          </div>
        </div>
      )}
    </div>
  )
}

export function RunMeta({ result }: { result: QAResult }) {
  const fallback = result.effective_strategy && result.effective_strategy !== result.strategy
  return (
    <div className="flex flex-wrap items-center gap-1.5 text-xs">
      <Badge color={STRATEGY_COLORS[result.strategy] ?? undefined}>requested: {STRATEGY_LABELS[result.strategy] ?? result.strategy}</Badge>
      {result.effective_strategy && (
        <Badge tone={fallback ? 'warning' : 'neutral'} title={fallback ? 'The backend used a different strategy than requested' : undefined}>
          effective: {STRATEGY_LABELS[result.effective_strategy] ?? result.effective_strategy}
        </Badge>
      )}
      {result.generator_backend && <Badge tone="neutral">generator: {result.generator_backend}</Badge>}
      <Badge tone="neutral">{result.mode === 'open_domain' ? 'open-domain' : 'multi-document'}</Badge>
      {result.top_k !== undefined && <Badge tone="neutral">top-k {result.top_k}</Badge>}
      {result.max_hops !== undefined && <Badge tone="neutral">hops {result.max_hops}</Badge>}
    </div>
  )
}

export function NotesList({ notes, warnings }: { notes: string[]; warnings: string[] }) {
  const all = [...(warnings ?? []).map((t) => ({ t, w: true })), ...(notes ?? []).map((t) => ({ t, w: false }))]
  if (!all.length) return null
  return (
    <ul className="space-y-1">
      {all.map((n, i) => (
        <li key={i} className="flex items-start gap-2 text-xs text-muted">
          <Info size={12} className={cx('mt-0.5 shrink-0', n.w ? 'text-amber-400/80' : 'text-subtle')} />
          <span>{n.t}</span>
        </li>
      ))}
    </ul>
  )
}
