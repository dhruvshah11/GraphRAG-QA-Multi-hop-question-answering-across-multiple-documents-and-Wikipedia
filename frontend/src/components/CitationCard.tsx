import type { ReactNode } from 'react'
import { ExternalLink, FileText, Globe } from 'lucide-react'
import type { Citation } from '../types/api'
import { cx, fmtScore } from '../lib/format'
import { Modal } from './Overlay'
import { ScoreBar } from './ScoreBar'
import { Badge } from './Badge'
import { STRATEGY_COLORS } from '../lib/constants'

/** Inline [n] chip that opens a citation. */
export function CitationChip({ index, onClick, className }: { index: number; onClick?: (index: number) => void; className?: string }) {
  return (
    <button
      type="button"
      onClick={() => onClick?.(index)}
      title={`Open citation [${index}]`}
      className={cx(
        'mx-0.5 inline-flex h-5 min-w-5 items-center justify-center rounded border border-indigo-500/40 bg-indigo-500/15 px-1 align-middle font-mono text-[11px] font-medium text-indigo-200 transition-colors hover:bg-indigo-500/30',
        className,
      )}
    >
      {index}
    </button>
  )
}

export function sourceLabel(c: { document_name: string; page_number: number | null; page_end?: number | null; section?: string | null }) {
  const page =
    c.page_number !== null && c.page_number !== undefined
      ? c.page_end && c.page_end !== c.page_number
        ? `pp. ${c.page_number}–${c.page_end}`
        : `p. ${c.page_number}`
      : null
  return [c.document_name, page, c.section].filter(Boolean).join(' · ')
}

export function CitationCard({ citation, onOpen }: { citation: Citation; onOpen: (index: number) => void }) {
  const isWiki = citation.source_type === 'wikipedia' || !!citation.url
  return (
    <button
      type="button"
      onClick={() => onOpen(citation.index)}
      className="group w-full rounded-lg border border-line bg-surface-2/50 p-3.5 text-left transition-colors hover:border-indigo-500/40 hover:bg-surface-2"
    >
      <div className="flex items-start gap-3">
        <span className="mt-0.5 flex h-6 w-6 shrink-0 items-center justify-center rounded-md bg-indigo-500/15 font-mono text-xs font-semibold text-indigo-200">
          {citation.index}
        </span>
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-1.5 text-xs text-muted">
            {isWiki ? <Globe size={12} /> : <FileText size={12} />}
            <span className="truncate">{sourceLabel(citation)}</span>
          </div>
          <p className="mt-1.5 line-clamp-3 text-sm leading-relaxed text-fg/90">“{citation.supporting_sentence}”</p>
          <div className="mt-2 flex flex-wrap gap-1.5">
            <Badge tone="neutral">retrieval {fmtScore(citation.retrieval_score, 2)}</Badge>
            {citation.graph_score !== null && <Badge tone="neutral">graph {fmtScore(citation.graph_score, 2)}</Badge>}
            {citation.gnn_score !== null && <Badge tone="neutral">gnn {fmtScore(citation.gnn_score, 2)}</Badge>}
            <Badge tone={(citation.support_score ?? 0) >= 0.5 ? 'success' : 'warning'}>
              support {fmtScore(citation.support_score, 2)}
            </Badge>
          </div>
        </div>
      </div>
    </button>
  )
}

/** Passage text with [start, end) highlighted. Offsets are validated against the text. */
export function HighlightedText({ text, start, end }: { text: string; start: number | null | undefined; end: number | null | undefined }) {
  const valid =
    typeof start === 'number' && typeof end === 'number' && start >= 0 && end > start && end <= text.length
  if (!valid) return <p className="text-sm leading-relaxed whitespace-pre-wrap text-fg/85">{text}</p>
  return (
    <p className="text-sm leading-relaxed whitespace-pre-wrap text-fg/70">
      {text.slice(0, start)}
      <mark className="rounded-sm bg-amber-400/25 px-0.5 text-amber-50 ring-1 ring-amber-400/40">{text.slice(start, end)}</mark>
      {text.slice(end)}
    </p>
  )
}

export function CitationModal({ citation, onClose }: { citation: Citation | null; onClose: () => void }) {
  return (
    <Modal
      open={citation !== null}
      onClose={onClose}
      className="max-w-3xl"
      title={citation ? `Citation [${citation.index}]` : ''}
      subtitle={citation ? sourceLabel(citation) : undefined}
    >
      {citation && (
        <div className="space-y-5">
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
            <div className="space-y-1.5 text-sm">
              <Row k="Document" v={citation.document_name} />
              <Row k="Page" v={citation.page_number === null ? '—' : citation.page_end && citation.page_end !== citation.page_number ? `${citation.page_number}–${citation.page_end}` : String(citation.page_number)} />
              <Row k="Section" v={citation.section ?? '—'} />
              <Row k="Source" v={citation.source_type} />
              <Row k="Chunk" v={<span className="mono text-muted">{citation.chunk_id}</span>} />
              {citation.url && (
                <Row
                  k="URL"
                  v={
                    <a href={citation.url} target="_blank" rel="noreferrer" className="inline-flex items-center gap-1 text-indigo-300 hover:underline">
                      {citation.url} <ExternalLink size={12} />
                    </a>
                  }
                />
              )}
            </div>
            <div className="space-y-2">
              <ScoreBar label="Retrieval" value={citation.retrieval_score} color={STRATEGY_COLORS.hybrid} />
              <ScoreBar label="Graph" value={citation.graph_score} color={STRATEGY_COLORS.graph} />
              <ScoreBar label="GNN" value={citation.gnn_score} color={STRATEGY_COLORS.gnn} />
              <ScoreBar label="Support" value={citation.support_score} color="#34d399" />
            </div>
          </div>
          {citation.claim && (
            <div>
              <p className="label">Claim supported</p>
              <p className="text-sm text-fg">{citation.claim}</p>
            </div>
          )}
          <div>
            <p className="label">Full passage (supporting sentence highlighted)</p>
            <div className="rounded-lg border border-line bg-bg/60 p-4">
              <HighlightedText text={citation.passage_text} start={citation.sentence_start} end={citation.sentence_end} />
            </div>
          </div>
        </div>
      )}
    </Modal>
  )
}

function Row({ k, v }: { k: string; v: ReactNode }) {
  return (
    <div className="flex gap-3">
      <span className="w-20 shrink-0 text-muted">{k}</span>
      <span className="min-w-0 break-words text-fg">{v}</span>
    </div>
  )
}
