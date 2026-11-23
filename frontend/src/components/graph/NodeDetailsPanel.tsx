import { Expand, X } from 'lucide-react'
import type { GraphEdge, GraphNode } from '../../types/api'
import { EDGE_TYPE_STYLES, NODE_TYPE_COLORS, entityColor } from '../../lib/constants'
import { fmtValue, humanize } from '../../lib/format'
import { Badge } from '../Badge'
import { ScoreBar } from '../ScoreBar'
import { SCORE_SERIES } from '../../lib/constants'
import { Spinner } from '../Feedback'

interface Props {
  node: GraphNode
  edges: GraphEdge[]
  nodesById: Map<string, GraphNode>
  onClose: () => void
  onExpand: () => void
  expanding: boolean
  expandError: string | null
  onSelect: (id: string) => void
}

const SCORE_KEYS = new Set<string>(SCORE_SERIES.map((s) => s.key))
const HIDDEN = new Set(['preview'])

export function NodeDetailsPanel({ node, edges, nodesById, onClose, onExpand, expanding, expandError, onSelect }: Props) {
  const color = node.type === 'entity' ? entityColor(node.data.category) : (NODE_TYPE_COLORS[node.type] ?? '#94a3b8')
  const preview = typeof node.data.preview === 'string' ? node.data.preview : null
  const scores = SCORE_SERIES.filter((s) => typeof node.data[s.key] === 'number')
  const fields = Object.entries(node.data).filter(([k]) => !HIDDEN.has(k) && !SCORE_KEYS.has(k))
  const incident = edges.filter((e) => e.source === node.id || e.target === node.id)

  return (
    <div className="flex h-full flex-col">
      <div className="flex items-start justify-between gap-2 border-b border-line px-4 py-3">
        <div className="min-w-0">
          <div className="flex items-center gap-2">
            <Badge color={color}>{node.type}</Badge>
            {typeof node.data.category === 'string' && <Badge color={color}>{node.data.category}</Badge>}
            {node.highlighted && <Badge tone="accent">evidence</Badge>}
          </div>
          <p className="mt-1.5 text-sm font-semibold break-words text-fg">{node.label}</p>
          <p className="mono mt-0.5 break-all text-subtle">{node.id}</p>
        </div>
        <button type="button" className="btn-ghost btn-sm -mr-2" onClick={onClose} aria-label="Close details">
          <X size={15} />
        </button>
      </div>
      <div className="min-h-0 flex-1 space-y-4 overflow-y-auto px-4 py-3">
        <button type="button" className="btn-secondary btn-sm w-full" onClick={onExpand} disabled={expanding || node.type === 'question'}>
          {expanding ? <Spinner size={13} /> : <Expand size={13} />} Expand neighbors
        </button>
        {node.type === 'question' && <p className="text-[11px] text-subtle">The question node is query-specific and has no corpus neighbors to expand.</p>}
        {expandError && <p className="text-xs text-rose-300">{expandError}</p>}

        {scores.length > 0 && (
          <div className="space-y-1.5">
            <p className="label">Retrieval scores</p>
            {scores.map((s) => (
              <ScoreBar key={s.key} label={s.label} value={node.data[s.key] as number} color={s.color} compact />
            ))}
          </div>
        )}

        {preview && (
          <div>
            <p className="label">Passage preview</p>
            <p className="rounded-md border border-line bg-bg/60 p-2.5 text-xs leading-relaxed whitespace-pre-wrap text-fg/80">{preview}</p>
          </div>
        )}

        {fields.length > 0 && (
          <div>
            <p className="label">Attributes</p>
            <dl className="space-y-1 text-xs">
              {fields.map(([k, v]) => (
                <div key={k} className="flex gap-2">
                  <dt className="w-28 shrink-0 text-muted">{humanize(k)}</dt>
                  <dd className="min-w-0 break-words text-fg">{fmtValue(v)}</dd>
                </div>
              ))}
            </dl>
          </div>
        )}

        <div>
          <p className="label">Connections in view ({incident.length})</p>
          {incident.length === 0 ? (
            <p className="text-xs text-subtle">No visible edges.</p>
          ) : (
            <ul className="space-y-1">
              {incident.slice(0, 60).map((e) => {
                const otherId = e.source === node.id ? e.target : e.source
                const other = nodesById.get(otherId)
                const style = EDGE_TYPE_STYLES[e.type]
                const rel = typeof e.data.relation === 'string' ? e.data.relation : null
                return (
                  <li key={e.id}>
                    <button type="button" className="flex w-full items-center gap-2 rounded px-1.5 py-1 text-left text-xs hover:bg-surface-2" onClick={() => onSelect(otherId)}>
                      <span className="shrink-0 font-mono text-[10px]" style={{ color: style?.color }}>
                        {e.source === node.id ? '→' : '←'} {rel ?? e.type}
                      </span>
                      <span className="truncate text-fg/85">{other?.label ?? otherId}</span>
                    </button>
                  </li>
                )
              })}
              {incident.length > 60 && <li className="text-[11px] text-subtle">+{incident.length - 60} more</li>}
            </ul>
          )}
        </div>
      </div>
    </div>
  )
}
