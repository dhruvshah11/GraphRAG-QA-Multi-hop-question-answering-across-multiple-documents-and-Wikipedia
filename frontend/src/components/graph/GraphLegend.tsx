import { useState } from 'react'
import { ChevronDown, ChevronUp } from 'lucide-react'
import { EDGE_TYPE_STYLES, ENTITY_CATEGORY_COLORS, NODE_TYPE_COLORS } from '../../lib/constants'
import { humanize } from '../../lib/format'

export function GraphLegend({ nodeCounts, edgeCounts, categories }: { nodeCounts: Record<string, number>; edgeCounts: Record<string, number>; categories: string[] }) {
  const [open, setOpen] = useState(() => typeof window !== 'undefined' && window.innerWidth >= 1700)
  return (
    <div className="w-56 rounded-lg border border-line-strong bg-surface/95 text-xs shadow-lg backdrop-blur">
      <button type="button" className="flex w-full items-center justify-between px-3 py-2 font-semibold text-fg" onClick={() => setOpen((v) => !v)}>
        Legend {open ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
      </button>
      {open && (
        <div className="space-y-3 border-t border-line px-3 py-2.5">
          <div className="space-y-1">
            <p className="text-[10px] font-semibold tracking-wider text-subtle uppercase">Nodes</p>
            {Object.entries(NODE_TYPE_COLORS).map(([t, c]) => (
              <div key={t} className="flex items-center gap-2">
                <span className="h-2.5 w-2.5 rounded-sm border" style={{ borderColor: c, background: `${c}33` }} />
                <span className="text-muted">{humanize(t)}</span>
                <span className="ml-auto text-subtle tabular-nums">{nodeCounts[t] ?? 0}</span>
              </div>
            ))}
          </div>
          {categories.length > 0 && (
            <div className="space-y-1">
              <p className="text-[10px] font-semibold tracking-wider text-subtle uppercase">Entity categories</p>
              <div className="flex flex-wrap gap-x-3 gap-y-1">
                {categories.map((c) => (
                  <span key={c} className="flex items-center gap-1.5 text-muted">
                    <span className="h-2 w-2 rounded-full" style={{ background: ENTITY_CATEGORY_COLORS[c] ?? '#86efac' }} />
                    {c.toLowerCase().replace(/_/g, ' ')}
                  </span>
                ))}
              </div>
            </div>
          )}
          <div className="space-y-1">
            <p className="text-[10px] font-semibold tracking-wider text-subtle uppercase">Edges</p>
            {Object.entries(EDGE_TYPE_STYLES).map(([t, s]) => (
              <div key={t} className="flex items-center gap-2">
                <svg width="22" height="6" aria-hidden>
                  <line x1="0" y1="3" x2="22" y2="3" stroke={s.color} strokeWidth="2" strokeDasharray={s.dash} />
                </svg>
                <span className="text-muted">{s.label}</span>
                <span className="ml-auto text-subtle tabular-nums">{edgeCounts[t] ?? 0}</span>
              </div>
            ))}
            <div className="flex items-center gap-2 pt-1">
              <svg width="22" height="6" aria-hidden>
                <line x1="0" y1="3" x2="22" y2="3" stroke="#a5b4fc" strokeWidth="3" />
              </svg>
              <span className="text-indigo-200">Evidence chain (highlighted)</span>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
