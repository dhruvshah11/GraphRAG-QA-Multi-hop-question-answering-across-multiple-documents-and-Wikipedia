import { useApi } from '../hooks/useApi'
import { usePolling } from '../hooks/usePolling'
import { getHealth } from '../services/api'
import { cx, fmtRelative, humanize } from '../lib/format'
import { useState } from 'react'

const DOT: Record<string, string> = {
  ok: 'bg-emerald-400',
  ready: 'bg-emerald-400',
  degraded: 'bg-amber-400',
  untrained: 'bg-amber-400',
  error: 'bg-rose-500',
  down: 'bg-rose-500',
}

/** Live backend health: polls GET /api/health every 15 s. */
export function HealthIndicator() {
  const { data, error, refetch } = useApi((s) => getHealth(s), [])
  const [checkedAt, setCheckedAt] = useState(() => new Date().toISOString())
  const [expanded, setExpanded] = useState(false)
  usePolling(() => {
    void refetch(true).then(() => setCheckedAt(new Date().toISOString()))
  }, 15000)

  const status = error ? 'down' : (data?.status ?? 'checking')
  const label = error ? 'Backend unreachable' : data ? `API ${data.status}` : 'Checking…'

  return (
    <div className="px-2">
      <button type="button" onClick={() => setExpanded((v) => !v)} className="flex w-full items-center gap-2.5 text-left" title={error ?? undefined}>
        <span className="relative flex h-2.5 w-2.5">
          {status === 'ok' && <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-emerald-400/40" />}
          <span className={cx('relative inline-flex h-2.5 w-2.5 rounded-full', DOT[status] ?? 'bg-slate-500')} />
        </span>
        <span className="text-xs font-medium text-fg">{label}</span>
        <span className="ml-auto text-[11px] text-subtle">{fmtRelative(checkedAt)}</span>
      </button>
      {expanded && data && (
        <ul className="mt-3 space-y-1">
          {Object.entries(data.components).map(([name, c]) => (
            <li key={name} className="flex items-center gap-2 text-[11px]">
              <span className={cx('h-1.5 w-1.5 rounded-full', DOT[c.status] ?? 'bg-slate-500')} />
              <span className="text-muted">{humanize(name)}</span>
              <span className="ml-auto text-subtle">{c.status}</span>
            </li>
          ))}
        </ul>
      )}
      {expanded && error && <p className="mt-2 text-[11px] text-rose-300">{error}</p>}
    </div>
  )
}
