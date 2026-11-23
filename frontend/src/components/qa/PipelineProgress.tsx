import { useEffect, useState } from 'react'
import { Check } from 'lucide-react'
import { PROGRESS_STAGES, QUERY_STAGE_ORDER, STAGE_LABELS } from '../../lib/constants'
import { cx, fmtMs } from '../../lib/format'
import { Spinner } from '../Feedback'

/**
 * While a request is in flight the stages are walked through on a timer (the
 * backend does not stream progress); the last stage waits until the response
 * arrives. No numbers are shown until real timings are available.
 */
export function PipelineRunning({ openDomain }: { openDomain: boolean }) {
  const stages = openDomain
    ? ['wikipedia_retrieval', ...PROGRESS_STAGES]
    : PROGRESS_STAGES
  const [active, setActive] = useState(0)
  useEffect(() => {
    setActive(0)
    const genIndex = stages.indexOf('answer_generation')
    const id = window.setInterval(() => {
      setActive((i) => (i < genIndex ? i + 1 : i))
    }, openDomain ? 900 : 260)
    return () => window.clearInterval(id)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [openDomain])

  return (
    <ol className="grid gap-1.5 sm:grid-cols-2 lg:grid-cols-4">
      {stages.map((s, i) => {
        const done = i < active
        const current = i === active
        return (
          <li
            key={s}
            className={cx(
              'flex items-center gap-2 rounded-md border px-2.5 py-1.5 text-xs transition-colors',
              current ? 'border-indigo-500/40 bg-indigo-500/10 text-indigo-100' : done ? 'border-line text-muted' : 'border-line/60 text-subtle',
            )}
          >
            {current ? <Spinner size={12} /> : done ? <Check size={12} className="text-emerald-400" /> : <span className="h-3 w-3 rounded-full border border-line-strong" />}
            {STAGE_LABELS[s] ?? s}
          </li>
        )
      })}
    </ol>
  )
}

/** Measured per-stage timings (timings_ms) as horizontal bars. */
export function StageTimings({ timings, total }: { timings: Record<string, number>; total: number | null }) {
  const entries = Object.entries(timings)
  entries.sort((a, b) => {
    const ia = QUERY_STAGE_ORDER.indexOf(a[0])
    const ib = QUERY_STAGE_ORDER.indexOf(b[0])
    return (ia === -1 ? 99 : ia) - (ib === -1 ? 99 : ib)
  })
  const max = Math.max(1, ...entries.map(([, v]) => v))
  return (
    <div className="space-y-1.5">
      {entries.map(([k, v]) => (
        <div key={k} className="flex items-center gap-3 text-xs">
          <span className="w-36 shrink-0 truncate text-muted">{STAGE_LABELS[k] ?? k}</span>
          <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-surface-3">
            <div className="h-full rounded-full bg-indigo-400/80" style={{ width: `${Math.max(1.5, (v / max) * 100)}%` }} />
          </div>
          <span className="w-16 shrink-0 text-right text-fg tabular-nums">{fmtMs(v)}</span>
        </div>
      ))}
      {total !== null && (
        <div className="flex items-center justify-between border-t border-line pt-2 text-xs">
          <span className="text-muted">Total (end-to-end)</span>
          <span className="font-semibold text-fg tabular-nums">{fmtMs(total)}</span>
        </div>
      )}
    </div>
  )
}
