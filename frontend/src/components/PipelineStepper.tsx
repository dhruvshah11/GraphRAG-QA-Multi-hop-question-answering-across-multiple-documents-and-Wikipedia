import type { ReactNode } from 'react'
import { Check, CircleDashed, CircleAlert, CircleDot } from 'lucide-react'
import type { PipelineStage } from '../types/api'
import { cx } from '../lib/format'

const STYLE: Record<string, { ring: string; text: string; line: string; icon: ReactNode; label: string }> = {
  ready: { ring: 'border-emerald-500/50 bg-emerald-500/15 text-emerald-300', text: 'text-emerald-300', line: 'bg-emerald-500/40', icon: <Check size={14} />, label: 'Ready' },
  degraded: { ring: 'border-amber-500/50 bg-amber-500/15 text-amber-300', text: 'text-amber-300', line: 'bg-amber-500/40', icon: <CircleAlert size={14} />, label: 'Degraded' },
  untrained: { ring: 'border-amber-500/50 bg-amber-500/10 text-amber-300', text: 'text-amber-300', line: 'bg-amber-500/30', icon: <CircleDot size={14} />, label: 'Untrained' },
  idle: { ring: 'border-slate-500/50 bg-slate-500/10 text-slate-400', text: 'text-slate-400', line: 'bg-slate-600/40', icon: <CircleDashed size={14} />, label: 'Idle' },
}

/** Horizontal stepper of the 9 pipeline stages with status colors (scrolls on narrow screens). */
export function PipelineStepper({ stages }: { stages: PipelineStage[] }) {
  return (
    <div className="overflow-x-auto pb-1">
      <ol className="flex min-w-[960px] items-start">
        {stages.map((s, i) => {
          const st = STYLE[s.status] ?? STYLE.idle
          return (
            <li key={s.key} className="relative flex flex-1 flex-col items-center px-1 text-center" title={`${s.tech}\n${s.detail}`}>
              {i > 0 && <span className={cx('absolute top-4 right-1/2 h-px w-full', st.line)} />}
              <span className={cx('relative z-10 flex h-8 w-8 items-center justify-center rounded-full border', st.ring)}>{st.icon}</span>
              <span className="mt-2 text-xs font-medium text-fg">{s.label}</span>
              <span className={cx('mt-0.5 text-[11px] font-medium', st.text)}>{st.label}</span>
              <span className="mt-1 line-clamp-2 px-1 text-[11px] leading-snug text-subtle">{s.tech}</span>
              <span className="mt-1 line-clamp-2 px-1 text-[11px] leading-snug text-muted">{s.detail}</span>
            </li>
          )
        })}
      </ol>
    </div>
  )
}
