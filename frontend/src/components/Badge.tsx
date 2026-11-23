import type { CSSProperties, ReactNode } from 'react'
import { cx } from '../lib/format'

export type Tone = 'neutral' | 'accent' | 'success' | 'warning' | 'danger' | 'info'

const TONES: Record<Tone, string> = {
  neutral: 'border-line-strong bg-surface-2 text-muted',
  accent: 'border-indigo-500/30 bg-indigo-500/10 text-indigo-300',
  success: 'border-emerald-500/30 bg-emerald-500/10 text-emerald-300',
  warning: 'border-amber-500/30 bg-amber-500/10 text-amber-300',
  danger: 'border-rose-500/30 bg-rose-500/10 text-rose-300',
  info: 'border-sky-500/30 bg-sky-500/10 text-sky-300',
}

interface BadgeProps {
  children: ReactNode
  tone?: Tone
  /** Custom hex color (overrides tone). */
  color?: string
  className?: string
  title?: string
  dot?: boolean
}

export function Badge({ children, tone = 'neutral', color, className, title, dot }: BadgeProps) {
  const style: CSSProperties | undefined = color
    ? { color, borderColor: `${color}55`, backgroundColor: `${color}1a` }
    : undefined
  return (
    <span
      title={title}
      style={style}
      className={cx(
        'inline-flex max-w-full items-center gap-1.5 truncate rounded-md border px-1.5 py-0.5 text-[11px] font-medium leading-4',
        !color && TONES[tone],
        className,
      )}
    >
      {dot && <span className="h-1.5 w-1.5 shrink-0 rounded-full bg-current" />}
      {children}
    </span>
  )
}

/** Map backend status strings to badge tones. */
export function statusTone(status: string | null | undefined): Tone {
  switch (status) {
    case 'ready':
    case 'ok':
    case 'processed':
    case 'completed':
      return 'success'
    case 'degraded':
    case 'untrained':
    case 'queued':
    case 'uploaded':
      return 'warning'
    case 'processing':
    case 'running':
      return 'info'
    case 'failed':
    case 'error':
    case 'unavailable':
      return 'danger'
    default:
      return 'neutral'
  }
}

export function StatusBadge({ status }: { status: string | null | undefined }) {
  return (
    <Badge tone={statusTone(status)} dot>
      {status ?? 'unknown'}
    </Badge>
  )
}
