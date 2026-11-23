import { clamp01, cx, fmtScore } from '../lib/format'

interface ScoreBarProps {
  label?: string
  value: number | null | undefined
  color?: string
  /** Maximum value represented by a full bar (default 1). */
  max?: number
  className?: string
  digits?: number
  compact?: boolean
}

export function ScoreBar({ label, value, color = '#818cf8', max = 1, className, digits = 3, compact }: ScoreBarProps) {
  const missing = value === null || value === undefined || Number.isNaN(value)
  const pct = missing ? 0 : clamp01((value as number) / max) * 100
  return (
    <div className={cx('flex items-center gap-2', className)}>
      {label && <span className={cx('shrink-0 text-xs text-muted', compact ? 'w-14' : 'w-24')}>{label}</span>}
      <div className="relative h-1.5 min-w-10 flex-1 overflow-hidden rounded-full bg-surface-3">
        {!missing && (
          <div className="h-full rounded-full transition-[width] duration-500" style={{ width: `${pct}%`, background: color }} />
        )}
      </div>
      <span className={cx('w-12 shrink-0 text-right text-xs tabular-nums', missing ? 'text-subtle' : 'text-fg')}>
        {missing ? 'n/a' : fmtScore(value, digits)}
      </span>
    </div>
  )
}
