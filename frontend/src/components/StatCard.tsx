import type { ReactNode } from 'react'
import { cx } from '../lib/format'

interface StatCardProps {
  label: string
  value: ReactNode
  hint?: ReactNode
  icon?: ReactNode
  className?: string
}

export function StatCard({ label, value, hint, icon, className }: StatCardProps) {
  return (
    <div className={cx('rounded-xl border border-line bg-surface px-4 py-3.5', className)}>
      <div className="flex items-center justify-between gap-2">
        <span className="truncate text-xs font-medium text-muted">{label}</span>
        {icon && <span className="text-subtle">{icon}</span>}
      </div>
      <div className="mt-1.5 text-2xl font-semibold tracking-tight text-fg tabular-nums">{value}</div>
      {hint && <div className="mt-0.5 truncate text-xs text-subtle">{hint}</div>}
    </div>
  )
}
