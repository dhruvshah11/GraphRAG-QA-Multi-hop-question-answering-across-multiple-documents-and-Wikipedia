import type { ReactNode } from 'react'
import { AlertTriangle, Inbox, Loader2, RotateCw, XCircle } from 'lucide-react'
import { cx } from '../lib/format'

export function Spinner({ className, size = 16 }: { className?: string; size?: number }) {
  return <Loader2 size={size} className={cx('animate-spin text-accent-strong', className)} aria-label="Loading" />
}

export function Skeleton({ className }: { className?: string }) {
  return <div className={cx('skeleton h-4 w-full', className)} />
}

/** A block of skeleton rows used while a card's data loads. */
export function SkeletonBlock({ rows = 4, className }: { rows?: number; className?: string }) {
  return (
    <div className={cx('space-y-2.5', className)} aria-busy="true">
      {Array.from({ length: rows }, (_, i) => (
        <Skeleton key={i} className={i % 3 === 2 ? 'w-2/3' : i % 3 === 1 ? 'w-5/6' : 'w-full'} />
      ))}
    </div>
  )
}

export function SkeletonCards({ count = 4, className }: { count?: number; className?: string }) {
  return (
    <div className={cx('grid gap-3', className)}>
      {Array.from({ length: count }, (_, i) => (
        <div key={i} className="rounded-xl border border-line bg-surface p-4">
          <Skeleton className="h-3 w-1/2" />
          <Skeleton className="mt-3 h-6 w-1/3" />
        </div>
      ))}
    </div>
  )
}

interface EmptyStateProps {
  title: string
  description?: ReactNode
  icon?: ReactNode
  action?: ReactNode
  className?: string
}

export function EmptyState({ title, description, icon, action, className }: EmptyStateProps) {
  return (
    <div className={cx('flex flex-col items-center justify-center px-6 py-10 text-center', className)}>
      <div className="mb-3 flex h-10 w-10 items-center justify-center rounded-full border border-line bg-surface-2 text-subtle">
        {icon ?? <Inbox size={18} />}
      </div>
      <p className="text-sm font-medium text-fg">{title}</p>
      {description && <div className="mt-1 max-w-md text-xs leading-relaxed text-muted">{description}</div>}
      {action && <div className="mt-4">{action}</div>}
    </div>
  )
}

interface ErrorBannerProps {
  message: string
  title?: string
  onRetry?: () => void
  onDismiss?: () => void
  className?: string
}

export function ErrorBanner({ message, title = 'Request failed', onRetry, onDismiss, className }: ErrorBannerProps) {
  return (
    <div
      role="alert"
      className={cx('flex items-start gap-3 rounded-lg border border-rose-500/30 bg-rose-500/[0.07] px-4 py-3', className)}
    >
      <XCircle size={18} className="mt-0.5 shrink-0 text-rose-400" />
      <div className="min-w-0 flex-1">
        <p className="text-sm font-medium text-rose-200">{title}</p>
        <p className="mt-0.5 text-sm break-words whitespace-pre-wrap text-rose-200/80">{message}</p>
      </div>
      <div className="flex shrink-0 gap-1">
        {onRetry && (
          <button type="button" className="btn-ghost btn-sm" onClick={onRetry}>
            <RotateCw size={13} /> Retry
          </button>
        )}
        {onDismiss && (
          <button type="button" className="btn-ghost btn-sm" onClick={onDismiss}>
            Dismiss
          </button>
        )}
      </div>
    </div>
  )
}

export function WarningPanel({ title, children, className }: { title: string; children?: ReactNode; className?: string }) {
  return (
    <div className={cx('rounded-lg border border-amber-500/35 bg-amber-500/[0.08] px-4 py-3', className)}>
      <div className="flex items-start gap-3">
        <AlertTriangle size={18} className="mt-0.5 shrink-0 text-amber-400" />
        <div className="min-w-0 flex-1">
          <p className="text-sm font-semibold text-amber-200">{title}</p>
          {children && <div className="mt-1 text-sm text-amber-100/80">{children}</div>}
        </div>
      </div>
    </div>
  )
}

export function ProgressBar({ value, className, indeterminate }: { value?: number | null; className?: string; indeterminate?: boolean }) {
  const pct = Math.max(0, Math.min(1, value ?? 0)) * 100
  return (
    <div className={cx('relative h-1.5 w-full overflow-hidden rounded-full bg-surface-3', className)}>
      {indeterminate ? (
        <div className="skeleton absolute inset-0 rounded-full opacity-80" />
      ) : (
        <div className="h-full rounded-full bg-accent transition-[width] duration-300" style={{ width: `${pct}%` }} />
      )}
    </div>
  )
}
