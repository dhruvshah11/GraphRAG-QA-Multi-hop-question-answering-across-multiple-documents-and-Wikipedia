import type { ReactNode } from 'react'
import { cx } from '../lib/format'

interface CardProps {
  title?: ReactNode
  subtitle?: ReactNode
  actions?: ReactNode
  icon?: ReactNode
  children?: ReactNode
  className?: string
  bodyClassName?: string
  /** Remove body padding (for tables / full-bleed charts). */
  flush?: boolean
}

export function Card({ title, subtitle, actions, icon, children, className, bodyClassName, flush }: CardProps) {
  const hasHeader = title !== undefined || actions !== undefined
  return (
    <section className={cx('min-w-0 rounded-xl border border-line bg-surface', className)}>
      {hasHeader && (
        <header className="flex flex-wrap items-start justify-between gap-3 border-b border-line px-4 py-3 sm:px-5">
          <div className="flex min-w-0 items-start gap-2.5">
            {icon && <span className="mt-0.5 text-accent-strong">{icon}</span>}
            <div className="min-w-0">
              {title && <h2 className="text-sm font-semibold text-fg">{title}</h2>}
              {subtitle && <p className="mt-0.5 text-xs text-muted">{subtitle}</p>}
            </div>
          </div>
          {actions && <div className="flex flex-wrap items-center gap-2">{actions}</div>}
        </header>
      )}
      <div className={cx(flush ? '' : 'p-4 sm:p-5', bodyClassName)}>{children}</div>
    </section>
  )
}
