import type { ReactNode } from 'react'
import { cx } from '../lib/format'

export interface TabItem<K extends string> {
  key: K
  label: ReactNode
  count?: number
  disabled?: boolean
}

interface TabsProps<K extends string> {
  items: TabItem<K>[]
  value: K
  onChange: (key: K) => void
  className?: string
  /** "underline" for page sections, "pill" for compact segmented controls. */
  variant?: 'underline' | 'pill'
}

export function Tabs<K extends string>({ items, value, onChange, className, variant = 'underline' }: TabsProps<K>) {
  if (variant === 'pill') {
    return (
      <div role="tablist" className={cx('inline-flex flex-wrap gap-0.5 rounded-lg border border-line bg-surface-2 p-0.5', className)}>
        {items.map((t) => (
          <button
            key={t.key}
            role="tab"
            type="button"
            aria-selected={t.key === value}
            disabled={t.disabled}
            onClick={() => onChange(t.key)}
            className={cx(
              'rounded-md px-2.5 py-1 text-xs font-medium transition-colors disabled:opacity-40',
              t.key === value ? 'bg-surface-3 text-fg shadow-sm' : 'text-muted hover:text-fg',
            )}
          >
            {t.label}
            {t.count !== undefined && <span className="ml-1.5 text-subtle tabular-nums">{t.count}</span>}
          </button>
        ))}
      </div>
    )
  }
  return (
    <div role="tablist" className={cx('flex gap-1 overflow-x-auto border-b border-line', className)}>
      {items.map((t) => (
        <button
          key={t.key}
          role="tab"
          type="button"
          aria-selected={t.key === value}
          disabled={t.disabled}
          onClick={() => onChange(t.key)}
          className={cx(
            '-mb-px border-b-2 px-3 py-2 text-sm font-medium whitespace-nowrap transition-colors disabled:opacity-40',
            t.key === value ? 'border-accent text-fg' : 'border-transparent text-muted hover:text-fg',
          )}
        >
          {t.label}
          {t.count !== undefined && (
            <span className="ml-1.5 rounded bg-surface-3 px-1.5 py-px text-[11px] text-muted tabular-nums">{t.count}</span>
          )}
        </button>
      ))}
    </div>
  )
}
