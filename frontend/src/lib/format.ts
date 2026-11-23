// Formatting helpers shared across pages.

export function fmtNumber(n: number | null | undefined, digits = 0): string {
  if (n === null || n === undefined || Number.isNaN(n)) return '—'
  return n.toLocaleString(undefined, { maximumFractionDigits: digits, minimumFractionDigits: 0 })
}

export function fmtScore(n: number | null | undefined, digits = 3): string {
  if (n === null || n === undefined || Number.isNaN(n)) return '—'
  return n.toFixed(digits)
}

export function fmtPct(n: number | null | undefined, digits = 1): string {
  if (n === null || n === undefined || Number.isNaN(n)) return '—'
  return `${(n * 100).toFixed(digits)}%`
}

export function fmtMs(n: number | null | undefined): string {
  if (n === null || n === undefined || Number.isNaN(n)) return '—'
  if (n >= 1000) return `${(n / 1000).toFixed(2)} s`
  if (n >= 10) return `${n.toFixed(0)} ms`
  return `${n.toFixed(2)} ms`
}

export function fmtBytes(n: number | null | undefined): string {
  if (n === null || n === undefined || Number.isNaN(n)) return '—'
  const units = ['B', 'KB', 'MB', 'GB', 'TB']
  let v = n
  let i = 0
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024
    i++
  }
  return `${v.toFixed(i === 0 ? 0 : v >= 100 ? 0 : 1)} ${units[i]}`
}

/** Parse backend timestamps. Some lack a timezone (e.g. finished_at) and are read as local time. */
export function parseDate(s: string | null | undefined): Date | null {
  if (!s) return null
  const d = new Date(s)
  return Number.isNaN(d.getTime()) ? null : d
}

export function fmtDate(s: string | null | undefined): string {
  const d = parseDate(s)
  if (!d) return '—'
  return d.toLocaleString(undefined, { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' })
}

export function fmtRelative(s: string | null | undefined): string {
  const d = parseDate(s)
  if (!d) return '—'
  const diff = (Date.now() - d.getTime()) / 1000
  if (diff < 0) return fmtDate(s)
  if (diff < 60) return 'just now'
  if (diff < 3600) return `${Math.floor(diff / 60)} min ago`
  if (diff < 86400) return `${Math.floor(diff / 3600)} h ago`
  return fmtDate(s)
}

export function humanize(key: string): string {
  return key.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase())
}

export function truthy(v: boolean | number | null | undefined): boolean {
  return v === true || (typeof v === 'number' && v !== 0)
}

export function clamp01(n: number | null | undefined): number {
  if (n === null || n === undefined || Number.isNaN(n)) return 0
  return Math.max(0, Math.min(1, n))
}

export function cx(...classes: Array<string | false | null | undefined>): string {
  return classes.filter(Boolean).join(' ')
}

/** Render any JSON-ish value compactly for key/value detail tables. */
export function fmtValue(v: unknown): string {
  if (v === null || v === undefined) return '—'
  if (typeof v === 'number') return Number.isInteger(v) ? fmtNumber(v) : fmtScore(v, 4)
  if (typeof v === 'boolean') return v ? 'yes' : 'no'
  if (typeof v === 'string') return v
  if (Array.isArray(v)) return v.length === 0 ? '[]' : v.map((x) => (typeof x === 'object' ? JSON.stringify(x) : String(x))).join(', ')
  return JSON.stringify(v)
}
