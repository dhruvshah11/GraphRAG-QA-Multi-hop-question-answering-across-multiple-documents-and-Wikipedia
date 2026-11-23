import { useRef, useState, type DragEvent } from 'react'
import { CheckCircle2, CircleAlert, CopyCheck, UploadCloud, X } from 'lucide-react'
import { errorMessage, uploadDocuments } from '../../services/api'
import type { UploadResult } from '../../types/api'
import { cx, fmtBytes } from '../../lib/format'
import { ProgressBar } from '../Feedback'

interface Props {
  maxMb: number
  allowed: string[]
  /** Called with every result returned by the backend (to start polling jobs). */
  onUploaded: (results: UploadResult[]) => void
}

interface Rejected {
  name: string
  reason: string
}

export function UploadDropzone({ maxMb, allowed, onUploaded }: Props) {
  const inputRef = useRef<HTMLInputElement>(null)
  const [dragging, setDragging] = useState(false)
  const [progress, setProgress] = useState<number | null>(null)
  const [uploadingNames, setUploadingNames] = useState<string[]>([])
  const [rejected, setRejected] = useState<Rejected[]>([])
  const [results, setResults] = useState<UploadResult[]>([])
  const [error, setError] = useState<string | null>(null)
  const abortRef = useRef<(() => void) | null>(null)

  const busy = progress !== null

  const handleFiles = async (list: FileList | File[]) => {
    const files = Array.from(list)
    if (!files.length || busy) return
    const ok: File[] = []
    const bad: Rejected[] = []
    for (const f of files) {
      const ext = f.name.includes('.') ? f.name.slice(f.name.lastIndexOf('.')).toLowerCase() : ''
      if (!allowed.includes(ext)) bad.push({ name: f.name, reason: `unsupported type (allowed: ${allowed.join(', ')})` })
      else if (f.size > maxMb * 1024 * 1024) bad.push({ name: f.name, reason: `${fmtBytes(f.size)} exceeds the ${maxMb} MB limit` })
      else ok.push(f)
    }
    setRejected(bad)
    setResults([])
    setError(null)
    if (!ok.length) return
    setUploadingNames(ok.map((f) => f.name))
    setProgress(0)
    const { promise, abort } = uploadDocuments(ok, setProgress)
    abortRef.current = abort
    try {
      const res = await promise
      setResults(res.results)
      onUploaded(res.results)
    } catch (err) {
      setError(errorMessage(err))
    } finally {
      setProgress(null)
      abortRef.current = null
      if (inputRef.current) inputRef.current.value = ''
    }
  }

  const onDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    setDragging(false)
    void handleFiles(e.dataTransfer.files)
  }

  return (
    <div>
      <div
        role="button"
        tabIndex={0}
        onClick={() => !busy && inputRef.current?.click()}
        onKeyDown={(e) => {
          if ((e.key === 'Enter' || e.key === ' ') && !busy) inputRef.current?.click()
        }}
        onDragOver={(e) => {
          e.preventDefault()
          setDragging(true)
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        className={cx(
          'flex cursor-pointer flex-col items-center justify-center rounded-xl border-2 border-dashed px-6 py-8 text-center transition-colors',
          dragging ? 'border-accent bg-accent-soft' : 'border-line-strong bg-surface-2/30 hover:border-indigo-500/50 hover:bg-surface-2/60',
          busy && 'cursor-progress',
        )}
      >
        <UploadCloud size={28} className={dragging ? 'text-indigo-300' : 'text-subtle'} />
        <p className="mt-3 text-sm font-medium text-fg">Drop documents here or click to browse</p>
        <p className="mt-1 text-xs text-muted">
          {allowed.join(', ')} · up to {maxMb} MB each · multiple files supported
        </p>
        <input
          ref={inputRef}
          type="file"
          multiple
          accept={allowed.join(',')}
          className="hidden"
          onChange={(e) => e.target.files && void handleFiles(e.target.files)}
        />
      </div>

      {busy && (
        <div className="mt-3 rounded-lg border border-line bg-surface-2/50 p-3">
          <div className="mb-2 flex items-center justify-between text-xs">
            <span className="truncate text-muted">
              Uploading {uploadingNames.length} file{uploadingNames.length > 1 ? 's' : ''}: {uploadingNames.join(', ')}
            </span>
            <span className="flex items-center gap-2">
              <span className="font-medium text-fg tabular-nums">{Math.round((progress ?? 0) * 100)}%</span>
              <button type="button" className="text-subtle hover:text-fg" onClick={() => abortRef.current?.()} aria-label="Cancel upload">
                <X size={14} />
              </button>
            </span>
          </div>
          <ProgressBar value={progress} />
          {progress !== null && progress >= 1 && <p className="mt-2 text-xs text-subtle">Upload complete — waiting for the server…</p>}
        </div>
      )}

      {(results.length > 0 || rejected.length > 0 || error) && (
        <ul className="mt-3 space-y-1.5">
          {error && (
            <li className="flex items-start gap-2 text-xs text-rose-300">
              <CircleAlert size={14} className="mt-px shrink-0" /> {error}
            </li>
          )}
          {rejected.map((r) => (
            <li key={`rej-${r.name}`} className="flex items-start gap-2 text-xs text-rose-300">
              <CircleAlert size={14} className="mt-px shrink-0" />
              <span>
                <span className="font-medium">{r.name}</span> — {r.reason}
              </span>
            </li>
          ))}
          {results.map((r, i) => (
            <li key={`${r.filename}-${i}`} className={cx('flex items-start gap-2 text-xs', r.error ? 'text-rose-300' : r.duplicate ? 'text-amber-300' : 'text-emerald-300')}>
              {r.error ? <CircleAlert size={14} className="mt-px shrink-0" /> : r.duplicate ? <CopyCheck size={14} className="mt-px shrink-0" /> : <CheckCircle2 size={14} className="mt-px shrink-0" />}
              <span>
                <span className="font-medium">{r.filename}</span> —{' '}
                {r.error ? r.error : r.duplicate ? 'already in the corpus (duplicate checksum)' : r.job_id ? 'uploaded, processing started' : 'uploaded'}
              </span>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}
