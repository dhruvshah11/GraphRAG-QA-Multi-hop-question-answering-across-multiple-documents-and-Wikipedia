import { useState, type FormEvent } from 'react'
import { Play } from 'lucide-react'
import type { EvaluationDataset, EvaluationRequest, GeneratorBackend, SettingsResponse } from '../../types/api'
import { BACKEND_LABELS, STRATEGIES, STRATEGY_COLORS, STRATEGY_LABELS } from '../../lib/constants'
import { cx } from '../../lib/format'
import { Spinner, WarningPanel } from '../Feedback'

interface Props {
  datasets: EvaluationDataset[]
  dataset: string
  onDatasetChange: (id: string) => void
  settings: SettingsResponse | null
  busy: boolean
  onSubmit: (req: EvaluationRequest) => void
}

export function RunForm({ datasets, dataset, onDatasetChange, settings, busy, onSubmit }: Props) {
  const [strategies, setStrategies] = useState<string[]>(['faiss', 'bm25', 'hybrid', 'graph', 'gnn'])
  const [k, setK] = useState(2)
  const [limit, setLimit] = useState<string>('')
  const [topK, setTopK] = useState<string>('')
  const [maxHops, setMaxHops] = useState<string>('')
  const [backend, setBackend] = useState<GeneratorBackend | ''>('')

  const ds = datasets.find((d) => d.id === dataset)
  const gnnAvailable = settings?.gnn_status.available
  const n = Math.min(ds?.questions ?? 0, limit ? Number(limit) : Infinity)
  const big = n * strategies.length > 200

  const submit = (e: FormEvent) => {
    e.preventDefault()
    const req: EvaluationRequest = { dataset, strategies, k }
    if (limit) req.limit = Number(limit)
    if (topK) req.top_k = Number(topK)
    if (maxHops) req.max_hops = Number(maxHops)
    if (backend) req.backend = backend
    onSubmit(req)
  }

  return (
    <form onSubmit={submit} className="space-y-4">
      <div>
        <label className="label" htmlFor="ds">
          Dataset
        </label>
        <select id="ds" className="input" value={dataset} onChange={(e) => onDatasetChange(e.target.value)}>
          {datasets.map((d) => (
            <option key={d.id} value={d.id} disabled={!d.available}>
              {d.name} ({d.questions} questions){d.available ? '' : ' — unavailable'}
            </option>
          ))}
        </select>
      </div>
      <div>
        <span className="label">Strategies</span>
        <div className="flex flex-wrap gap-2">
          {STRATEGIES.map((s) => {
            const on = strategies.includes(s)
            return (
              <button
                key={s}
                type="button"
                onClick={() => setStrategies((prev) => (on ? prev.filter((x) => x !== s) : STRATEGIES.filter((x) => x === s || prev.includes(x))))}
                className={cx(
                  'inline-flex items-center gap-1.5 rounded-lg border px-2.5 py-1.5 text-xs font-medium transition-colors',
                  on ? 'border-indigo-500/50 bg-indigo-500/10 text-fg' : 'border-line text-muted hover:text-fg',
                )}
                aria-pressed={on}
              >
                <span className="h-2 w-2 rounded-full" style={{ background: on ? STRATEGY_COLORS[s] : '#334155' }} />
                {STRATEGY_LABELS[s]}
              </button>
            )
          })}
        </div>
        {strategies.includes('gnn') && gnnAvailable === false && (
          <p className="mt-1.5 text-[11px] text-amber-300/90">No trained GNN checkpoint — the backend will skip the GNN strategy.</p>
        )}
      </div>
      <div className="grid grid-cols-2 gap-3">
        <div>
          <label className="label" htmlFor="k">
            K (for P@K / R@K)
          </label>
          <input id="k" type="number" min={1} max={20} className="input" value={k} onChange={(e) => setK(Math.max(1, Number(e.target.value) || 1))} />
        </div>
        <div>
          <label className="label" htmlFor="limit">
            Question limit
          </label>
          <input id="limit" type="number" min={1} className="input" placeholder={`all (${ds?.questions ?? '—'})`} value={limit} onChange={(e) => setLimit(e.target.value)} />
        </div>
        <div>
          <label className="label" htmlFor="etopk">
            Top-K retrieved
          </label>
          <input id="etopk" type="number" min={1} max={50} className="input" placeholder={`default (${settings?.runtime.top_k ?? '—'})`} value={topK} onChange={(e) => setTopK(e.target.value)} />
        </div>
        <div>
          <label className="label" htmlFor="ehops">
            Max hops
          </label>
          <input id="ehops" type="number" min={1} max={4} className="input" placeholder={`default (${settings?.runtime.max_hops ?? '—'})`} value={maxHops} onChange={(e) => setMaxHops(e.target.value)} />
        </div>
      </div>
      <div>
        <label className="label" htmlFor="ebackend">
          Generator backend
        </label>
        <select id="ebackend" className="input" value={backend} onChange={(e) => setBackend(e.target.value as GeneratorBackend | '')}>
          <option value="">Default ({settings?.runtime.generator_backend ?? 'auto'})</option>
          {(settings?.options.generator_backends ?? []).map((b) => {
            const unavailable = b !== 'auto' && settings?.generator_status[b] !== undefined && !settings.generator_status[b].available
            return (
              <option key={b} value={b} disabled={unavailable}>
                {BACKEND_LABELS[b] ?? b}
                {unavailable ? ' — unavailable' : ''}
              </option>
            )
          })}
        </select>
      </div>
      {big && (
        <WarningPanel title="Large run">
          {n} questions × {strategies.length} strategies ≈ {n * strategies.length} pipeline runs at a few seconds each. Consider setting a question limit.
        </WarningPanel>
      )}
      {dataset === 'wikipedia' && (
        <p className="text-[11px] text-subtle">The Wikipedia dataset needs live Wikipedia access; runs fail if it is not configured on the server.</p>
      )}
      <button type="submit" className="btn-primary w-full" disabled={busy || !strategies.length || !ds?.available}>
        {busy ? <Spinner size={14} className="text-white" /> : <Play size={14} />} Start evaluation
      </button>
    </form>
  )
}
