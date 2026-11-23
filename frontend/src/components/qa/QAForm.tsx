import type { FormEvent } from 'react'
import { BookOpenText, Globe, Lightbulb, Send } from 'lucide-react'
import type { GeneratorBackend, QAMode, SettingsResponse, Strategy } from '../../types/api'
import { BACKEND_LABELS, EXAMPLE_QUESTIONS, STRATEGIES, STRATEGY_LABELS } from '../../lib/constants'
import { cx } from '../../lib/format'
import { Spinner } from '../Feedback'

export interface QAFormState {
  question: string
  mode: QAMode
  strategy: Strategy
  topK: number
  maxHops: number
  backend: GeneratorBackend
}

interface Props {
  state: QAFormState
  onChange: (patch: Partial<QAFormState>) => void
  onSubmit: () => void
  busy: boolean
  settings: SettingsResponse | null
}

export function QAForm({ state, onChange, onSubmit, busy, settings }: Props) {
  const backends = settings?.options.generator_backends ?? (['auto', 'extractive', 'hf_reader', 'hf_generator', 'ollama', 'anthropic'] as GeneratorBackend[])
  const status = settings?.generator_status ?? {}
  const gnnAvailable = settings?.gnn_status.available

  const submit = (e: FormEvent) => {
    e.preventDefault()
    if (!busy && state.question.trim()) onSubmit()
  }

  return (
    <form onSubmit={submit} className="space-y-4">
      <div className="inline-flex rounded-lg border border-line bg-surface-2 p-0.5">
        {(
          [
            ['multi_document', 'Multi-document', <BookOpenText key="i" size={14} />],
            ['open_domain', 'Open-domain (Wikipedia)', <Globe key="g" size={14} />],
          ] as const
        ).map(([key, label, icon]) => (
          <button
            key={key}
            type="button"
            onClick={() => onChange({ mode: key })}
            className={cx(
              'inline-flex items-center gap-1.5 rounded-md px-3 py-1.5 text-xs font-medium transition-colors',
              state.mode === key ? 'bg-surface-3 text-fg' : 'text-muted hover:text-fg',
            )}
          >
            {icon}
            {label}
          </button>
        ))}
      </div>

      <div>
        <label htmlFor="question" className="sr-only">
          Question
        </label>
        <textarea
          id="question"
          rows={3}
          value={state.question}
          onChange={(e) => onChange({ question: e.target.value })}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) submit(e)
          }}
          placeholder={
            state.mode === 'multi_document'
              ? 'Ask a multi-hop question about your uploaded documents…'
              : 'Ask an open-domain multi-hop question answered from Wikipedia…'
          }
          className="input resize-y text-base leading-relaxed"
        />
      </div>

      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-[1.25fr_1.25fr_1fr_1fr]">
        <div>
          <label className="label" htmlFor="strategy">
            Retrieval strategy
          </label>
          <select id="strategy" className="input" value={state.strategy} onChange={(e) => onChange({ strategy: e.target.value as Strategy })}>
            {STRATEGIES.map((s) => (
              <option key={s} value={s}>
                {STRATEGY_LABELS[s]}
              </option>
            ))}
          </select>
          {state.strategy === 'gnn' && gnnAvailable === false && (
            <p className="mt-1 text-[11px] text-amber-300/90">GNN untrained — the backend falls back to graph heuristic.</p>
          )}
        </div>
        <div>
          <label className="label" htmlFor="backend">
            Generator backend
          </label>
          <select id="backend" className="input" value={state.backend} onChange={(e) => onChange({ backend: e.target.value as GeneratorBackend })}>
            {backends.map((b) => {
              const unavailable = b !== 'auto' && status[b] !== undefined && !status[b].available
              return (
                <option key={b} value={b} disabled={unavailable}>
                  {BACKEND_LABELS[b] ?? b}
                  {unavailable ? ' — unavailable' : ''}
                </option>
              )
            })}
          </select>
        </div>
        <div>
          <label className="label" htmlFor="topk">
            Top-K passages <span className="text-fg tabular-nums">{state.topK}</span>
          </label>
          <input id="topk" type="range" min={1} max={20} value={state.topK} onChange={(e) => onChange({ topK: Number(e.target.value) })} className="mt-2.5 w-full accent-indigo-500" />
        </div>
        <div>
          <label className="label" htmlFor="hops">
            Max hops <span className="text-fg tabular-nums">{state.maxHops}</span>
          </label>
          <input id="hops" type="range" min={1} max={4} value={state.maxHops} onChange={(e) => onChange({ maxHops: Number(e.target.value) })} className="mt-2.5 w-full accent-indigo-500" />
        </div>
      </div>

      <div className="flex flex-wrap items-center justify-between gap-3">
        <span className="text-xs text-subtle">Ctrl / ⌘ + Enter to ask</span>
        <button type="submit" className="btn-primary px-5" disabled={busy || !state.question.trim()}>
          {busy ? <Spinner className="text-white" size={15} /> : <Send size={15} />}
          {busy ? 'Answering…' : 'Ask'}
        </button>
      </div>

      <div>
        <p className="label flex items-center gap-1.5">
          <Lightbulb size={12} /> Example questions
        </p>
        <div className="flex flex-wrap gap-2">
          {EXAMPLE_QUESTIONS.map((q) => (
            <button
              key={q}
              type="button"
              onClick={() => onChange({ question: q, mode: 'multi_document' })}
              className="max-w-full truncate rounded-full border border-line bg-surface-2/60 px-3 py-1 text-left text-xs text-muted transition-colors hover:border-indigo-500/40 hover:text-fg"
              title={q}
            >
              {q}
            </button>
          ))}
        </div>
      </div>
    </form>
  )
}
