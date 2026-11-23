import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useSearchParams } from 'react-router-dom'
import { BookMarked, Gauge, GitCommitVertical, Layers, ListTree, Sparkles, Timer, History } from 'lucide-react'
import { useApi } from '../hooks/useApi'
import { ApiError, askMultiDocument, askOpenDomain, errorMessage, getQAHistory, getQuery, getSettings } from '../services/api'
import type { Citation, QAResult } from '../types/api'
import { PageHeader } from '../components/PageHeader'
import { Card } from '../components/Card'
import { ErrorBanner, EmptyState, SkeletonBlock, WarningPanel } from '../components/Feedback'
import { CitationCard, CitationModal } from '../components/CitationCard'
import { PassageCard } from '../components/PassageCard'
import { QAForm, type QAFormState } from '../components/qa/QAForm'
import { PipelineRunning, StageTimings } from '../components/qa/PipelineProgress'
import { Chips, ConfidencePanel, EvidenceChain, NotesList, ReasoningSteps, RunMeta } from '../components/qa/ResultSections'
import { fmtMs, fmtPct, fmtRelative, truthy } from '../lib/format'
import { STRATEGY_SHORT } from '../lib/constants'

interface QAErr {
  status: number
  message: string
}

function errorTitle(status: number): string {
  if (status === 409) return 'No documents to search'
  if (status === 503) return 'Service unavailable'
  if (status === 0) return 'Backend unreachable'
  if (status === 404) return 'Not found'
  if (status === 422 || status === 400) return 'Invalid request'
  return 'Question answering failed'
}

export default function QAPage() {
  const [params, setParams] = useSearchParams()
  const queryParam = params.get('query')
  const settings = useApi((s) => getSettings(s), [])
  const history = useApi((s) => getQAHistory(s), [])

  const [form, setForm] = useState<QAFormState>({
    question: '',
    mode: 'multi_document',
    strategy: 'gnn',
    topK: 8,
    maxHops: 3,
    backend: 'auto',
  })
  const initialised = useRef(false)
  useEffect(() => {
    if (settings.data && !initialised.current) {
      initialised.current = true
      const rt = settings.data.runtime
      setForm((f) => ({
        ...f,
        topK: Math.min(20, Math.max(1, rt.top_k)),
        maxHops: Math.min(4, Math.max(1, rt.max_hops)),
        backend: rt.generator_backend,
      }))
    }
  }, [settings.data])

  const [result, setResult] = useState<QAResult | null>(null)
  const [busy, setBusy] = useState(false)
  const [loadingPrev, setLoadingPrev] = useState(false)
  const [err, setErr] = useState<QAErr | null>(null)
  const [openCitation, setOpenCitation] = useState<Citation | null>(null)
  const abortRef = useRef<AbortController | null>(null)

  // Load a previous answer from ?query=<id>
  useEffect(() => {
    if (!queryParam || result?.query_id === queryParam) return
    const controller = new AbortController()
    setLoadingPrev(true)
    setErr(null)
    getQuery(queryParam, controller.signal)
      .then((r) => {
        setResult(r)
        setForm((f) => ({
          ...f,
          question: r.question,
          mode: r.mode === 'open_domain' ? 'open_domain' : 'multi_document',
          strategy: (r.strategy as QAFormState['strategy']) ?? f.strategy,
          topK: r.top_k ? Math.min(20, r.top_k) : f.topK,
          maxHops: r.max_hops ?? f.maxHops,
        }))
      })
      .catch((e) => {
        if (controller.signal.aborted) return
        setErr({ status: e instanceof ApiError ? e.status : 0, message: errorMessage(e) })
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoadingPrev(false)
      })
    return () => controller.abort()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [queryParam])

  const ask = useCallback(async () => {
    abortRef.current?.abort()
    const controller = new AbortController()
    abortRef.current = controller
    setBusy(true)
    setErr(null)
    setResult(null)
    const req = {
      question: form.question.trim(),
      strategy: form.strategy,
      top_k: form.topK,
      max_hops: form.maxHops,
      backend: form.backend,
    }
    try {
      const r = form.mode === 'open_domain' ? await askOpenDomain(req, controller.signal) : await askMultiDocument(req, controller.signal)
      setResult(r)
      setParams({ query: r.query_id }, { replace: false })
      void history.refetch(true)
    } catch (e) {
      if (controller.signal.aborted) return
      setErr({ status: e instanceof ApiError ? e.status : 0, message: errorMessage(e) })
    } finally {
      if (!controller.signal.aborted) setBusy(false)
    }
  }, [form, setParams, history])

  useEffect(() => () => abortRef.current?.abort(), [])

  const openByIndex = useCallback(
    (index: number) => {
      const c = result?.citations.find((x) => x.index === index)
      if (c) setOpenCitation(c)
    },
    [result],
  )

  return (
    <>
      <PageHeader
        title="Question Answering"
        description="Multi-hop answers grounded in retrieved evidence, with an explicit evidence chain, validated citations and confidence."
      />
      <div className="grid grid-cols-1 gap-6 xl:grid-cols-[minmax(0,1fr)_300px]">
        <div className="min-w-0 space-y-6">
          <Card>
            <QAForm state={form} onChange={(p) => setForm((f) => ({ ...f, ...p }))} onSubmit={() => void ask()} busy={busy} settings={settings.data} />
          </Card>

          {busy && (
            <Card title="Running pipeline" subtitle={form.mode === 'open_domain' ? 'Fetching Wikipedia articles and building the evidence graph…' : 'Retrieving, building the evidence graph and generating a grounded answer…'} icon={<Timer size={16} />}>
              <PipelineRunning openDomain={form.mode === 'open_domain'} />
            </Card>
          )}

          {err && <ErrorBanner title={errorTitle(err.status)} message={err.message} onDismiss={() => setErr(null)} />}

          {loadingPrev && !result && (
            <Card>
              <SkeletonBlock rows={6} />
            </Card>
          )}

          {result && !busy && <ResultView result={result} onOpen={openByIndex} />}

          {!result && !busy && !loadingPrev && !err && (
            <Card>
              <EmptyState
                icon={<Sparkles size={18} />}
                title="Ask a question to get started"
                description="Pick an example or write your own multi-hop question. Results include the reasoning trace, evidence chain, citations and per-passage retrieval scores."
              />
            </Card>
          )}
        </div>

        <aside className="space-y-6">
          {result && !busy && (
            <div className="hidden space-y-6 xl:block">
              <InsightCards result={result} />
            </div>
          )}
          <Card title="Recent questions" icon={<History size={16} />} flush>
            {history.error ? (
              <ErrorBanner message={history.error} className="m-3" onRetry={() => void history.refetch()} />
            ) : history.loading && !history.data ? (
              <div className="p-4">
                <SkeletonBlock rows={5} />
              </div>
            ) : history.data && history.data.queries.length ? (
              <ul className="max-h-[50vh] divide-y divide-line overflow-y-auto">
                {history.data.queries.map((q) => (
                  <li key={q.id}>
                    <button
                      type="button"
                      onClick={() => setParams({ query: q.id })}
                      className={`w-full px-4 py-2.5 text-left transition-colors hover:bg-surface-2/60 ${result?.query_id === q.id ? 'bg-accent-soft' : ''}`}
                    >
                      <p className="line-clamp-2 text-xs font-medium text-fg">{q.question}</p>
                      <p className="mt-1 truncate text-[11px] text-subtle">
                        {truthy(q.insufficient_evidence) ? <span className="text-amber-300">insufficient</span> : <span className="text-indigo-200">{q.answer}</span>} ·{' '}
                        {STRATEGY_SHORT[q.strategy] ?? q.strategy} · {fmtMs(q.latency_ms)} · {fmtRelative(q.created_at)}
                      </p>
                    </button>
                  </li>
                ))}
              </ul>
            ) : (
              <EmptyState title="No history yet" />
            )}
          </Card>
        </aside>
      </div>
      <CitationModal citation={openCitation} onClose={() => setOpenCitation(null)} />
    </>
  )
}

function InsightCards({ result }: { result: QAResult }) {
  return (
    <>
      <Card title="Confidence & coverage" icon={<Gauge size={16} />}>
        <ConfidencePanel result={result} />
      </Card>
      <Card title="Stage timings" subtitle="Measured on the server (timings_ms)" icon={<Timer size={16} />}>
        {Object.keys(result.timings_ms ?? {}).length ? <StageTimings timings={result.timings_ms} total={result.total_ms} /> : <EmptyState title="No timings recorded" />}
      </Card>
    </>
  )
}

function ResultView({ result, onOpen }: { result: QAResult; onOpen: (i: number) => void }) {
  const insufficient = truthy(result.insufficient_evidence)
  const chainIds = useMemo(() => new Set(result.evidence_chain.map((s) => s.chunk_id)), [result])
  const citedIds = useMemo(() => new Set(result.citations.map((c) => c.chunk_id)), [result])

  return (
    <div className="space-y-6">
      {/* 1. Final answer */}
      <Card>
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div className="min-w-0 flex-1">
            <p className="label">Final answer</p>
            <p className={`text-2xl font-semibold tracking-tight sm:text-3xl ${insufficient ? 'text-amber-200' : 'text-fg'}`}>
              {result.answer || (insufficient ? 'Insufficient evidence' : '—')}
            </p>
            {result.answer_sentence && (
              <p className="mt-2 text-sm leading-relaxed text-muted">
                {result.answer_sentence}
                <Chips ids={result.answer_citations} onOpen={onOpen} />
              </p>
            )}
            <p className="mt-2 text-xs text-subtle">Question: {result.question}</p>
          </div>
          <div className="text-right">
            <p className="label">Confidence</p>
            <p className="text-xl font-semibold text-fg tabular-nums">{fmtPct(result.confidence, 0)}</p>
            <p className="mt-1 text-xs text-subtle">{fmtMs(result.total_ms)} total</p>
          </div>
        </div>
        <div className="mt-4">
          <RunMeta result={result} />
        </div>
        {insufficient && (
          <WarningPanel title="Insufficient evidence — the system abstained or the answer is weakly supported" className="mt-4">
            {result.insufficient_reasons.length ? (
              <ul className="list-disc space-y-0.5 pl-4">
                {result.insufficient_reasons.map((r, i) => (
                  <li key={i}>{r}</li>
                ))}
              </ul>
            ) : (
              'The retrieved passages do not support a confident answer.'
            )}
          </WarningPanel>
        )}
        {(result.notes.length > 0 || result.generator_warnings.length > 0) && (
          <div className="mt-4 border-t border-line pt-3">
            <NotesList notes={result.notes} warnings={result.generator_warnings} />
          </div>
        )}
      </Card>

      {/* 6. Confidence & timings: shown inline below xl, in the page side column at xl+ */}
      <div className="grid grid-cols-1 gap-6 md:grid-cols-2 xl:hidden">
        <InsightCards result={result} />
      </div>

      <div className="space-y-6">
        <div className="min-w-0 space-y-6">
          {/* 2. Reasoning */}
          <Card title="Reasoning summary" subtitle="Retrieved facts (quoted, cited) vs. model inference" icon={<ListTree size={16} />}>
            <ReasoningSteps steps={result.reasoning_steps} onOpen={onOpen} />
          </Card>

          {/* 3. Evidence chain */}
          <Card title="Evidence chain" subtitle="Hop-by-hop path through passages linked by bridge entities" icon={<GitCommitVertical size={16} />}>
            <EvidenceChain chain={result.evidence_chain} queryId={result.query_id} onOpen={onOpen} />
          </Card>

          {/* 4. Citations */}
          <Card title="Supporting citations" subtitle="Click a citation to view the full passage with the supporting sentence highlighted" icon={<BookMarked size={16} />}>
            {result.citations.length ? (
              <div className="grid grid-cols-1 gap-3 lg:grid-cols-2">
                {result.citations.map((c) => (
                  <CitationCard key={c.index} citation={c} onOpen={onOpen} />
                ))}
              </div>
            ) : (
              <EmptyState title="No validated citations" description="No passage sentence passed citation validation for this answer." />
            )}
          </Card>
        </div>

      </div>

      {/* 5. Retrieved passages */}
      <Card title="Retrieved passages" subtitle={`${result.retrieved_passages.length} passages ranked by final score`} icon={<Layers size={16} />}>
        {result.retrieved_passages.length ? (
          <div className="space-y-3">
            {result.retrieved_passages.map((p) => (
              <PassageCard key={p.chunk_id} passage={p} inChain={chainIds.has(p.chunk_id)} cited={citedIds.has(p.chunk_id)} />
            ))}
          </div>
        ) : (
          <EmptyState title="No passages retrieved" />
        )}
      </Card>
    </div>
  )
}
