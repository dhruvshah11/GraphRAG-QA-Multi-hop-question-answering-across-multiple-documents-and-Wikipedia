import { useState } from 'react'
import { BrainCircuit, Play } from 'lucide-react'
import { CartesianGrid, Legend, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import type { GnnStatus } from '../../types/api'
import { errorMessage, trainGnn } from '../../services/api'
import { useJob } from '../../hooks/useJobs'
import { CHART } from '../../lib/constants'
import { fmtDate, fmtNumber, fmtScore, fmtValue, humanize } from '../../lib/format'
import { StatusBadge } from '../Badge'
import { Card } from '../Card'
import { EmptyState, ErrorBanner, ProgressBar, WarningPanel } from '../Feedback'
import { ConfirmDialog } from '../Overlay'
import { KeyValue } from '../PageHeader'

function str(v: unknown): string | null {
  return typeof v === 'string' ? v : typeof v === 'number' ? String(v) : null
}

export function GnnPanel({ gnn, onChanged }: { gnn: GnnStatus; onChanged: () => void }) {
  const [confirm, setConfirm] = useState(false)
  const [starting, setStarting] = useState(false)
  const [err, setErr] = useState<string | null>(null)
  const latest = gnn.latest_job
  const [jobId, setJobId] = useState<string | null>(latest && (latest.status === 'running' || latest.status === 'queued') ? latest.id : null)
  const { job } = useJob(jobId, () => onChanged(), 3000)

  // When the checkpoint cannot be loaded, fall back to the metadata recorded in training_history.json
  const meta: Record<string, unknown> = gnn.meta && Object.keys(gnn.meta).length ? gnn.meta : (gnn.history?.meta ?? {})
  const hasMeta = Object.keys(meta).length > 0
  const history = gnn.history?.history ?? []
  const summary = gnn.history?.summary
  const baseline = summary?.baseline_val_hybrid ?? (meta.baseline_val_hybrid as Record<string, number> | undefined)
  const running = job ? job.status === 'running' || job.status === 'queued' : false

  const start = async () => {
    setStarting(true)
    setErr(null)
    try {
      const r = await trainGnn('hotpotqa')
      setJobId(r.job_id)
      setConfirm(false)
    } catch (e) {
      setErr(errorMessage(e))
      setConfirm(false)
    } finally {
      setStarting(false)
    }
  }

  return (
    <div className="grid grid-cols-1 gap-6 xl:grid-cols-5">
      <Card
        className="xl:col-span-2"
        title="GNN reranker (GAT)"
        icon={<BrainCircuit size={16} />}
        actions={<StatusBadge status={gnn.available ? 'ready' : gnn.checkpoint_exists ? 'degraded' : 'untrained'} />}
      >
        <div className="space-y-4">
          {!gnn.available && (gnn.checkpoint_exists || hasMeta) && (
            <WarningPanel title={gnn.checkpoint_exists ? 'Checkpoint present but not usable' : 'No usable checkpoint'}>
              {gnn.reason ?? 'The checkpoint could not be loaded.'} The “GNN” strategy currently falls back to the graph heuristic ranking.
              {hasMeta && <span className="mt-1 block text-xs text-amber-100/60">Details below are from the last recorded training run.</span>}
            </WarningPanel>
          )}
          {gnn.available || hasMeta ? (
            <KeyValue
              items={[
                ['Trained on', str(meta.dataset_label) ?? str(meta.training_source) ?? '—'],
                ['Source', str(meta.training_source) ?? '—'],
                ['Trained at', fmtDate(str(meta.trained_at))],
                ['Train graphs', fmtNumber((meta.num_train_graphs ?? summary?.num_train_graphs) as number | undefined)],
                ['Val graphs', fmtNumber(meta.num_val_graphs as number | undefined)],
                ['Best epoch', fmtNumber((summary?.best_epoch ?? meta.best_epoch) as number | undefined)],
                ['Epochs run', fmtNumber(summary?.epochs_run)],
                ['Parameters', fmtNumber(gnn.num_parameters ?? summary?.num_parameters)],
                ['Embeddings', str(meta.embedding_model) ?? '—'],
                ['Architecture', gnn.architecture ?? '—'],
              ]}
            />
          ) : gnn.checkpoint_exists ? (
            <p className="text-sm text-muted">No training metadata is available.</p>
          ) : (
            <EmptyState
              className="py-6"
              title="GNN not trained yet"
              description={
                <>
                  {gnn.reason ?? 'No trained checkpoint is available.'} Until a checkpoint exists, the “GNN” strategy falls back to the graph heuristic ranking.
                </>
              }
              action={
                gnn.hotpotqa_available ? (
                  <button type="button" className="btn-primary" onClick={() => setConfirm(true)} disabled={running || starting}>
                    <Play size={14} /> Train on HotpotQA
                  </button>
                ) : (
                  <p className="text-xs text-subtle">HotpotQA training data not found — run scripts/download_datasets.py first.</p>
                )
              }
            />
          )}
          {(gnn.available || hasMeta || gnn.checkpoint_exists) && gnn.hotpotqa_available && !running && (
            <button type="button" className="btn-secondary btn-sm" onClick={() => setConfirm(true)}>
              <Play size={13} /> {gnn.available ? 'Retrain' : 'Train'} on HotpotQA
            </button>
          )}
          {(job || latest) && (
            <div className="rounded-lg border border-line bg-surface-2/50 p-3">
              {(() => {
                const j = job ?? latest
                if (!j) return null
                return (
                  <>
                    <div className="mb-1.5 flex items-center justify-between text-xs">
                      <span className="text-muted">Training job <span className="mono">{j.id}</span></span>
                      <StatusBadge status={j.status} />
                    </div>
                    <ProgressBar value={j.progress} />
                    <p className="mt-1.5 text-[11px] text-subtle">{j.message ?? ''}</p>
                  </>
                )
              })()}
            </div>
          )}
          {err && <ErrorBanner message={err} onDismiss={() => setErr(null)} />}
          {gnn.checkpoint_path && (
            <p className="text-[11px] break-all text-subtle">
              checkpoint: <span className="mono">{gnn.checkpoint_path}</span>
              {gnn.feature_spec_version && <> · features {gnn.feature_spec_version}</>}
            </p>
          )}
        </div>
      </Card>

      <Card className="xl:col-span-3" title="Training curves" subtitle="Per-epoch loss and validation retrieval metrics (training_history.json)">
        {history.length ? (
          <div className="space-y-6">
            <div>
              <p className="label">Loss</p>
              <ResponsiveContainer width="100%" height={200}>
                <LineChart data={history} margin={{ top: 4, right: 12, left: -12, bottom: 0 }}>
                  <CartesianGrid stroke={CHART.grid} vertical={false} />
                  <XAxis dataKey="epoch" tick={CHART.tick} axisLine={{ stroke: CHART.grid }} tickLine={false} />
                  <YAxis tick={CHART.tick} axisLine={false} tickLine={false} />
                  <Tooltip {...CHART.tooltip} formatter={(v) => (typeof v === 'number' ? v.toFixed(4) : 'n/a')} labelFormatter={(l) => `Epoch ${l}`} />
                  <Legend itemSorter={null} wrapperStyle={{ fontSize: 12 }} iconType="plainline" />
                  <Line type="monotone" dataKey="train_loss" name="Train loss" stroke="#818cf8" strokeWidth={2} dot={false} />
                  <Line type="monotone" dataKey="val_loss" name="Val loss" stroke="#f472b6" strokeWidth={2} dot={false} />
                  {summary?.best_epoch !== undefined && <ReferenceLine x={summary.best_epoch} stroke="#475569" strokeDasharray="4 4" label={{ value: 'best', fill: '#94a3b8', fontSize: 10 }} />}
                </LineChart>
              </ResponsiveContainer>
            </div>
            <div>
              <p className="label">Validation metrics {baseline && <span className="text-subtle">(dashed = hybrid baseline)</span>}</p>
              <ResponsiveContainer width="100%" height={200}>
                <LineChart data={history} margin={{ top: 4, right: 12, left: -12, bottom: 0 }}>
                  <CartesianGrid stroke={CHART.grid} vertical={false} />
                  <XAxis dataKey="epoch" tick={CHART.tick} axisLine={{ stroke: CHART.grid }} tickLine={false} />
                  <YAxis domain={[0, 1]} tick={CHART.tick} axisLine={false} tickLine={false} />
                  <Tooltip {...CHART.tooltip} formatter={(v) => (typeof v === 'number' ? v.toFixed(4) : 'n/a')} labelFormatter={(l) => `Epoch ${l}`} />
                  <Legend itemSorter={null} wrapperStyle={{ fontSize: 12 }} iconType="plainline" />
                  <Line type="monotone" dataKey="val_mrr" name="Val MRR" stroke="#2dd4bf" strokeWidth={2} dot={false} />
                  <Line type="monotone" dataKey="val_recall_at_k" name="Val R@K" stroke="#f59e0b" strokeWidth={2} dot={false} />
                  <Line type="monotone" dataKey="val_precision_at_k" name="Val P@K" stroke="#60a5fa" strokeWidth={2} dot={false} />
                  {typeof baseline?.mrr === 'number' && <ReferenceLine y={baseline.mrr} stroke="#2dd4bf" strokeOpacity={0.6} strokeDasharray="5 4" />}
                  {typeof baseline?.recall_at_k === 'number' && <ReferenceLine y={baseline.recall_at_k} stroke="#f59e0b" strokeOpacity={0.6} strokeDasharray="5 4" />}
                </LineChart>
              </ResponsiveContainer>
            </div>
            {(summary?.best_val || baseline) && (
              <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
                {summary?.best_val && <MetricList title={`Best epoch ${summary.best_epoch ?? ''}`} values={summary.best_val as Record<string, unknown>} />}
                {baseline && <MetricList title="Hybrid baseline (validation)" values={baseline} />}
              </div>
            )}
          </div>
        ) : (
          <EmptyState title="No training history" description="Training curves appear here after the GAT reranker has been trained." />
        )}
      </Card>

      <ConfirmDialog
        open={confirm}
        title="Train the GNN reranker?"
        confirmLabel={starting ? 'Starting…' : 'Start training'}
        busy={starting}
        message={
          <>
            This starts a background job that trains the GAT reranker on the HotpotQA sample. It is CPU-intensive and <strong className="text-fg">takes several minutes</strong>;
            question answering and evaluation will be slower while it runs. Progress is shown here.
          </>
        }
        onCancel={() => setConfirm(false)}
        onConfirm={() => void start()}
      />
    </div>
  )
}

function MetricList({ title, values }: { title: string; values: Record<string, unknown> }) {
  return (
    <div className="rounded-lg border border-line p-3">
      <p className="mb-1.5 text-xs font-semibold text-fg">{title}</p>
      <dl className="space-y-0.5 text-xs">
        {Object.entries(values)
          .filter(([k]) => k !== 'epoch')
          .map(([k, v]) => (
            <div key={k} className="flex justify-between gap-2">
              <dt className="text-muted">{humanize(k)}</dt>
              <dd className="text-fg tabular-nums">{typeof v === 'number' ? (Number.isInteger(v) ? fmtNumber(v) : fmtScore(v, 4)) : fmtValue(v)}</dd>
            </div>
          ))}
      </dl>
    </div>
  )
}
