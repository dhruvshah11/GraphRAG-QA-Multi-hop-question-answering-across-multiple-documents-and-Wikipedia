import { useState } from 'react'
import { Eye, FileText, Layers, RefreshCw, RotateCw, Sigma, Tags, Trash2, Boxes, FileStack } from 'lucide-react'
import { useApi } from '../hooks/useApi'
import { usePolling } from '../hooks/usePolling'
import { useJobTracker } from '../hooks/useJobs'
import { deleteDocument, errorMessage, getSettings, listDocuments, processDocument } from '../services/api'
import type { DocumentRecord, UploadResult } from '../types/api'
import { PageHeader } from '../components/PageHeader'
import { Card } from '../components/Card'
import { StatCard } from '../components/StatCard'
import { EmptyState, ErrorBanner, ProgressBar, SkeletonBlock, SkeletonCards } from '../components/Feedback'
import { Badge, StatusBadge } from '../components/Badge'
import { ConfirmDialog } from '../components/Overlay'
import { UploadDropzone } from '../components/documents/UploadDropzone'
import { DocumentDetailDrawer } from '../components/documents/DocumentDetailDrawer'
import { fmtBytes, fmtDate, fmtMs, fmtNumber } from '../lib/format'

const ACTIVE_STATUSES = new Set(['uploaded', 'queued', 'processing'])

export default function DocumentsPage() {
  const docs = useApi((s) => listDocuments(s), [])
  const settings = useApi((s) => getSettings(s), [])
  const [detailId, setDetailId] = useState<string | null>(null)
  const [toDelete, setToDelete] = useState<DocumentRecord | null>(null)
  const [deleting, setDeleting] = useState(false)
  const [actionError, setActionError] = useState<string | null>(null)

  const tracker = useJobTracker(() => void docs.refetch(true))

  const anyActive = !!docs.data?.documents.some((d) => ACTIVE_STATUSES.has(d.status))
  usePolling(() => void docs.refetch(true), 2500, anyActive || tracker.active)

  const onUploaded = (results: UploadResult[]) => {
    for (const r of results) if (r.job_id) tracker.track(r.job_id, r.filename)
    void docs.refetch(true)
  }

  const reprocess = async (d: DocumentRecord) => {
    setActionError(null)
    try {
      const { job_id } = await processDocument(d.id)
      tracker.track(job_id, d.name)
      void docs.refetch(true)
    } catch (err) {
      setActionError(errorMessage(err))
    }
  }

  const confirmDelete = async () => {
    if (!toDelete) return
    setDeleting(true)
    setActionError(null)
    try {
      await deleteDocument(toDelete.id)
      if (detailId === toDelete.id) setDetailId(null)
      setToDelete(null)
      void docs.refetch(true)
    } catch (err) {
      setActionError(errorMessage(err))
      setToDelete(null)
    } finally {
      setDeleting(false)
    }
  }

  const stats = docs.data?.stats
  const limits = settings.data?.limits

  return (
    <>
      <PageHeader
        title="Documents"
        description="Upload PDF, TXT and DOCX files. Each document is extracted, chunked, entity-tagged, embedded into FAISS / BM25 and linked into the evidence graph."
        actions={
          <button type="button" className="btn-secondary" onClick={() => void docs.refetch()} disabled={docs.loading}>
            <RefreshCw size={14} className={docs.loading ? 'animate-spin' : ''} /> Refresh
          </button>
        }
      />

      {stats ? (
        <div className="mb-6 grid grid-cols-2 gap-3 md:grid-cols-4 xl:grid-cols-7">
          <StatCard label="Documents" value={fmtNumber(stats.total_documents)} hint={`${fmtNumber(stats.processed_documents)} processed`} icon={<FileText size={15} />} />
          <StatCard label="Pages" value={fmtNumber(stats.total_pages)} icon={<FileStack size={15} />} />
          <StatCard label="Chunks" value={fmtNumber(stats.total_chunks)} icon={<Layers size={15} />} />
          <StatCard label="Avg chunk tokens" value={fmtNumber(stats.avg_chunk_tokens, 1)} icon={<Sigma size={15} />} />
          <StatCard label="Avg chunk chars" value={fmtNumber(stats.avg_chunk_chars, 1)} icon={<Sigma size={15} />} />
          <StatCard label="Entities" value={fmtNumber(stats.total_entities)} icon={<Boxes size={15} />} />
          <StatCard label="Entity mentions" value={fmtNumber(stats.entity_mentions)} icon={<Tags size={15} />} />
        </div>
      ) : docs.loading ? (
        <SkeletonCards count={7} className="mb-6 grid-cols-2 md:grid-cols-4 xl:grid-cols-7" />
      ) : null}

      <div className="space-y-6">
        <div className={tracker.jobs.length > 0 ? 'grid gap-6 lg:grid-cols-2' : ''}>
          <Card title="Upload documents">
            {limits ? (
              <UploadDropzone maxMb={limits.max_upload_mb} allowed={limits.allowed_extensions} onUploaded={onUploaded} />
            ) : settings.error ? (
              <ErrorBanner message={settings.error} onRetry={() => void settings.refetch()} />
            ) : (
              <SkeletonBlock rows={3} />
            )}
          </Card>
          {tracker.jobs.length > 0 && (
            <Card title="Processing jobs" actions={<span className="text-xs text-subtle">polling /api/jobs</span>}>
              <ul className="space-y-3">
                {tracker.jobs.map((j) => (
                  <li key={j.id}>
                    <div className="mb-1 flex items-center justify-between gap-2 text-xs">
                      <span className="truncate font-medium text-fg">{j.label}</span>
                      <span className="flex items-center gap-2">
                        <StatusBadge status={j.error ? 'error' : (j.job?.status ?? 'queued')} />
                        {(j.job?.status === 'completed' || j.job?.status === 'failed' || j.error) && (
                          <button type="button" className="text-subtle hover:text-fg" onClick={() => tracker.dismiss(j.id)}>
                            dismiss
                          </button>
                        )}
                      </span>
                    </div>
                    <ProgressBar value={j.job?.progress ?? 0} indeterminate={!j.job} />
                    <p className="mt-1 truncate text-[11px] text-subtle">{j.error ?? j.job?.message ?? 'waiting for worker…'}</p>
                  </li>
                ))}
              </ul>
            </Card>
          )}
        </div>

        <Card title="Corpus" subtitle={docs.data ? `${docs.data.documents.length} document(s)` : undefined} flush>
          {actionError && <ErrorBanner message={actionError} onDismiss={() => setActionError(null)} className="m-4" />}
          {docs.error && <ErrorBanner message={docs.error} onRetry={() => void docs.refetch()} className="m-4" />}
          {docs.loading && !docs.data ? (
            <div className="p-5">
              <SkeletonBlock rows={6} />
            </div>
          ) : docs.data && docs.data.documents.length === 0 ? (
            <EmptyState title="No documents yet" description="Drop PDF, TXT or DOCX files into the upload area to build the corpus." icon={<FileText size={18} />} />
          ) : docs.data ? (
            <div className="overflow-x-auto">
              <table className="table-base">
                <thead>
                  <tr>
                    <th>Document</th>
                    <th>Status</th>
                    <th className="text-right">Size</th>
                    <th className="text-right">Pages</th>
                    <th className="text-right">Chunks</th>
                    <th className="text-right">Entities</th>
                    <th>Processed</th>
                    <th className="text-right">Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {docs.data.documents.map((d) => (
                    <tr key={d.id}>
                      <td className="max-w-[360px] min-w-[220px]">
                        <button type="button" className="block max-w-full truncate text-left font-medium text-fg hover:text-indigo-300" onClick={() => setDetailId(d.id)}>
                          {d.name}
                        </button>
                        <div className="mt-0.5 flex items-center gap-1.5">
                          <Badge tone="neutral">{d.file_type}</Badge>
                          <span className="text-[11px] text-subtle">{fmtDate(d.created_at)}</span>
                        </div>
                        {d.error && <p className="mt-1 text-xs text-rose-300">{d.error}</p>}
                      </td>
                      <td>
                        <StatusBadge status={d.status} />
                      </td>
                      <td className="text-right whitespace-nowrap text-muted tabular-nums">{fmtBytes(d.size_bytes)}</td>
                      <td className="text-right tabular-nums">{fmtNumber(d.num_pages)}</td>
                      <td className="text-right tabular-nums">{fmtNumber(d.num_chunks)}</td>
                      <td className="text-right tabular-nums">{fmtNumber(d.num_entities)}</td>
                      <td className="text-xs whitespace-nowrap text-muted">
                        {fmtDate(d.processed_at)}
                        {d.processing_ms !== null && <div className="text-subtle">{fmtMs(d.processing_ms)}</div>}
                      </td>
                      <td>
                        <div className="flex justify-end gap-1">
                          <button type="button" className="btn-ghost btn-sm" title="View details" onClick={() => setDetailId(d.id)}>
                            <Eye size={14} />
                          </button>
                          <button
                            type="button"
                            className="btn-ghost btn-sm"
                            title="Reprocess"
                            disabled={ACTIVE_STATUSES.has(d.status)}
                            onClick={() => void reprocess(d)}
                          >
                            <RotateCw size={14} />
                          </button>
                          <button type="button" className="btn-ghost btn-sm hover:text-rose-300" title="Delete" onClick={() => setToDelete(d)}>
                            <Trash2 size={14} />
                          </button>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : null}
        </Card>
      </div>

      <DocumentDetailDrawer documentId={detailId} onClose={() => setDetailId(null)} />
      <ConfirmDialog
        open={toDelete !== null}
        title="Delete document?"
        danger
        busy={deleting}
        confirmLabel={deleting ? 'Deleting…' : 'Delete'}
        message={
          <>
            <span className="font-medium text-fg">{toDelete?.name}</span> will be removed from the corpus, the FAISS and BM25 indexes and the
            evidence graph. This cannot be undone.
          </>
        }
        onCancel={() => setToDelete(null)}
        onConfirm={() => void confirmDelete()}
      />
    </>
  )
}
