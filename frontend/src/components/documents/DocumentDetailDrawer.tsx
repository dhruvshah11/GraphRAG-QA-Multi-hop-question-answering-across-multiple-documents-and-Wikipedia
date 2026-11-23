import { useMemo, useState } from 'react'
import { ArrowRight } from 'lucide-react'
import { useApi } from '../../hooks/useApi'
import { getDocument } from '../../services/api'
import type { DocumentDetail, DocumentEntity } from '../../types/api'
import { NER_LABEL_COLORS } from '../../lib/constants'
import { fmtBytes, fmtDate, fmtMs, fmtNumber } from '../../lib/format'
import { Drawer } from '../Overlay'
import { Tabs } from '../Tabs'
import { EmptyState, ErrorBanner, SkeletonBlock } from '../Feedback'
import { Badge, StatusBadge } from '../Badge'
import { KeyValue } from '../PageHeader'

type TabKey = 'text' | 'chunks' | 'entities' | 'relations' | 'metadata'

export function DocumentDetailDrawer({ documentId, onClose }: { documentId: string | null; onClose: () => void }) {
  const { data, error, loading, refetch } = useApi((s) => getDocument(documentId as string, s), [documentId], documentId !== null)
  const [tab, setTab] = useState<TabKey>('text')
  const doc = data?.document

  return (
    <Drawer
      open={documentId !== null}
      onClose={onClose}
      title={doc?.name ?? 'Document'}
      subtitle={doc ? <span className="mono">{doc.id}</span> : undefined}
    >
      {error && <ErrorBanner message={error} onRetry={() => void refetch()} className="m-5" />}
      {loading && !data ? (
        <div className="p-5">
          <SkeletonBlock rows={10} />
        </div>
      ) : data && data.document.id === documentId ? (
        <>
          <div className="sticky top-0 z-10 bg-surface px-5 pt-3">
            <Tabs
              value={tab}
              onChange={setTab}
              items={[
                { key: 'text', label: 'Extracted text', count: data.pages.length },
                { key: 'chunks', label: 'Chunks', count: data.chunks.length },
                { key: 'entities', label: 'Entities', count: data.entities.length },
                { key: 'relations', label: 'Relations', count: data.relations.length },
                { key: 'metadata', label: 'Metadata' },
              ]}
            />
          </div>
          <div className="p-5">
            {tab === 'text' && <TextTab data={data} />}
            {tab === 'chunks' && <ChunksTab data={data} />}
            {tab === 'entities' && <EntitiesTab entities={data.entities} />}
            {tab === 'relations' && <RelationsTab data={data} />}
            {tab === 'metadata' && <MetadataTab data={data} />}
          </div>
        </>
      ) : null}
    </Drawer>
  )
}

function TextTab({ data }: { data: DocumentDetail }) {
  if (!data.pages.length) return <EmptyState title="No extracted text" description="The document has not been processed yet." />
  return (
    <div className="space-y-4">
      {data.sections.length > 0 && (
        <div>
          <p className="label">Detected sections</p>
          <div className="flex flex-wrap gap-1.5">
            {data.sections.map((s, i) => (
              <Badge key={i} tone="neutral" title={`chars ${s.char_start}–${s.char_end}`}>
                {s.title}
              </Badge>
            ))}
          </div>
        </div>
      )}
      {data.pages.map((p) => (
        <div key={p.page_number} className="rounded-lg border border-line">
          <div className="flex items-center justify-between border-b border-line px-3 py-1.5 text-xs text-muted">
            <span className="font-medium text-fg">Page {p.page_number}</span>
            <span>
              offset {fmtNumber(p.char_offset)} · {fmtNumber(p.text.length)} chars
            </span>
          </div>
          <pre className="max-h-[60vh] overflow-auto px-4 py-3 font-sans text-sm leading-relaxed whitespace-pre-wrap text-fg/85">{p.text}</pre>
        </div>
      ))}
    </div>
  )
}

function ChunksTab({ data }: { data: DocumentDetail }) {
  const s = data.chunk_stats
  if (!data.chunks.length) return <EmptyState title="No chunks" description="This document produced no passages." />
  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 gap-2 sm:grid-cols-5">
        {(
          [
            ['Chunks', fmtNumber(s.total_chunks)],
            ['Avg tokens', fmtNumber(s.avg_tokens, 1)],
            ['Min tokens', fmtNumber(s.min_tokens)],
            ['Max tokens', fmtNumber(s.max_tokens)],
            ['Avg chars', fmtNumber(s.avg_chars, 1)],
          ] as const
        ).map(([k, v]) => (
          <div key={k} className="rounded-lg border border-line bg-surface-2/50 px-3 py-2">
            <div className="text-[11px] text-muted">{k}</div>
            <div className="text-sm font-semibold text-fg tabular-nums">{v}</div>
          </div>
        ))}
      </div>
      {data.chunks.map((c) => (
        <div key={c.id} className="rounded-lg border border-line">
          <div className="flex flex-wrap items-center gap-2 border-b border-line px-3 py-1.5 text-xs text-muted">
            <span className="font-medium text-fg">#{c.chunk_index + 1}</span>
            <Badge tone="accent">{c.token_count} tokens</Badge>
            <span>
              chars {fmtNumber(c.char_start)}–{fmtNumber(c.char_end)}
            </span>
            <span>
              {c.page_number !== null ? (c.page_end && c.page_end !== c.page_number ? `pp. ${c.page_number}–${c.page_end}` : `p. ${c.page_number}`) : ''}
            </span>
            {c.section && <span className="truncate">§ {c.section}</span>}
            <span className="mono ml-auto text-subtle">{c.id}</span>
          </div>
          <p className="px-4 py-3 text-sm leading-relaxed whitespace-pre-wrap text-fg/85">{c.text}</p>
        </div>
      ))}
    </div>
  )
}

function EntitiesTab({ entities }: { entities: DocumentEntity[] }) {
  const groups = useMemo(() => {
    const m = new Map<string, DocumentEntity[]>()
    for (const e of entities) {
      const arr = m.get(e.label) ?? []
      arr.push(e)
      m.set(e.label, arr)
    }
    return [...m.entries()].sort((a, b) => b[1].length - a[1].length)
  }, [entities])
  if (!entities.length) return <EmptyState title="No entities extracted" />
  return (
    <div className="space-y-5">
      {groups.map(([label, list]) => {
        const color = NER_LABEL_COLORS[label] ?? '#94a3b8'
        return (
          <div key={label}>
            <div className="mb-2 flex items-center gap-2">
              <span className="h-2 w-2 rounded-full" style={{ background: color }} />
              <span className="text-xs font-semibold tracking-wide text-fg">{label}</span>
              <span className="text-xs text-subtle">{list.length}</span>
            </div>
            <div className="flex flex-wrap gap-1.5">
              {list
                .slice()
                .sort((a, b) => b.mentions - a.mentions)
                .map((e) => (
                  <Badge key={e.normalized} color={color} title={`${e.mentions} mentions in ${e.chunks} chunk(s) · normalized: ${e.normalized}`}>
                    {e.text}
                    <span className="opacity-60">×{e.mentions}</span>
                  </Badge>
                ))}
            </div>
          </div>
        )
      })}
    </div>
  )
}

function RelationsTab({ data }: { data: DocumentDetail }) {
  if (!data.relations.length) return <EmptyState title="No relations extracted" />
  return (
    <ul className="space-y-2">
      {data.relations.map((r, i) => (
        <li key={i} className="rounded-lg border border-line px-3 py-2.5">
          <div className="flex flex-wrap items-center gap-2 text-sm">
            <span className="font-medium text-emerald-300">{r.source_text}</span>
            <span className="inline-flex items-center gap-1 rounded bg-surface-3 px-1.5 py-0.5 text-xs text-indigo-200">
              {r.relation} <ArrowRight size={12} />
            </span>
            <span className="font-medium text-emerald-300">{r.target_text}</span>
          </div>
          <p className="mt-1 line-clamp-2 text-xs text-muted" title={r.sentence}>
            {r.sentence}
          </p>
        </li>
      ))}
    </ul>
  )
}

function MetadataTab({ data }: { data: DocumentDetail }) {
  const d = data.document
  return (
    <div className="space-y-4">
      <KeyValue
        items={[
          ['Name', d.name],
          ['ID', <span className="mono">{d.id}</span>],
          ['Status', <StatusBadge status={d.status} />],
          ['File type', d.file_type],
          ['Source', d.source_type ?? '—'],
          ['Size', fmtBytes(d.size_bytes)],
          ['Pages', fmtNumber(d.num_pages)],
          ['Chunks', fmtNumber(d.num_chunks)],
          ['Entities', fmtNumber(d.num_entities)],
          ['Sections', fmtNumber(data.sections.length)],
          ['Created', fmtDate(d.created_at)],
          ['Processed', fmtDate(d.processed_at)],
          ['Processing time', fmtMs(d.processing_ms)],
          ['Checksum', d.checksum ? <span className="mono break-all">{d.checksum}</span> : '—'],
          ['Stored at', d.path ? <span className="mono break-all">{d.path}</span> : '—'],
          ...(d.url ? ([['URL', d.url]] as Array<[string, string]>) : []),
        ]}
      />
      {d.error && <ErrorBanner title="Processing error" message={d.error} />}
    </div>
  )
}
