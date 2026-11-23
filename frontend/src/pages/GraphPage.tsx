import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Link, useSearchParams } from 'react-router-dom'
import {
  Background,
  BackgroundVariant,
  Controls,
  MarkerType,
  MiniMap,
  ReactFlow,
  ReactFlowProvider,
  useEdgesState,
  useNodesState,
  useReactFlow,
  type Edge,
} from '@xyflow/react'
import '@xyflow/react/dist/style.css'
import { Crosshair, Network, RefreshCw, Search } from 'lucide-react'
import { useApi } from '../hooks/useApi'
import { errorMessage, getGraph, getNeighbors, getQAHistory, queryGraph } from '../services/api'
import type { GraphEdge, GraphNode, GraphNodeType, GraphResponse } from '../types/api'
import { nodeTypes, type FlowNode } from '../components/graph/GraphNodes'
import { GraphLegend } from '../components/graph/GraphLegend'
import { NodeDetailsPanel } from '../components/graph/NodeDetailsPanel'
import { Tabs } from '../components/Tabs'
import { EmptyState, ErrorBanner, Spinner } from '../components/Feedback'
import { EDGE_TYPE_STYLES, NODE_TYPE_COLORS, entityColor } from '../lib/constants'
import { cx, fmtNumber } from '../lib/format'

type View = 'full' | 'subgraph' | 'chain'
const DIRECTED = new Set(['RELATED_TO', 'SUPPORTS', 'CONTAINS', 'MENTIONS'])
const ALL_TYPES: GraphNodeType[] = ['document', 'passage', 'entity']

interface WorkingGraph {
  nodes: GraphNode[]
  edges: GraphEdge[]
  meta: GraphResponse
  loadKey: number
}

export default function GraphPage() {
  return (
    <ReactFlowProvider>
      <GraphExplorer />
    </ReactFlowProvider>
  )
}

function GraphExplorer() {
  const [params, setParams] = useSearchParams()
  const queryId = params.get('query')
  const view: View = (params.get('view') as View | null) ?? (queryId ? 'subgraph' : 'full')

  const [limitDraft, setLimitDraft] = useState(150)
  const [limit, setLimit] = useState(150)
  const [types, setTypes] = useState<GraphNodeType[]>(ALL_TYPES)
  const [search, setSearch] = useState('')
  const [dimOthers, setDimOthers] = useState(true)
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [graph, setGraph] = useState<WorkingGraph | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [reloadTick, setReloadTick] = useState(0)
  const [expanding, setExpanding] = useState(false)
  const [expandError, setExpandError] = useState<string | null>(null)
  const [expandedIds, setExpandedIds] = useState<Set<string>>(() => new Set())

  const history = useApi((s) => getQAHistory(s), [])

  const setView = (v: View) => {
    const next = new URLSearchParams(params)
    next.set('view', v)
    if (v !== 'full' && !next.get('query') && history.data?.queries[0]) next.set('query', history.data.queries[0].id)
    setParams(next)
  }
  const setQuery = (id: string) => {
    const next = new URLSearchParams(params)
    next.set('query', id)
    if (view === 'full') next.set('view', 'subgraph')
    setParams(next)
  }

  // ---- load graph data
  useEffect(() => {
    if (view !== 'full' && !queryId) {
      setGraph(null)
      return
    }
    const controller = new AbortController()
    setLoading(true)
    setError(null)
    setSelectedId(null)
    const p =
      view === 'full'
        ? getGraph({ limit, node_types: types.length === ALL_TYPES.length ? undefined : types }, controller.signal)
        : queryGraph({ query_id: queryId as string, view }, controller.signal)
    p.then((res) => {
      setExpandedIds(new Set())
      setGraph((prev) => ({ nodes: res.nodes, edges: res.edges, meta: res, loadKey: (prev?.loadKey ?? 0) + 1 }))
    })
      .catch((e) => {
        if (!controller.signal.aborted) {
          setError(errorMessage(e))
          setGraph(null)
        }
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false)
      })
    return () => controller.abort()
  }, [view, queryId, limit, types, reloadTick])

  const nodesById = useMemo(() => new Map((graph?.nodes ?? []).map((n) => [n.id, n])), [graph])

  // ---- flow state
  const [nodes, setNodes, onNodesChange] = useNodesState<FlowNode>([])
  const [edges, setEdges, onEdgesChange] = useEdgesState<Edge>([])
  const { fitView } = useReactFlow()

  const term = search.trim().toLowerCase()
  const matches = useMemo(() => {
    if (!term || !graph) return new Set<string>()
    return new Set(graph.nodes.filter((n) => n.label.toLowerCase().includes(term) || n.id.toLowerCase().includes(term)).map((n) => n.id))
  }, [graph, term])

  const anyHighlighted = !!graph?.nodes.some((n) => n.highlighted)
  const emphasis = anyHighlighted && dimOthers
  const lastLoadKey = useRef<number | null>(null)

  useEffect(() => {
    if (!graph) {
      setNodes([])
      setEdges([])
      return
    }
    // expanded neighbors and the selected node are never dimmed by evidence emphasis
    const keep = (id: string) => expandedIds.has(id) || id === selectedId
    const dimmedNode = (n: GraphNode) => (term ? !matches.has(n.id) : emphasis ? !n.highlighted && !keep(n.id) : false)
    const dimmedIds = new Set(graph.nodes.filter(dimmedNode).map((n) => n.id))
    const freshLoad = lastLoadKey.current !== graph.loadKey
    lastLoadKey.current = graph.loadKey
    setNodes((prev) => {
      // keep user-dragged positions within one load; use server positions for a new load
      const pos = new Map(freshLoad ? [] : prev.map((n) => [n.id, n.position]))
      return graph.nodes.map((n) => ({
        id: n.id,
        type: n.type,
        position: pos.get(n.id) ?? n.position ?? { x: 0, y: 0 },
        selected: n.id === selectedId,
        zIndex: n.highlighted ? 10 : n.type === 'question' ? 5 : 1,
        data: { node: n, dimmed: dimmedIds.has(n.id), matched: matches.has(n.id), emphasized: n.highlighted && anyHighlighted },
      }))
    })
    setEdges(
      graph.edges
        .filter((e) => nodesById.has(e.source) && nodesById.has(e.target))
        .map((e) => {
          const st = EDGE_TYPE_STYLES[e.type] ?? { color: '#64748b', label: e.type }
          const incident = selectedId !== null && (e.source === selectedId || e.target === selectedId)
          const dimmed = term
            ? dimmedIds.has(e.source) || dimmedIds.has(e.target)
            : emphasis
              ? !e.highlighted && (dimmedIds.has(e.source) || dimmedIds.has(e.target) || !(keep(e.source) || keep(e.target)))
              : false
          const color = e.highlighted ? '#a5b4fc' : st.color
          return {
            id: e.id,
            source: e.source,
            target: e.target,
            type: 'straight',
            animated: e.highlighted,
            zIndex: e.highlighted ? 9 : 0,
            style: {
              stroke: color,
              strokeWidth: e.highlighted ? 2.6 : incident ? 1.8 : 1,
              strokeDasharray: e.highlighted ? undefined : st.dash,
              opacity: dimmed && !incident ? 0.07 : e.highlighted || incident ? 1 : 0.45,
            },
            markerEnd: DIRECTED.has(e.type) ? { type: MarkerType.ArrowClosed, color, width: 14, height: 14 } : undefined,
          }
        }),
    )
  }, [graph, matches, term, emphasis, anyHighlighted, selectedId, expandedIds, nodesById, setNodes, setEdges])

  const focusMatches = () => {
    if (matches.size) void fitView({ nodes: [...matches].map((id) => ({ id })), duration: 400, maxZoom: 1.4, padding: 0.3 })
  }

  // ---- neighbor expansion
  const expand = useCallback(async () => {
    if (!selectedId || !graph) return
    setExpanding(true)
    setExpandError(null)
    try {
      const res = await getNeighbors(selectedId)
      const center = nodes.find((n) => n.id === selectedId)?.position ?? { x: 0, y: 0 }
      const existing = new Set(graph.nodes.map((n) => n.id))
      const fresh = res.nodes.filter((n) => !existing.has(n.id))
      const radius = 180 + Math.min(220, fresh.length * 9)
      const placed = fresh.map((n, i) => {
        const angle = (2 * Math.PI * i) / Math.max(1, fresh.length) - Math.PI / 2
        return { ...n, position: { x: center.x + radius * Math.cos(angle), y: center.y + radius * Math.sin(angle) } }
      })
      const edgeIds = new Set(graph.edges.map((e) => e.id))
      const newEdges = res.edges.filter((e) => !edgeIds.has(e.id))
      setGraph({ ...graph, nodes: [...graph.nodes, ...placed], edges: [...graph.edges, ...newEdges] })
      const neighborIds = res.nodes.map((n) => n.id)
      setExpandedIds((prev) => new Set([...prev, ...neighborIds]))
      if (!fresh.length && !newEdges.length) setExpandError('All neighbors are already on the canvas.')
      else window.setTimeout(() => void fitView({ nodes: [selectedId, ...neighborIds].map((id) => ({ id })), duration: 450, padding: 0.25, maxZoom: 1.2 }), 60)
    } catch (e) {
      setExpandError(errorMessage(e))
    } finally {
      setExpanding(false)
    }
  }, [selectedId, graph, nodes, fitView])

  useEffect(() => setExpandError(null), [selectedId])

  const selected = selectedId ? nodesById.get(selectedId) ?? null : null
  const stats = graph?.meta.stats
  const liveNodeCounts = useMemo(() => {
    const c: Record<string, number> = {}
    for (const n of graph?.nodes ?? []) c[n.type] = (c[n.type] ?? 0) + 1
    return c
  }, [graph])
  const liveEdgeCounts = useMemo(() => {
    const c: Record<string, number> = {}
    for (const e of graph?.edges ?? []) c[e.type] = (c[e.type] ?? 0) + 1
    return c
  }, [graph])
  const categories = useMemo(() => {
    const s = new Set<string>()
    for (const n of graph?.nodes ?? []) if (n.type === 'entity' && typeof n.data.category === 'string') s.add(n.data.category)
    return [...s].sort()
  }, [graph])

  const queries = history.data?.queries ?? []

  return (
    <div className="flex flex-col gap-4 lg:h-[calc(100vh-3rem)]">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="text-xl font-semibold tracking-tight text-fg sm:text-2xl">Knowledge Graph Explorer</h1>
          <p className="mt-1 text-sm text-muted">Documents, passages and entities linked by the evidence graph. Select a node to inspect it or expand its neighborhood.</p>
        </div>
        <button type="button" className="btn-secondary" onClick={() => setReloadTick((t) => t + 1)} disabled={loading}>
          <RefreshCw size={14} className={loading ? 'animate-spin' : ''} /> Reload
        </button>
      </div>

      {/* toolbar */}
      <div className="flex flex-wrap items-end gap-x-5 gap-y-3 rounded-xl border border-line bg-surface px-4 py-3">
        <div>
          <span className="label">View</span>
          <Tabs
            variant="pill"
            value={view}
            onChange={setView}
            items={[
              { key: 'full', label: 'Full graph' },
              { key: 'subgraph', label: 'Question subgraph' },
              { key: 'chain', label: 'Evidence chain' },
            ]}
          />
        </div>

        {view === 'full' ? (
          <>
            <div className="w-48">
              <label className="label" htmlFor="limit">
                Node limit <span className="text-fg tabular-nums">{limitDraft}</span>
              </label>
              <input
                id="limit"
                type="range"
                min={10}
                max={1000}
                step={10}
                value={limitDraft}
                onChange={(e) => setLimitDraft(Number(e.target.value))}
                onPointerUp={() => setLimit(limitDraft)}
                onKeyUp={() => setLimit(limitDraft)}
                className="w-full accent-indigo-500"
              />
            </div>
            <div>
              <span className="label">Node types</span>
              <div className="flex gap-3 py-1">
                {ALL_TYPES.map((t) => (
                  <label key={t} className="flex items-center gap-1.5 text-xs text-muted">
                    <input
                      type="checkbox"
                      className="accent-indigo-500"
                      checked={types.includes(t)}
                      onChange={(e) => setTypes((prev) => (e.target.checked ? ALL_TYPES.filter((x) => x === t || prev.includes(x)) : prev.filter((x) => x !== t)))}
                    />
                    <span style={{ color: NODE_TYPE_COLORS[t] }}>{t}</span>
                  </label>
                ))}
              </div>
            </div>
          </>
        ) : (
          <div className="min-w-[240px] flex-1 sm:max-w-md">
            <label className="label" htmlFor="qpick">
              Question
            </label>
            <select id="qpick" className="input" value={queryId ?? ''} onChange={(e) => setQuery(e.target.value)}>
              <option value="" disabled>
                {history.loading ? 'Loading questions…' : queries.length ? 'Select a previous question' : 'No questions asked yet'}
              </option>
              {queryId && !queries.some((q) => q.id === queryId) && <option value={queryId}>{graph?.meta.question ?? queryId}</option>}
              {queries.map((q) => (
                <option key={q.id} value={q.id}>
                  {q.question}
                </option>
              ))}
            </select>
          </div>
        )}

        <div className="min-w-[200px] flex-1 sm:max-w-xs">
          <label className="label" htmlFor="gsearch">
            Search nodes {term && <span className="text-fg">· {matches.size} match{matches.size === 1 ? '' : 'es'}</span>}
          </label>
          <div className="flex gap-1.5">
            <div className="relative flex-1">
              <Search size={14} className="pointer-events-none absolute top-1/2 left-2.5 -translate-y-1/2 text-subtle" />
              <input
                id="gsearch"
                className="input pl-8"
                placeholder="Filter by label…"
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && focusMatches()}
              />
            </div>
            <button type="button" className="btn-secondary btn-sm" onClick={focusMatches} disabled={!matches.size} title="Zoom to matches">
              <Crosshair size={14} />
            </button>
          </div>
        </div>

        {anyHighlighted && (
          <label className="flex items-center gap-2 pb-2 text-xs text-muted">
            <input type="checkbox" className="accent-indigo-500" checked={dimOthers} onChange={(e) => setDimOthers(e.target.checked)} />
            Dim non-evidence
          </label>
        )}
      </div>

      {error && <ErrorBanner message={error} onRetry={() => setReloadTick((t) => t + 1)} />}

      <div className="flex min-h-[560px] flex-1 flex-col gap-4 lg:flex-row">
        <div className="relative min-h-[520px] flex-1 overflow-hidden rounded-xl border border-line bg-[#0a0e16]">
          {view !== 'full' && !queryId ? (
            <EmptyState
              className="h-full"
              icon={<Network size={18} />}
              title="Pick a question"
              description={
                queries.length ? (
                  'Select a previously answered question above to view its subgraph or evidence chain.'
                ) : (
                  <>
                    No questions have been answered yet.{' '}
                    <Link to="/qa" className="text-indigo-300 hover:underline">
                      Ask one
                    </Link>{' '}
                    to build a question subgraph.
                  </>
                )
              }
            />
          ) : graph && graph.nodes.length === 0 && !loading ? (
            <EmptyState className="h-full" icon={<Network size={18} />} title="The graph is empty" description="Upload and process documents to populate the evidence graph." />
          ) : (
            <ReactFlow
              key={graph?.loadKey ?? 0}
              nodes={nodes}
              edges={edges}
              nodeTypes={nodeTypes}
              onNodesChange={onNodesChange}
              onEdgesChange={onEdgesChange}
              onNodeClick={(_, n) => setSelectedId(n.id)}
              onPaneClick={() => setSelectedId(null)}
              nodesConnectable={false}
              fitView
              fitViewOptions={{ padding: 0.15 }}
              minZoom={0.05}
              maxZoom={2.5}
              onlyRenderVisibleElements={nodes.length > 400}
              proOptions={{ hideAttribution: false }}
              colorMode="dark"
            >
              <Background variant={BackgroundVariant.Dots} gap={22} size={1} color="#1e2736" />
              <Controls showInteractive={false} position="bottom-left" />
              <MiniMap
                pannable
                zoomable
                position="bottom-right"
                nodeColor={(n) => {
                  const g = (n as FlowNode).data?.node
                  if (!g) return '#475569'
                  return g.type === 'entity' ? entityColor(g.data.category) : (NODE_TYPE_COLORS[g.type] ?? '#475569')
                }}
                nodeStrokeWidth={0}
                maskColor="rgba(8,11,18,0.7)"
              />
            </ReactFlow>
          )}

          {loading && (
            <div className="absolute inset-0 z-10 flex items-center justify-center bg-bg/50 backdrop-blur-[1px]">
              <div className="flex items-center gap-2 rounded-lg border border-line bg-surface px-3 py-2 text-sm text-muted">
                <Spinner /> Loading graph…
              </div>
            </div>
          )}

          {graph && (
            <>
              <div className="pointer-events-none absolute top-3 left-3 z-[5] flex max-w-[60%] flex-col gap-1.5">
                <div className="pointer-events-auto rounded-lg border border-line-strong bg-surface/95 px-3 py-2 text-xs text-muted backdrop-blur">
                  <span className="font-semibold text-fg tabular-nums">{fmtNumber(graph.nodes.length)}</span> nodes ·{' '}
                  <span className="font-semibold text-fg tabular-nums">{fmtNumber(graph.edges.length)}</span> edges
                  {graph.meta.full_stats && (
                    <span>
                      {' '}
                      of {fmtNumber(graph.meta.full_stats.nodes)} / {fmtNumber(graph.meta.full_stats.edges)}
                    </span>
                  )}
                  {graph.meta.truncated && <span className="ml-1 text-amber-300">(truncated to most connected)</span>}
                  {stats && graph.nodes.length > stats.nodes && <span className="ml-1 text-indigo-300">+{graph.nodes.length - stats.nodes} expanded</span>}
                </div>
                {graph.meta.question && view !== 'full' && (
                  <div className="pointer-events-auto line-clamp-2 rounded-lg border border-line-strong bg-surface/95 px-3 py-2 text-xs text-fg backdrop-blur">
                    <span className="text-subtle">Q: </span>
                    {graph.meta.question}{' '}
                    <Link to={`/qa?query=${encodeURIComponent(queryId ?? '')}`} className="text-indigo-300 hover:underline">
                      answer →
                    </Link>
                  </div>
                )}
              </div>
              <div className={cx('absolute top-3 right-3 z-[5] hidden', selected ? '2xl:block' : 'md:block')}>
                <GraphLegend nodeCounts={liveNodeCounts} edgeCounts={liveEdgeCounts} categories={categories} />
              </div>
            </>
          )}
        </div>

        {selected && (
          <aside className={cx('w-full shrink-0 overflow-hidden rounded-xl border border-line bg-surface lg:w-80', 'max-h-[70vh] lg:max-h-none')}>
            <NodeDetailsPanel
              node={selected}
              edges={graph?.edges ?? []}
              nodesById={nodesById}
              onClose={() => setSelectedId(null)}
              onExpand={() => void expand()}
              expanding={expanding}
              expandError={expandError}
              onSelect={(id) => {
                setSelectedId(id)
                void fitView({ nodes: [{ id }], duration: 350, maxZoom: 1.3, padding: 0.6 })
              }}
            />
          </aside>
        )}
      </div>
    </div>
  )
}
