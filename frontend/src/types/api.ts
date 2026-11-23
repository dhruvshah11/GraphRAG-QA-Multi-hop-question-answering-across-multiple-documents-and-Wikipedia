// TypeScript shapes of the GraphRAG-QA FastAPI responses.
// Derived from real responses of the running backend; fields the backend may
// omit or null are typed accordingly.

export type Strategy = 'faiss' | 'bm25' | 'hybrid' | 'graph' | 'gnn'
export type GeneratorBackend = 'auto' | 'extractive' | 'hf_reader' | 'hf_generator' | 'ollama' | 'anthropic'
export type QAMode = 'multi_document' | 'open_domain'
export type JobStatus = 'queued' | 'running' | 'completed' | 'failed'
export type DocumentStatus = 'uploaded' | 'queued' | 'processing' | 'processed' | 'failed'
export type PipelineStatus = 'ready' | 'degraded' | 'untrained' | 'idle'
export type GraphNodeType = 'question' | 'document' | 'passage' | 'entity'
export type GraphEdgeType = 'CONTAINS' | 'MENTIONS' | 'RELATED_TO' | 'CO_OCCURS_WITH' | 'SUPPORTS' | 'CONNECTS_TO'

/** FastAPI error payload. `detail` is a string for HTTPException, an array for validation errors. */
export interface ApiErrorBody {
  detail?: string | Array<{ loc?: Array<string | number>; msg?: string; type?: string }>
}

// --------------------------------------------------------------------- health
export interface HealthComponent {
  status: string
  [key: string]: unknown
}

export interface HealthResponse {
  status: string
  components: Record<string, HealthComponent>
  warmup?: WarmupState | null
  started_at?: string | null
}

export interface WarmupState {
  status: string
  started_at?: string | null
  finished_at?: string | null
  ms?: Record<string, number>
}

// ----------------------------------------------------------------------- jobs
export interface Job {
  id: string
  kind: string
  status: JobStatus
  progress: number
  message?: string | null
  result?: unknown
  created_at?: string | null
  updated_at?: string | null
}

// ------------------------------------------------------------------ documents
export interface DocumentRecord {
  id: string
  name: string
  file_type: string
  source_type?: string | null
  path?: string | null
  url?: string | null
  size_bytes: number
  num_pages: number | null
  num_chunks: number | null
  num_entities: number | null
  status: DocumentStatus
  error: string | null
  checksum?: string | null
  created_at: string | null
  processed_at: string | null
  processing_ms: number | null
}

export interface DocumentStats {
  total_documents: number
  processed_documents: number
  total_pages: number
  total_chunks: number
  avg_chunk_tokens: number | null
  avg_chunk_chars: number | null
  total_entities: number
  entity_mentions: number
}

export interface DocumentsResponse {
  documents: DocumentRecord[]
  stats: DocumentStats
}

export interface DocumentPage {
  page_number: number
  text: string
  char_offset: number
}

export interface DocumentSection {
  title: string
  char_start: number
  char_end: number
}

export interface DocumentChunk {
  id: string
  document_id?: string
  document_name?: string
  chunk_index: number
  text: string
  page_number: number | null
  page_end: number | null
  char_start: number
  char_end: number
  token_count: number
  section: string | null
  url?: string | null
  source_type?: string | null
}

export interface ChunkStats {
  total_chunks: number
  avg_tokens: number | null
  min_tokens: number | null
  max_tokens: number | null
  avg_chars: number | null
}

export interface DocumentEntity {
  normalized: string
  label: string
  text: string
  mentions: number
  chunks: number
}

export interface DocumentRelation {
  source_text: string
  relation: string
  target_text: string
  sentence: string
  source_normalized?: string
  target_normalized?: string
  chunk_id?: string
  document_id?: string
}

export interface DocumentDetail {
  document: DocumentRecord
  pages: DocumentPage[]
  sections: DocumentSection[]
  chunks: DocumentChunk[]
  chunk_stats: ChunkStats
  entities: DocumentEntity[]
  relations: DocumentRelation[]
}

export interface UploadResult {
  filename: string
  document?: DocumentRecord | null
  duplicate?: boolean | null
  job_id?: string | null
  error?: string | null
}

export interface UploadResponse {
  results: UploadResult[]
}

// ------------------------------------------------------------------------- QA
export interface QARequest {
  question: string
  mode?: QAMode
  strategy: Strategy
  top_k: number
  max_hops: number
  backend: GeneratorBackend
}

export interface ReasoningStep {
  step: number
  kind: 'fact' | 'inference' | string
  text: string
  citations: number[]
}

export interface EvidenceChainStep {
  step: number
  chunk_id: string
  document_name: string
  page_number: number | null
  section: string | null
  url?: string | null
  via_entity_id?: string | null
  via_entity_name: string | null
  relevance: number | null
  reason: string | null
  evidence_index: number | null
  text: string
}

export interface Citation {
  index: number
  chunk_id: string
  document_id?: string
  document_name: string
  page_number: number | null
  page_end: number | null
  section: string | null
  url: string | null
  source_type: string
  passage_text: string
  supporting_sentence: string
  sentence_start: number
  sentence_end: number
  retrieval_score: number | null
  graph_score: number | null
  gnn_score: number | null
  support_score: number | null
  claim: string | null
}

export interface RetrievedPassage {
  chunk_id: string
  document_id?: string
  document_name: string
  page_number: number | null
  page_end?: number | null
  section: string | null
  url?: string | null
  text: string
  char_start?: number
  char_end?: number
  semantic_score: number | null
  bm25_raw?: number | null
  bm25_score: number | null
  hybrid_score: number | null
  graph_score: number | null
  gnn_score: number | null
  final_score: number | null
  rank: number
  origin: string
  hop: number | null
}

export interface ChainStepRef {
  chunk_id: string
  via_entity: string | null
  via_entity_name: string | null
  relevance: number | null
  reason: string | null
}

export interface ReasoningChain {
  chunk_ids: string[]
  steps: ChainStepRef[]
  score: number
  components: Record<string, number>
  covered_terms?: string[]
  missing_terms?: string[]
  length?: number
}

export interface ScoreRow {
  chunk_id: string
  document_name: string
  page_number: number | null
  section?: string | null
  preview: string
  origin: string
  hop: number | null
  semantic_score: number | null
  bm25_score: number | null
  hybrid_score: number | null
  graph_score: number | null
  gnn_score: number | null
  ranks: Partial<Record<Strategy, number | null>>
}

export interface EvidenceCoverage {
  question_terms: string[]
  covered: string[]
  missing: string[]
  ratio: number
}

export interface QAResult {
  query_id: string
  question: string
  mode: QAMode | string
  strategy: Strategy | string
  effective_strategy: string | null
  answer: string
  candidate_answer?: string | null
  answer_sentence: string | null
  answer_citations?: number[]
  reasoning_summary?: string | null
  reasoning_steps: ReasoningStep[]
  evidence_chain: EvidenceChainStep[]
  chains?: ReasoningChain[]
  citations: Citation[]
  retrieved_passages: RetrievedPassage[]
  strategy_rankings?: Record<string, string[]>
  score_table?: ScoreRow[]
  confidence: number
  confidence_components: Record<string, number>
  evidence_coverage: EvidenceCoverage | null
  insufficient_evidence: boolean | number
  insufficient_reasons: string[]
  generator_backend: string | null
  generator_warnings: string[]
  notes: string[]
  gnn_used?: boolean
  timings_ms: Record<string, number>
  total_ms: number | null
  top_k?: number
  max_hops?: number
  graph_stats?: GraphStats | null
  corpus?: { name: string; documents: number; chunks: number } | null
  open_domain?: Record<string, unknown> | null
}

export interface QueryHistoryItem {
  id: string
  question: string
  mode: string
  strategy: string
  top_k?: number
  max_hops?: number
  created_at: string
  latency_ms: number | null
  status?: string
  error?: string | null
  answer: string | null
  confidence: number | null
  insufficient_evidence: boolean | number | null
  generator_backend?: string | null
}

export interface QAHistoryResponse {
  queries: QueryHistoryItem[]
}

// ---------------------------------------------------------------------- graph
export interface GraphNode {
  id: string
  type: GraphNodeType
  label: string
  data: Record<string, unknown>
  highlighted: boolean
  position?: { x: number; y: number } | null
}

export interface GraphEdge {
  id: string
  source: string
  target: string
  type: GraphEdgeType
  weight: number
  data: Record<string, unknown>
  highlighted: boolean
}

export interface GraphStats {
  nodes: number
  edges: number
  node_types: Record<string, number>
  edge_types: Record<string, number>
}

export interface GraphResponse {
  nodes: GraphNode[]
  edges: GraphEdge[]
  stats: GraphStats
  full_stats?: GraphStats
  truncated?: boolean
  query_id?: string
  question?: string
  view?: string
}

export interface NeighborsResponse {
  center: string
  node: GraphNode
  nodes: GraphNode[]
  edges: GraphEdge[]
  stats?: GraphStats
}

// ------------------------------------------------------------------ analytics
export interface PipelineStage {
  key: string
  label: string
  tech: string
  status: PipelineStatus
  detail: string
}

export interface StrategyMetrics {
  questions?: number
  exact_match?: number | null
  f1?: number | null
  precision_at_k?: number | null
  recall_at_k?: number | null
  mrr?: number | null
  evidence_recall?: number | null
  citation_coverage?: number | null
  citation_precision?: number | null
  abstention_accuracy?: number | null
  false_abstention_rate?: number | null
  answer_latency_ms?: number | null
  graph_latency_ms?: number | null
}

export interface OverviewResponse {
  project: { name: string; description: string }
  counts: {
    documents: number
    processed_documents: number
    pages: number
    passages: number
    entities: number
    entity_mentions: number
    relations: number
    graph_nodes: number
    graph_edges: number
    questions_answered: number
    wikipedia_cache_entries: number
    avg_chunk_tokens: number | null
  }
  graph: GraphStats
  recent_queries: QueryHistoryItem[]
  latency: { queries_measured: number; avg_query_ms: number | null; p95_query_ms: number | null }
  latest_evaluations: Record<string, { run_id: string; created_at?: string; summary: Record<string, StrategyMetrics> }>
  pipeline: PipelineStage[]
  health: HealthResponse
}

export interface StageLatency {
  stage: string
  count: number
  mean_ms: number
  p50_ms: number
  p95_ms: number
  max_ms: number
}

export interface PerformanceResponse {
  query_stages: StageLatency[]
  ingestion_stages: StageLatency[]
  recent_queries: Array<{
    query_id: string
    question: string
    mode: string
    strategy: string
    latency_ms: number | null
    created_at: string
  }>
  benchmark: Record<string, unknown> | null
  warmup?: WarmupState | null
  model_load_ms: Record<string, number | null>
}

export interface GnnHistoryRecord {
  epoch: number
  train_loss: number
  val_loss?: number | null
  val_mrr?: number | null
  val_recall_at_k?: number | null
  val_precision_at_k?: number | null
  [key: string]: number | null | undefined
}

export interface GnnHistory {
  config?: Record<string, unknown>
  meta?: Record<string, unknown>
  history: GnnHistoryRecord[]
  summary?: {
    best_epoch?: number
    epochs_run?: number
    train_seconds?: number
    best_val?: GnnHistoryRecord
    baseline_val_hybrid?: Record<string, number>
    num_parameters?: number
    [key: string]: unknown
  }
}

export interface GnnStatus {
  available: boolean
  reason: string | null
  checkpoint_path?: string
  checkpoint_exists?: boolean
  feature_spec_version?: string
  architecture?: string
  num_parameters: number | null
  meta: Record<string, unknown>
  history_available?: boolean
  history?: GnnHistory | null
  latest_job?: Job | null
  hotpotqa_available?: boolean
}

export interface EvaluationSummary {
  by_strategy?: Record<string, StrategyMetrics>
  by_type?: Record<string, Record<string, StrategyMetrics>>
  skipped?: string[]
  num_questions?: number
}

export interface EvaluationRunRef {
  run_id: string
  dataset: string
  created_at: string
  config: EvaluationConfig
  summary: EvaluationSummary
}

export interface RetrievalAnalyticsResponse {
  query_id: string | null
  question: string | null
  score_table: ScoreRow[]
  strategy_rankings: Record<string, string[]>
  chains: ReasoningChain[]
  effective_strategy?: string | null
  evaluation_runs: EvaluationRunRef[]
  gnn: GnnStatus
  recent_queries: QueryHistoryItem[]
}

// ----------------------------------------------------------------- evaluation
export interface EvaluationDataset {
  id: string
  name: string
  questions: number
  available: boolean
  description: string
}

export interface EvaluationConfig {
  dataset: string
  strategies: string[]
  top_k?: number | null
  max_hops?: number | null
  k?: number
  limit?: number | null
  backend?: string | null
  [key: string]: unknown
}

export interface EvaluationRun {
  id: string
  dataset: string
  status: JobStatus | string
  created_at: string
  finished_at: string | null
  config: EvaluationConfig
  summary: EvaluationSummary | null
  progress: number | null
  error: string | null
}

export interface EvaluationResultsResponse {
  runs: EvaluationRun[]
  datasets: EvaluationDataset[]
}

export interface EvaluationRow {
  id?: number
  run_id?: string
  question_id: string
  question_type: string
  question: string
  strategy: string
  expected_answer: string
  predicted_answer: string
  insufficient: boolean | number
  exact_match: number | null
  f1: number | null
  precision_at_k: number | null
  recall_at_k: number | null
  mrr: number | null
  evidence_recall: number | null
  citation_coverage: number | null
  citation_precision: number | null
  answer_latency_ms: number | null
  graph_latency_ms: number | null
  details: {
    generator_backend?: string
    confidence?: number
    effective_strategy?: string
    candidate_answer?: string
    top_ranked?: string[]
    gold?: string[]
    [key: string]: unknown
  } | null
}

export interface EvaluationRunDetail {
  run: EvaluationRun
  rows: EvaluationRow[]
  job: Job | null
}

export interface EvaluationRequest {
  dataset: string
  strategies: string[]
  top_k?: number
  max_hops?: number
  limit?: number
  k?: number
  backend?: GeneratorBackend
}

// ------------------------------------------------------------------- settings
export interface RuntimeSettings {
  chunk_size: number
  chunk_overlap: number
  top_k: number
  max_hops: number
  hybrid_alpha: number
  generator_backend: GeneratorBackend
  max_evidence_passages: number
  embedding_model: string
}

export interface GeneratorStatusEntry {
  available: boolean
  model?: string | null
  loaded?: boolean
  error?: string | null
  description?: string
  url?: string
}

export interface SettingsResponse {
  runtime: RuntimeSettings
  defaults: RuntimeSettings
  options: { generator_backends: GeneratorBackend[]; embedding_models: string[]; strategies: Strategy[] }
  generator_status: Record<string, GeneratorStatusEntry>
  gnn_status: GnnStatus
  models: Record<string, string | null>
  storage: { paths: Record<string, string>; bytes: Record<string, number> }
  limits: { max_upload_mb: number; allowed_extensions: string[] }
}

export interface SettingsUpdateResponse {
  runtime: RuntimeSettings
  changed: string[]
  requires_reprocessing: boolean
  requires_reindex: boolean
  reindex_job?: string | null
}

export interface ReprocessAllResponse {
  jobs: string[]
}
