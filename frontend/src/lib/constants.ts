import type { Strategy } from '../types/api'

export const STRATEGIES: Strategy[] = ['faiss', 'bm25', 'hybrid', 'graph', 'gnn']

export const STRATEGY_LABELS: Record<string, string> = {
  faiss: 'FAISS semantic',
  bm25: 'BM25 lexical',
  hybrid: 'Hybrid',
  graph: 'Graph heuristic rerank',
  gnn: 'GNN (GAT) rerank',
}

export const STRATEGY_SHORT: Record<string, string> = {
  faiss: 'FAISS',
  bm25: 'BM25',
  hybrid: 'Hybrid',
  graph: 'Graph',
  gnn: 'GNN',
}

/** Fixed categorical colors per strategy, used consistently by every chart. */
export const STRATEGY_COLORS: Record<string, string> = {
  faiss: '#60a5fa',
  bm25: '#f59e0b',
  hybrid: '#2dd4bf',
  graph: '#a78bfa',
  gnn: '#f472b6',
}

/** Passage-level score series (score bars and grouped bar charts). */
export const SCORE_SERIES = [
  { key: 'semantic_score', label: 'Semantic', color: STRATEGY_COLORS.faiss },
  { key: 'bm25_score', label: 'BM25', color: STRATEGY_COLORS.bm25 },
  { key: 'hybrid_score', label: 'Hybrid', color: STRATEGY_COLORS.hybrid },
  { key: 'graph_score', label: 'Graph', color: STRATEGY_COLORS.graph },
  { key: 'gnn_score', label: 'GNN', color: STRATEGY_COLORS.gnn },
] as const

export const BACKEND_LABELS: Record<string, string> = {
  auto: 'Auto (best available)',
  extractive: 'Extractive (model-free)',
  hf_reader: 'HF reader (extractive QA)',
  hf_generator: 'HF generator (seq2seq)',
  ollama: 'Ollama (local LLM)',
  anthropic: 'Anthropic API',
}

export const METRIC_LABELS: Record<string, string> = {
  exact_match: 'Exact match',
  f1: 'Token F1',
  precision_at_k: 'P@K',
  recall_at_k: 'R@K',
  mrr: 'MRR',
  evidence_recall: 'Evidence recall',
  citation_coverage: 'Citation coverage',
  citation_precision: 'Citation precision',
  abstention_accuracy: 'Abstention acc.',
  false_abstention_rate: 'False abstention',
  answer_latency_ms: 'Answer latency',
  graph_latency_ms: 'Graph latency',
}

/** Pipeline stages in execution order, used for the QA progress indicator. */
export const QUERY_STAGE_ORDER = [
  'query_embedding',
  'semantic_retrieval',
  'bm25_retrieval',
  'hybrid_retrieval',
  'question_analysis',
  'wikipedia_retrieval',
  'graph_construction',
  'graph_traversal',
  'graph_ranking',
  'gnn_reranking',
  'answer_generation',
  'citation_validation',
]

/** Stages animated while a request is in flight (a subset that always runs). */
export const PROGRESS_STAGES = [
  'query_embedding',
  'hybrid_retrieval',
  'question_analysis',
  'graph_construction',
  'graph_traversal',
  'graph_ranking',
  'answer_generation',
  'citation_validation',
]

export const STAGE_LABELS: Record<string, string> = {
  query_embedding: 'Query embedding',
  semantic_retrieval: 'Semantic retrieval',
  bm25_retrieval: 'BM25 retrieval',
  hybrid_retrieval: 'Hybrid fusion',
  question_analysis: 'Question analysis',
  wikipedia_retrieval: 'Wikipedia retrieval',
  graph_construction: 'Graph construction',
  graph_traversal: 'Graph traversal',
  graph_ranking: 'Graph ranking',
  gnn_reranking: 'GNN reranking',
  answer_generation: 'Answer generation',
  citation_validation: 'Citation validation',
  extraction: 'Text extraction',
  chunking: 'Chunking',
  entity_extraction: 'Entity extraction',
  store_entities: 'Store entities',
  embedding: 'Embedding',
  faiss_update: 'FAISS update',
  graph_refresh: 'Graph refresh',
}

export const NODE_TYPE_COLORS: Record<string, string> = {
  question: '#f43f5e',
  document: '#38bdf8',
  passage: '#818cf8',
  entity: '#34d399',
}

export const ENTITY_CATEGORY_COLORS: Record<string, string> = {
  PERSON: '#34d399',
  ORGANIZATION: '#fbbf24',
  LOCATION: '#22d3ee',
  DATE: '#a3a3a3',
  EVENT: '#fb7185',
  WORK_OF_ART: '#c084fc',
  PRODUCT: '#f97316',
  NORP: '#facc15',
  LAW: '#94a3b8',
  QUANTITY: '#a1a1aa',
  MISC: '#86efac',
  OTHER: '#86efac',
}

/** spaCy NER labels -> colors (documents page). */
export const NER_LABEL_COLORS: Record<string, string> = {
  PERSON: ENTITY_CATEGORY_COLORS.PERSON,
  ORG: ENTITY_CATEGORY_COLORS.ORGANIZATION,
  ORGANIZATION: ENTITY_CATEGORY_COLORS.ORGANIZATION,
  GPE: ENTITY_CATEGORY_COLORS.LOCATION,
  LOC: ENTITY_CATEGORY_COLORS.LOCATION,
  LOCATION: ENTITY_CATEGORY_COLORS.LOCATION,
  FAC: '#67e8f9',
  DATE: ENTITY_CATEGORY_COLORS.DATE,
  TIME: ENTITY_CATEGORY_COLORS.DATE,
  EVENT: ENTITY_CATEGORY_COLORS.EVENT,
  WORK_OF_ART: ENTITY_CATEGORY_COLORS.WORK_OF_ART,
  PRODUCT: ENTITY_CATEGORY_COLORS.PRODUCT,
  NORP: ENTITY_CATEGORY_COLORS.NORP,
  LAW: ENTITY_CATEGORY_COLORS.LAW,
  MONEY: '#4ade80',
  CARDINAL: '#a1a1aa',
  ORDINAL: '#a1a1aa',
  QUANTITY: '#a1a1aa',
  PERCENT: '#a1a1aa',
  LANGUAGE: '#f0abfc',
}

export function entityColor(category: unknown): string {
  if (typeof category === 'string') {
    return ENTITY_CATEGORY_COLORS[category] ?? NER_LABEL_COLORS[category] ?? '#86efac'
  }
  return NODE_TYPE_COLORS.entity
}

export const EDGE_TYPE_STYLES: Record<string, { color: string; dash?: string; label: string }> = {
  CONTAINS: { color: '#38bdf8', label: 'Contains' },
  MENTIONS: { color: '#818cf8', label: 'Mentions' },
  RELATED_TO: { color: '#34d399', label: 'Related to' },
  CO_OCCURS_WITH: { color: '#64748b', dash: '3 4', label: 'Co-occurs with' },
  SUPPORTS: { color: '#f43f5e', label: 'Supports' },
  CONNECTS_TO: { color: '#f59e0b', dash: '6 3', label: 'Connects to' },
}

export const EXAMPLE_QUESTIONS = [
  'From which university did the 2024 Nobel Prize in Physics laureate from the University of Toronto receive his PhD?',
  'Which co-founder of DNNresearch later became chief scientist of OpenAI?',
  'Which company acquired the London AI lab whose chief executive shared the 2024 Nobel Prize in Chemistry?',
  'In which city was the AlphaGo match against Lee Sedol held?',
  'Who won the 2030 FIFA World Cup final?',
]

/** Recharts styling shared by all charts (dark theme). */
export const CHART = {
  grid: '#1e293b',
  axis: '#64748b',
  tick: { fill: '#94a3b8', fontSize: 11 },
  tooltip: {
    contentStyle: {
      background: '#0f172a',
      border: '1px solid #334155',
      borderRadius: 8,
      fontSize: 12,
      color: '#e2e8f0',
    },
    labelStyle: { color: '#cbd5e1', fontWeight: 600, marginBottom: 4 },
    itemStyle: { color: '#e2e8f0', padding: 0 },
    cursor: { fill: 'rgba(148,163,184,0.08)' },
  },
} as const
