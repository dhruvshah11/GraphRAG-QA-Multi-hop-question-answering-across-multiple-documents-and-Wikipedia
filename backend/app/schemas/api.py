"""Pydantic request/response schemas of the REST API (also drive /docs)."""
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

Strategy = Literal["faiss", "bm25", "hybrid", "graph", "gnn"]
Mode = Literal["multi_document", "open_domain"]
Backend = Literal["auto", "extractive", "hf_reader", "hf_generator", "ollama", "anthropic"]


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000, examples=[
        "From which university did the 2024 Nobel Prize in Physics laureate from the University of Toronto "
        "receive his PhD?"])
    mode: Mode = "multi_document"
    strategy: Strategy = "gnn"
    top_k: Optional[int] = Field(default=None, ge=1, le=50)
    max_hops: Optional[int] = Field(default=None, ge=1, le=4)
    backend: Optional[Backend] = None


class OpenDomainRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000, examples=[
        "Which university did the director of the film Inception study at?"])
    strategy: Strategy = "gnn"
    top_k: Optional[int] = Field(default=None, ge=1, le=50)
    max_hops: Optional[int] = Field(default=None, ge=1, le=4)
    backend: Optional[Backend] = None


class CitationOut(BaseModel):
    index: int
    chunk_id: str
    document_id: str
    document_name: str
    page_number: Optional[int]
    page_end: Optional[int]
    section: Optional[str]
    url: Optional[str]
    source_type: str
    passage_text: str
    supporting_sentence: str
    sentence_start: int
    sentence_end: int
    retrieval_score: float
    graph_score: Optional[float]
    gnn_score: Optional[float]
    support_score: float
    claim: str


class ReasoningStepOut(BaseModel):
    step: int
    kind: Literal["fact", "inference"]
    text: str
    citations: list[int]


class QAResponse(BaseModel):
    """Structured answer. Extra diagnostic fields (score tables, timings) are included as well."""

    model_config = ConfigDict(extra="allow")

    query_id: str
    question: str
    mode: Mode
    strategy: Strategy
    effective_strategy: Strategy
    answer: str = Field(description="Final answer, or the insufficient-evidence message")
    candidate_answer: Optional[str] = Field(default=None, description="Generator output before the confidence check")
    answer_sentence: Optional[str] = Field(default=None, description="Evidence sentence containing the answer")
    answer_citations: list[int] = Field(default_factory=list)
    reasoning_summary: str
    reasoning_steps: list[ReasoningStepOut]
    evidence_chain: list[dict[str, Any]] = Field(description="Ordered chain steps with bridge entities")
    chains: list[dict[str, Any]] = Field(default_factory=list, description="Top scored evidence chains")
    citations: list[CitationOut]
    evidence_passages: list[dict[str, Any]] = Field(default_factory=list)
    retrieved_passages: list[dict[str, Any]]
    strategy_rankings: dict[str, list[str]] = Field(default_factory=dict)
    score_table: list[dict[str, Any]] = Field(default_factory=list,
                                              description="Per-passage scores and ranks for every strategy")
    confidence: float
    confidence_components: dict[str, float] = Field(default_factory=dict)
    evidence_coverage: dict[str, Any] = Field(default_factory=dict)
    insufficient_evidence: bool
    insufficient_reasons: list[str]
    question_entities: list[dict[str, Any]] = Field(default_factory=list)
    generator_backend: str
    generator_warnings: list[str] = Field(default_factory=list)
    removed_citations: list[str] = Field(default_factory=list)
    gnn_used: bool = False
    notes: list[str] = Field(default_factory=list)
    graph_stats: dict[str, Any] = Field(default_factory=dict)
    expansion_log: list[dict[str, Any]] = Field(default_factory=list)
    timings_ms: dict[str, float]
    total_ms: float = 0.0
    top_k: int
    max_hops: int
    corpus: dict[str, Any] = Field(default_factory=dict)
    open_domain: Optional[dict[str, Any]] = Field(default=None,
                                                  description="Wikipedia searches, articles and bridge entities")


class GraphQueryRequest(BaseModel):
    query_id: Optional[str] = Field(default=None, description="Load the saved subgraph of an answered question")
    question: Optional[str] = Field(default=None, description="Or build a subgraph for a new question")
    view: Literal["subgraph", "chain"] = "subgraph"
    top_k: Optional[int] = Field(default=None, ge=1, le=50)
    max_hops: Optional[int] = Field(default=None, ge=1, le=4)


class EvaluationRequest(BaseModel):
    dataset: Literal["demo", "hotpotqa", "wikipedia"] = "demo"
    strategies: list[Strategy] = Field(default_factory=lambda: ["faiss", "bm25", "hybrid", "graph", "gnn"])
    top_k: Optional[int] = Field(default=None, ge=1, le=50)
    max_hops: Optional[int] = Field(default=None, ge=1, le=4)
    limit: Optional[int] = Field(default=None, ge=1, le=5000)
    k: int = Field(default=2, ge=1, le=20, description="Cut-off for Precision@K / Recall@K")
    backend: Optional[Backend] = None


class TrainRequest(BaseModel):
    source: Literal["hotpotqa", "demo"] = "hotpotqa"
    epochs: int = Field(default=40, ge=1, le=500)
    train_limit: Optional[int] = Field(default=None, ge=10, le=20000)
    val_limit: Optional[int] = Field(default=None, ge=5, le=5000)
    lr: float = Field(default=1e-3, gt=0, le=0.1)
    seed: int = 42


class SettingsUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_size: Optional[int] = None
    chunk_overlap: Optional[int] = None
    top_k: Optional[int] = None
    max_hops: Optional[int] = None
    hybrid_alpha: Optional[float] = None
    generator_backend: Optional[Backend] = None
    max_evidence_passages: Optional[int] = None
    embedding_model: Optional[str] = None


class JobOut(BaseModel):
    id: str
    kind: str
    status: str
    progress: float
    message: Optional[str] = None
    result: Optional[Any] = None
    created_at: str
    updated_at: str


class ErrorOut(BaseModel):
    detail: str
