"""Domain model shared by every module of the pipeline.

These plain dataclasses are the objects that flow through the system:

    Document -> PageText -> TextChunk -> EntityMention/Entity/Relation
    question -> RetrievedPassage -> EvidenceChain -> Citation -> AnswerResult

Keeping them free of framework code makes the pipeline easy to test and to
explain (each class appears in docs/diagrams/class_diagram.puml).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional


@dataclass
class PageText:
    """Normalized text of one page (PDF) or one logical page (TXT/DOCX)."""

    page_number: int  # 1-based
    text: str
    char_offset: int  # position of the page inside the full document text


@dataclass
class Section:
    """A titled region of the document text (DOCX headings, Wikipedia sections)."""

    title: Optional[str]
    char_start: int
    char_end: int


@dataclass
class Document:
    id: str
    name: str
    file_type: str  # pdf | txt | docx | wikipedia
    source_type: str = "upload"  # upload | wikipedia
    path: Optional[str] = None
    url: Optional[str] = None
    size_bytes: int = 0
    num_pages: int = 0
    num_chunks: int = 0
    num_entities: int = 0
    status: str = "uploaded"  # uploaded | processing | processed | failed
    error: Optional[str] = None
    checksum: Optional[str] = None
    created_at: Optional[str] = None
    processed_at: Optional[str] = None
    text: str = ""
    pages: list[PageText] = field(default_factory=list)
    sections: list[Section] = field(default_factory=list)

    def page_for_offset(self, offset: int) -> int:
        """Return the 1-based page number that contains ``offset``."""
        page = 1
        for p in self.pages:
            if p.char_offset <= offset:
                page = p.page_number
            else:
                break
        return page


@dataclass
class TextChunk:
    """A passage: the unit of retrieval, graph construction and citation."""

    id: str
    document_id: str
    document_name: str
    chunk_index: int
    text: str
    page_number: Optional[int]
    page_end: Optional[int]
    char_start: int
    char_end: int
    token_count: int
    section: Optional[str] = None
    url: Optional[str] = None
    source_type: str = "upload"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EntityMention:
    """One occurrence of a named entity inside a chunk."""

    text: str  # original surface form
    label: str  # spaCy label (PERSON, ORG, GPE, ...)
    normalized: str
    chunk_id: str
    document_id: str
    char_start: int  # offsets relative to the chunk text
    char_end: int
    sentence_index: int
    entity_id: Optional[str] = None  # filled by the EntityRegistry


@dataclass
class Relation:
    """A verb-mediated relation between two entity mentions in one sentence."""

    source_text: str
    target_text: str
    source_normalized: str
    target_normalized: str
    relation: str  # verb lemma phrase, e.g. "win", "work at"
    chunk_id: str
    document_id: str
    sentence: str
    source_id: Optional[str] = None
    target_id: Optional[str] = None


@dataclass
class Entity:
    """A canonical entity after duplicate resolution."""

    id: str
    name: str
    label: str
    category: str
    normalized: str
    aliases: set[str] = field(default_factory=set)
    mention_count: int = 0
    chunk_ids: set[str] = field(default_factory=set)
    document_ids: set[str] = field(default_factory=set)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "label": self.label,
            "category": self.category,
            "normalized": self.normalized,
            "aliases": sorted(self.aliases),
            "mention_count": self.mention_count,
            "num_chunks": len(self.chunk_ids),
            "num_documents": len(self.document_ids),
        }


@dataclass
class RetrievedPassage:
    """A passage together with every score computed for the current question."""

    chunk: TextChunk
    semantic_score: float = 0.0
    bm25_raw: float = 0.0
    bm25_score: float = 0.0  # normalized to [0, 1]
    hybrid_score: float = 0.0
    graph_score: Optional[float] = None
    gnn_score: Optional[float] = None
    final_score: float = 0.0
    rank: int = 0
    origin: str = "retrieval"  # retrieval | graph_expansion
    hop: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk.id,
            "document_id": self.chunk.document_id,
            "document_name": self.chunk.document_name,
            "page_number": self.chunk.page_number,
            "page_end": self.chunk.page_end,
            "section": self.chunk.section,
            "url": self.chunk.url,
            "text": self.chunk.text,
            "char_start": self.chunk.char_start,
            "char_end": self.chunk.char_end,
            "semantic_score": round(self.semantic_score, 4),
            "bm25_raw": round(self.bm25_raw, 4),
            "bm25_score": round(self.bm25_score, 4),
            "hybrid_score": round(self.hybrid_score, 4),
            "graph_score": None if self.graph_score is None else round(self.graph_score, 4),
            "gnn_score": None if self.gnn_score is None else round(self.gnn_score, 4),
            "final_score": round(self.final_score, 4),
            "rank": self.rank,
            "origin": self.origin,
            "hop": self.hop,
        }


@dataclass
class ChainStep:
    chunk_id: str
    via_entity: Optional[str]  # entity id used to reach this passage (None for the first step)
    via_entity_name: Optional[str]
    relevance: float
    reason: str


@dataclass
class EvidenceChain:
    """An ordered path of passages connected by bridge entities."""

    steps: list[ChainStep]
    score: float
    components: dict[str, float]
    covered_terms: list[str]
    missing_terms: list[str]

    @property
    def chunk_ids(self) -> list[str]:
        return [s.chunk_id for s in self.steps]

    @property
    def length(self) -> int:
        return len(self.steps)

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_ids": self.chunk_ids,
            "steps": [asdict(s) for s in self.steps],
            "score": round(self.score, 4),
            "components": {k: round(v, 4) for k, v in self.components.items()},
            "covered_terms": self.covered_terms,
            "missing_terms": self.missing_terms,
            "length": self.length,
        }


@dataclass
class Citation:
    """A pointer from a claim in the answer to the exact supporting text."""

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
    sentence_start: int  # offsets of the supporting sentence inside passage_text
    sentence_end: int
    retrieval_score: float
    graph_score: Optional[float]
    gnn_score: Optional[float]
    support_score: float
    claim: str

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        for key in ("retrieval_score", "graph_score", "gnn_score", "support_score"):
            if d[key] is not None:
                d[key] = round(float(d[key]), 4)
        return d
