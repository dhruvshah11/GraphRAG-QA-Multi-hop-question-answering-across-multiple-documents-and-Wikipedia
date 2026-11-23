"""Corpus snapshots and their construction.

A :class:`Corpus` bundles everything needed to answer questions over a set of
passages: chunks, embeddings (aligned with chunks), the FAISS store, the BM25
index, entity mentions/registry and the corpus evidence graph.

* The *persistent* corpus (uploaded documents) is managed by
  :class:`CorpusManager`; it is backed by SQLite + a FAISS file and rebuilt
  incrementally when documents are added or removed.
* *Ephemeral* corpora (Wikipedia articles fetched for one question, HotpotQA
  evaluation items) are built in memory by :func:`build_ephemeral_corpus`.
"""
from __future__ import annotations

import hashlib
import json
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np

from app.core.config import Settings
from app.core.domain import Document, EntityMention, Relation, TextChunk
from app.core.logging_setup import get_logger
from app.core.timing import StageTimer
from app.embeddings.embedder import EmbeddingGenerator
from app.entities.extractor import EntityExtractor
from app.entities.registry import EntityRegistry
from app.graph.builder import GraphBuilder
from app.graph.evidence_graph import EvidenceGraph
from app.preprocessing.chunker import Chunker
from app.retrieval.bm25 import BM25Index
from app.retrieval.vector_store import VectorStore

log = get_logger(__name__)


@dataclass
class Corpus:
    name: str
    chunks: list[TextChunk]
    embeddings: np.ndarray
    vector_ids: list[int]
    vector_store: VectorStore
    bm25: BM25Index
    mentions: list[EntityMention]
    relations: list[Relation]
    registry: EntityRegistry
    graph: EvidenceGraph
    documents: dict[str, dict[str, Any]]
    built_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))
    build_ms: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.pos_by_chunk = {c.id: i for i, c in enumerate(self.chunks)}
        self.pos_by_vid = {int(v): i for i, v in enumerate(self.vector_ids)}
        self.entity_ids_by_chunk: dict[str, set[str]] = defaultdict(set)
        for m in self.mentions:
            if m.entity_id:
                self.entity_ids_by_chunk[m.chunk_id].add(m.entity_id)
        self._neighbors: dict[str, list[str]] = defaultdict(list)
        by_doc: dict[str, list[TextChunk]] = defaultdict(list)
        for c in self.chunks:
            by_doc[c.document_id].append(c)
        for doc_chunks in by_doc.values():
            doc_chunks.sort(key=lambda c: c.chunk_index)
            for a, b in zip(doc_chunks, doc_chunks[1:]):
                self._neighbors[a.id].append(b.id)
                self._neighbors[b.id].append(a.id)

    def chunk(self, chunk_id: str) -> TextChunk:
        return self.chunks[self.pos_by_chunk[chunk_id]]

    def adjacent_chunks(self, chunk_id: str) -> list[str]:
        return self._neighbors.get(chunk_id, [])

    def embedding(self, chunk_id: str) -> np.ndarray:
        return self.embeddings[self.pos_by_chunk[chunk_id]]

    @property
    def is_empty(self) -> bool:
        return not self.chunks

    def stats(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "documents": len(self.documents),
            "chunks": len(self.chunks),
            "vectors": self.vector_store.size,
            "entities": len(self.registry.entities),
            "mentions": len(self.mentions),
            "relations": len(self.relations),
            "graph": self.graph.stats(),
            "built_at": self.built_at,
            "build_ms": {k: round(v, 1) for k, v in self.build_ms.items()},
        }


def assemble_corpus(
    name: str,
    chunks: list[TextChunk],
    embeddings: np.ndarray,
    vector_ids: list[int],
    vector_store: VectorStore,
    mentions: list[EntityMention],
    relations: list[Relation],
    documents: dict[str, dict[str, Any]],
    graph_builder: GraphBuilder,
    timer: Optional[StageTimer] = None,
    graph: Optional[EvidenceGraph] = None,
) -> Corpus:
    timer = timer or StageTimer()
    with timer.stage("entity_resolution"):
        registry = EntityRegistry.build(mentions, total_chunks=len(chunks))
    with timer.stage("bm25_index"):
        bm25 = BM25Index([c.text for c in chunks])
    with timer.stage("graph_construction"):
        if graph is None:
            graph = graph_builder.build_corpus_graph(chunks, documents, registry, mentions, relations)
    return Corpus(
        name=name, chunks=chunks, embeddings=embeddings, vector_ids=vector_ids, vector_store=vector_store,
        bm25=bm25, mentions=mentions, relations=relations, registry=registry, graph=graph, documents=documents,
        build_ms=dict(timer.durations_ms),
    )


def build_ephemeral_corpus(
    name: str,
    embedder: EmbeddingGenerator,
    extractor: EntityExtractor,
    graph_builder: GraphBuilder,
    chunks: list[TextChunk],
    documents: Optional[dict[str, dict[str, Any]]] = None,
    timer: Optional[StageTimer] = None,
    precomputed_embeddings: Optional[np.ndarray] = None,
    precomputed_mentions: Optional[tuple[list[EntityMention], list[Relation]]] = None,
) -> Corpus:
    """Build an in-memory corpus (Wikipedia per-question corpus, HotpotQA item)."""
    timer = timer or StageTimer()
    with timer.stage("embedding"):
        embeddings = (
            precomputed_embeddings if precomputed_embeddings is not None
            else embedder.encode_passages([c.text for c in chunks])
        )
    with timer.stage("entity_extraction"):
        mentions, relations = precomputed_mentions or extractor.extract_chunks(chunks)
    store = VectorStore(dim=embeddings.shape[1] if len(chunks) else (embedder.dim or 384), model_id=embedder.model_id)
    with timer.stage("faiss_index"):
        store.add(embeddings, list(range(len(chunks))))
    if documents is None:
        documents = {}
        for c in chunks:
            documents.setdefault(c.document_id, {"name": c.document_name, "source_type": c.source_type, "url": c.url,
                                                 "file_type": c.source_type, "num_chunks": 0})
            documents[c.document_id]["num_chunks"] += 1
    return assemble_corpus(name, chunks, embeddings, list(range(len(chunks))), store, mentions, relations,
                           documents, graph_builder, timer)


class CorpusManager:
    """Owns the persistent corpus of uploaded documents (thread-safe snapshots)."""

    def __init__(self, settings: Settings, repo: Any, embedder: EmbeddingGenerator, extractor: EntityExtractor,
                 graph_builder: GraphBuilder) -> None:
        self.settings = settings
        self.repo = repo
        self.embedder = embedder
        self.extractor = extractor
        self.graph_builder = graph_builder
        self._lock = threading.RLock()
        self._corpus: Optional[Corpus] = None
        self._store: Optional[VectorStore] = None
        self.index_path = settings.indexes_dir / "faiss.index"
        self.graph_path = settings.graphs_dir / "corpus_graph.json"
        self.graphml_path = settings.graphs_dir / "corpus_graph.graphml"
        self.last_refresh_ms: dict[str, float] = {}

    # --------------------------------------------------------------- loading
    @property
    def store(self) -> VectorStore:
        if self._store is None:
            with self._lock:
                if self._store is None:
                    self.embedder.load()
                    store = VectorStore(self.embedder.dim or 384, self.embedder.model_id, self.index_path)
                    loaded = store.load()
                    self._store = store
                    expected = len(self.repo.get_chunks_with_vector_ids(only_processed=True))
                    if not loaded or store.size != expected:
                        log.info("FAISS index missing/out of date (%s vs %s vectors): rebuilding", store.size, expected)
                        self.rebuild_index()
        return self._store

    @property
    def corpus(self) -> Corpus:
        if self._corpus is None:
            self.refresh()
        assert self._corpus is not None
        return self._corpus

    def _fingerprint(self, chunks: list[TextChunk], mentions: list[EntityMention], relations: list[Relation]) -> str:
        h = hashlib.sha1()
        for c in chunks:
            h.update(c.id.encode())
        h.update(f"{len(mentions)}|{len(relations)}|{self.graph_builder.max_df_ratio}|{self.graph_builder.max_df_abs}".encode())
        h.update(self.embedder.model_id.encode())
        return h.hexdigest()

    def refresh(self) -> Corpus:
        """Rebuild the in-memory snapshot (registry, BM25, graph) from storage."""
        with self._lock:
            timer = StageTimer()
            store = self.store
            with timer.stage("load_chunks"):
                pairs = self.repo.get_chunks_with_vector_ids(only_processed=True)
                chunks = [c for c, _ in pairs]
                vids = [v for _, v in pairs]
                dim = store.dim
                embeddings = (
                    np.stack([store.reconstruct(v) for v in vids]).astype(np.float32)
                    if vids else np.zeros((0, dim), dtype=np.float32)
                )
                mentions = self.repo.get_mentions(only_processed=True)
                relations = self.repo.get_relations(only_processed=True)
                documents = {
                    d["id"]: {k: d[k] for k in ("name", "file_type", "source_type", "url", "num_chunks", "num_pages")}
                    for d in self.repo.list_documents(source_type="upload") if d["status"] == "processed"
                }
            fingerprint = self._fingerprint(chunks, mentions, relations)
            cached_graph = None
            if self.graph_path.exists():
                try:
                    payload = json.loads(self.graph_path.read_text(encoding="utf-8"))
                    if payload.get("fingerprint") == fingerprint:
                        cached_graph = EvidenceGraph.from_dict(payload["graph"])
                except Exception:  # corrupted file -> rebuild
                    cached_graph = None
            corpus = assemble_corpus("uploads", chunks, embeddings, vids, store, mentions, relations, documents,
                                     self.graph_builder, timer, graph=cached_graph)
            if cached_graph is None:
                with timer.stage("graph_serialization"):
                    self.graph_path.parent.mkdir(parents=True, exist_ok=True)
                    self.graph_path.write_text(
                        json.dumps({"fingerprint": fingerprint, "graph": corpus.graph.to_dict()}), encoding="utf-8"
                    )
                    corpus.graph.save_graphml(self.graphml_path)
            self.repo.replace_entities(corpus.registry.entities.values())
            corpus.build_ms = dict(timer.durations_ms)
            self.last_refresh_ms = dict(timer.durations_ms)
            self._corpus = corpus
            log.info("Corpus refreshed: %d chunks, %d entities, graph %d nodes / %d edges (%.0f ms)",
                     len(chunks), len(corpus.registry.entities), corpus.graph.num_nodes, corpus.graph.num_edges,
                     timer.total_ms)
            return corpus

    # ------------------------------------------------------------- mutations
    def index_document(self, doc: Document, chunks: list[TextChunk], timer: StageTimer) -> dict[str, Any]:
        """Embed + NER a processed document and add it to the persistent stores."""
        with timer.stage("embedding"):
            vectors = self.embedder.encode_passages([c.text for c in chunks])
        with timer.stage("entity_extraction"):
            mentions, relations = self.extractor.extract_chunks(chunks)
        with self._lock:
            store = self.store
            with timer.stage("faiss_update"):
                old_ids = self.repo.get_vector_ids(doc.id)
                store.remove(old_ids)
                start = self.repo.next_vector_id()
                vids = list(range(start, start + len(chunks)))
                self.repo.replace_chunks(doc.id, chunks, vids)
                store.add(vectors, vids)
                store.save()
            with timer.stage("store_entities"):
                self.repo.replace_mentions(doc.id, mentions, relations)
        return {
            "chunks": len(chunks),
            "mentions": len(mentions),
            "relations": len(relations),
            "distinct_entities": len({m.normalized for m in mentions}),
        }

    def remove_document(self, doc_id: str) -> None:
        with self._lock:
            ids = self.repo.get_vector_ids(doc_id)
            self.store.remove(ids)
            self.store.save()

    def rebuild_index(self) -> int:
        """Re-embed every processed chunk and rebuild FAISS from scratch."""
        with self._lock:
            self.embedder.load()
            if self._store is None or self._store.dim != self.embedder.dim:
                self._store = VectorStore(self.embedder.dim or 384, self.embedder.model_id, self.index_path)
            self._store.reset()
            pairs = self.repo.get_chunks_with_vector_ids(only_processed=True)
            if pairs:
                vectors = self.embedder.encode_passages([c.text for c, _ in pairs])
                self._store.add(vectors, [v for _, v in pairs])
            self._store.save()
            self._corpus = None
            return len(pairs)
