"""Retrievers: semantic (FAISS), lexical (BM25) and hybrid.

All three implement :class:`Retriever` and return :class:`RetrievedPassage`
objects that carry *every* score (semantic, BM25, hybrid) so that the UI and
the evaluation can compare strategies on identical candidates.

Hybrid score (min-max normalized BM25):

    hybrid(p) = alpha * max(0, cos(q, p)) + (1 - alpha) * bm25(q, p) / max_p' bm25(q, p')
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

from app.core.domain import RetrievedPassage

if TYPE_CHECKING:  # pragma: no cover
    from app.embeddings.embedder import EmbeddingGenerator
    from app.retrieval.corpus import Corpus


class QueryScores:
    """All per-passage scores for one question over one corpus (computed once)."""

    def __init__(self, corpus: "Corpus", embedder: "EmbeddingGenerator", question: str, alpha: float) -> None:
        self.corpus = corpus
        self.question = question
        self.alpha = alpha
        self.query_vec = embedder.encode_query(question)
        n = len(corpus.chunks)
        self.semantic = (
            np.clip(corpus.embeddings @ self.query_vec, 0.0, 1.0) if n else np.zeros(0, dtype=np.float32)
        )
        self.bm25_raw = corpus.bm25.scores(question) if n else np.zeros(0, dtype=np.float32)
        self.bm25 = corpus.bm25.normalize(self.bm25_raw)
        self.hybrid = alpha * self.semantic + (1.0 - alpha) * self.bm25

    def passage(self, pos: int, origin: str = "retrieval", hop: int = 0) -> RetrievedPassage:
        return RetrievedPassage(
            chunk=self.corpus.chunks[pos],
            semantic_score=float(self.semantic[pos]),
            bm25_raw=float(self.bm25_raw[pos]),
            bm25_score=float(self.bm25[pos]),
            hybrid_score=float(self.hybrid[pos]),
            origin=origin,
            hop=hop,
        )

    def passage_by_id(self, chunk_id: str, origin: str = "graph_expansion", hop: int = 0) -> RetrievedPassage:
        return self.passage(self.corpus.pos_by_chunk[chunk_id], origin=origin, hop=hop)


class Retriever(ABC):
    """Common interface of every first-stage retriever."""

    name: str = "base"

    def __init__(self, corpus: "Corpus", embedder: "EmbeddingGenerator", alpha: float = 0.7) -> None:
        self.corpus = corpus
        self.embedder = embedder
        self.alpha = alpha

    def scores(self, question: str) -> QueryScores:
        return QueryScores(self.corpus, self.embedder, question, self.alpha)

    def search(self, question: str, top_k: int, scores: QueryScores | None = None) -> list[RetrievedPassage]:
        if not self.corpus.chunks:
            return []
        scores = scores or self.scores(question)
        positions = self._rank(question, top_k, scores)
        results = []
        for rank, pos in enumerate(positions[:top_k], start=1):
            rp = scores.passage(int(pos))
            rp.rank = rank
            rp.final_score = self._final(rp)
            results.append(rp)
        return results

    @abstractmethod
    def _rank(self, question: str, top_k: int, scores: QueryScores) -> list[int]:
        """Return corpus positions in ranked order."""

    @abstractmethod
    def _final(self, rp: RetrievedPassage) -> float:
        """The score this retriever ranks by."""


class SemanticRetriever(Retriever):
    """Dense retrieval with FAISS (cosine similarity of normalized embeddings)."""

    name = "faiss"

    def _rank(self, question: str, top_k: int, scores: QueryScores) -> list[int]:
        _, ids = self.corpus.vector_store.search(scores.query_vec, top_k)
        return [self.corpus.pos_by_vid[int(i)] for i in ids if int(i) in self.corpus.pos_by_vid]

    def _final(self, rp: RetrievedPassage) -> float:
        return rp.semantic_score


class BM25Retriever(Retriever):
    """Lexical retrieval with Okapi BM25."""

    name = "bm25"

    def _rank(self, question: str, top_k: int, scores: QueryScores) -> list[int]:
        order = np.argsort(-scores.bm25_raw, kind="stable")
        return [int(i) for i in order[:top_k] if scores.bm25_raw[i] > 0]

    def _final(self, rp: RetrievedPassage) -> float:
        return rp.bm25_score


class HybridRetriever(Retriever):
    """Weighted fusion of FAISS and BM25 candidates."""

    name = "hybrid"

    def _rank(self, question: str, top_k: int, scores: QueryScores) -> list[int]:
        n_candidates = max(50, 5 * top_k)
        _, ids = self.corpus.vector_store.search(scores.query_vec, n_candidates)
        candidates = {self.corpus.pos_by_vid[int(i)] for i in ids if int(i) in self.corpus.pos_by_vid}
        lexical = np.argsort(-scores.bm25_raw, kind="stable")[:n_candidates]
        candidates |= {int(i) for i in lexical if scores.bm25_raw[i] > 0}
        return sorted(candidates, key=lambda p: (-scores.hybrid[p], p))

    def _final(self, rp: RetrievedPassage) -> float:
        return rp.hybrid_score


RETRIEVERS = {"faiss": SemanticRetriever, "bm25": BM25Retriever, "hybrid": HybridRetriever}
