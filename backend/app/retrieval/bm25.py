"""Lexical BM25 index (Okapi BM25 via ``rank_bm25``)."""
from __future__ import annotations

import numpy as np
from rank_bm25 import BM25Okapi

from app.preprocessing.tokenize import content_terms


class BM25Index:
    """BM25 over passage texts.

    score(q, d) = sum_t IDF(t) * f(t,d)(k1+1) / (f(t,d) + k1(1 - b + b|d|/avgdl))
    with k1 = 1.5 and b = 0.75 (rank_bm25 defaults).
    """

    def __init__(self, texts: list[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.size = len(texts)
        self._tokenized = [content_terms(t) for t in texts]
        # rank_bm25 cannot handle an empty corpus or all-empty documents
        self._bm25 = BM25Okapi([toks or ["__empty__"] for toks in self._tokenized], k1=k1, b=b) if texts else None

    def scores(self, query: str) -> np.ndarray:
        """Raw BM25 scores for every passage (aligned with the input order)."""
        if self._bm25 is None:
            return np.zeros(0, dtype=np.float32)
        terms = content_terms(query, drop_question_words=True) or content_terms(query)
        if not terms:
            return np.zeros(self.size, dtype=np.float32)
        raw = np.asarray(self._bm25.get_scores(terms), dtype=np.float32)
        return np.maximum(raw, 0.0)

    @staticmethod
    def normalize(raw: np.ndarray) -> np.ndarray:
        """Max-normalization to [0, 1] (the best lexical match gets 1)."""
        if raw.size == 0:
            return raw
        top = float(raw.max())
        return raw / top if top > 0 else np.zeros_like(raw)
