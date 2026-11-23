"""Embeddings, FAISS vector store, BM25 and hybrid retrieval."""
from __future__ import annotations

import numpy as np

from app.embeddings.embedder import HashingEmbedder
from app.retrieval.bm25 import BM25Index
from app.retrieval.retrievers import BM25Retriever, HybridRetriever, QueryScores, SemanticRetriever
from app.retrieval.vector_store import VectorStore


def test_embeddings_are_normalized_and_semantic(container):
    e = container.embedder
    vecs = e.encode(["A cat sits on the mat.", "A kitten rests on a rug.", "Stock markets fell sharply."])
    assert vecs.shape == (3, 384)
    assert np.allclose(np.linalg.norm(vecs, axis=1), 1.0, atol=1e-4)
    assert vecs[0] @ vecs[1] > vecs[0] @ vecs[2]


def test_long_passages_use_window_averaging(container):
    e = container.embedder
    long_text = " ".join(f"Sentence {i} is about neural networks and training." for i in range(80))
    assert len(e._windows(long_text)) > 1
    v = e.encode_passages([long_text])
    assert v.shape == (1, 384) and abs(np.linalg.norm(v) - 1) < 1e-4


def test_hashing_fallback_is_deterministic():
    h = HashingEmbedder(64)
    a, b = h.encode(["graph neural networks"]), h.encode(["graph neural networks"])
    assert np.array_equal(a, b) and abs(np.linalg.norm(a) - 1) < 1e-6


def test_faiss_insert_search_remove_persist(tmp_path):
    rng = np.random.default_rng(0)
    vecs = rng.normal(size=(20, 8)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    store = VectorStore(8, "test-model", tmp_path / "idx.index")
    store.add(vecs, list(range(100, 120)))
    scores, ids = store.search(vecs[3], 3)
    assert ids[0] == 103 and abs(scores[0] - 1.0) < 1e-5
    assert store.remove([103]) == 1
    _, ids = store.search(vecs[3], 3)
    assert 103 not in ids
    store.save()
    reloaded = VectorStore(8, "test-model", tmp_path / "idx.index")
    assert reloaded.load() and reloaded.size == 19
    assert np.allclose(reloaded.reconstruct(105), vecs[5], atol=1e-6)
    # an index built with another embedding model is rejected (forces a rebuild)
    assert not VectorStore(8, "other-model", tmp_path / "idx.index").load()


def test_bm25_ranks_lexical_match_first():
    idx = BM25Index(["the cat sat on the mat", "dogs bark loudly at night", "a cat and a dog play"])
    raw = idx.scores("which cat sat?")
    assert int(np.argmax(raw)) == 0
    norm = idx.normalize(raw)
    assert norm.max() == 1.0 and norm.min() >= 0.0


def test_hybrid_formula_and_retrievers(container, demo_corpus):
    q = "In which city was the AlphaGo match against Lee Sedol held?"
    alpha = 0.7
    s = QueryScores(demo_corpus, container.embedder, q, alpha)
    assert np.allclose(s.hybrid, alpha * s.semantic + (1 - alpha) * s.bm25)
    gold = next(c.id for c in demo_corpus.chunks if "Lee Sedol" in c.text)
    for cls in (SemanticRetriever, BM25Retriever, HybridRetriever):
        results = cls(demo_corpus, container.embedder, alpha).search(q, 3, s)
        assert results[0].chunk.id == gold, cls.__name__
        assert [r.rank for r in results] == list(range(1, len(results) + 1))
        assert all(r.semantic_score >= 0 and 0 <= r.bm25_score <= 1 for r in results)
