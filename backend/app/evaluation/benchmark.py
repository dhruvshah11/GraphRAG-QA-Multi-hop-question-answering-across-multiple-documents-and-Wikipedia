"""Reproducible stage-wise benchmark of the whole pipeline.

Every number is measured on this machine at run time:

1 ingestion  2 chunking  3 embedding  4 entity extraction  5 corpus graph construction
6 query embedding  7 semantic (FAISS) retrieval  8 BM25 retrieval  9 hybrid retrieval
10 question-graph construction  11 graph traversal  12 GNN inference  13 answer generation

Latency = wall clock (time.perf_counter); memory = process RSS (psutil) before/after a stage.
"""
from __future__ import annotations

import os
import platform
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional

import psutil

from app.core.domain import Document
from app.core.logging_setup import get_logger
from app.core.timing import StageTimer
from app.ingestion.document_processor import build_document_text
from app.ingestion.extractors import extract_pages
from app.retrieval.corpus import build_ephemeral_corpus
from app.retrieval.retrievers import BM25Retriever, HybridRetriever, QueryScores, SemanticRetriever

if TYPE_CHECKING:  # pragma: no cover
    from app.core.container import Container

log = get_logger(__name__)


def _rss_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / 1048576


def _stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    ordered = sorted(values)
    return {"mean_ms": round(statistics.mean(values), 3), "median_ms": round(statistics.median(values), 3),
            "p95_ms": round(ordered[min(len(ordered) - 1, int(0.95 * (len(ordered) - 1) + 0.5))], 3),
            "min_ms": round(ordered[0], 3), "max_ms": round(ordered[-1], 3), "n": len(values)}


class BenchmarkRunner:
    def __init__(self, container: "Container") -> None:
        self.c = container

    def measure(self, fn: Callable[[], Any]) -> tuple[Any, float, float]:
        before = _rss_mb()
        start = time.perf_counter()
        out = fn()
        ms = (time.perf_counter() - start) * 1000
        return out, ms, round(_rss_mb() - before, 2)

    def run(self, questions: list[str], repeats: int = 3) -> dict[str, Any]:
        c = self.c
        stages: dict[str, Any] = {}
        docs_dir = c.settings.samples_dir / "documents"
        files = sorted(p for p in docs_dir.iterdir() if p.suffix.lower() in c.settings.allowed_extensions)

        # 1 ingestion ------------------------------------------------------------
        documents: list[Document] = []
        per_file = []
        for path in files:
            (pages_text), ms, mem = self.measure(lambda p=path: build_document_text(extract_pages(p)))
            text, pages, sections = pages_text
            documents.append(Document(id="bench_" + path.stem, name=path.name, file_type=path.suffix[1:],
                                      text=text, pages=pages, sections=sections, num_pages=len(pages)))
            per_file.append({"file": path.name, "type": path.suffix[1:], "bytes": path.stat().st_size,
                             "pages": len(pages), "chars": len(text), "ms": round(ms, 2), "rss_delta_mb": mem})
        stages["1_ingestion"] = {"files": per_file, "total_ms": round(sum(f["ms"] for f in per_file), 2)}

        # 2 chunking -------------------------------------------------------------
        chunker = c.make_chunker()
        chunks, ms, mem = self.measure(lambda: [ch for d in documents for ch in chunker.chunk_document(d)])
        stages["2_chunking"] = {"ms": round(ms, 2), "chunks": len(chunks), "rss_delta_mb": mem,
                                "avg_tokens": round(statistics.mean(ch.token_count for ch in chunks), 1),
                                "chunk_size": chunker.chunk_size, "chunk_overlap": chunker.chunk_overlap}

        # 3 embedding ------------------------------------------------------------
        c.embedder.load()
        emb, ms, mem = self.measure(lambda: c.embedder.encode_passages([ch.text for ch in chunks]))
        stages["3_embedding"] = {"ms": round(ms, 2), "passages": len(chunks), "dim": int(emb.shape[1]),
                                 "ms_per_passage": round(ms / max(1, len(chunks)), 2), "rss_delta_mb": mem,
                                 "model": c.embedder.model_id}

        # 4 entity extraction ------------------------------------------------------
        c.extractor.load()
        (mentions, relations), ms, mem = self.measure(lambda: c.extractor.extract_chunks(chunks))
        stages["4_entity_extraction"] = {"ms": round(ms, 2), "mentions": len(mentions), "relations": len(relations),
                                         "ms_per_passage": round(ms / max(1, len(chunks)), 2), "rss_delta_mb": mem}

        # 5 corpus graph construction ---------------------------------------------
        timer = StageTimer()
        corpus, ms, mem = self.measure(lambda: build_ephemeral_corpus(
            "benchmark", c.embedder, c.extractor, c.graph_builder, chunks, timer=timer,
            precomputed_embeddings=emb, precomputed_mentions=(mentions, relations)))
        gstats = corpus.graph.stats()
        stages["5_graph_construction"] = {"ms": round(timer.durations_ms.get("graph_construction", 0), 2),
                                          "entity_resolution_ms": round(timer.durations_ms.get("entity_resolution", 0), 2),
                                          "faiss_index_ms": round(timer.durations_ms.get("faiss_index", 0), 3),
                                          "bm25_index_ms": round(timer.durations_ms.get("bm25_index", 0), 3),
                                          "nodes": gstats["nodes"], "edges": gstats["edges"],
                                          "node_types": gstats["node_types"], "edge_types": gstats["edge_types"],
                                          "rss_delta_mb": mem}

        # 6-12 per-question stages --------------------------------------------------
        rt = c.runtime()
        lat: dict[str, list[float]] = {k: [] for k in (
            "query_embedding", "semantic_retrieval", "bm25_retrieval", "hybrid_retrieval", "question_graph_construction",
            "graph_traversal", "graph_heuristic_ranking", "gnn_inference")}
        gen_lat: dict[str, list[float]] = {}
        sub_nodes, sub_edges, chain_lens, candidates = [], [], [], []
        for _ in range(repeats):
            for q in questions:
                c.embedder._query_cache.clear()  # measure the uncached path
                t = time.perf_counter()
                scores = QueryScores(corpus, c.embedder, q, rt.hybrid_alpha)
                lat["query_embedding"].append((time.perf_counter() - t) * 1000)
                for name, cls in (("semantic_retrieval", SemanticRetriever), ("bm25_retrieval", BM25Retriever),
                                  ("hybrid_retrieval", HybridRetriever)):
                    t = time.perf_counter()
                    cls(corpus, c.embedder, rt.hybrid_alpha).search(q, rt.top_k, scores)
                    lat[name].append((time.perf_counter() - t) * 1000)
                timer = StageTimer()
                res = c.reasoner.analyze(corpus, q, top_k=rt.top_k, max_hops=rt.max_hops, alpha=rt.hybrid_alpha,
                                         timer=timer)
                lat["question_graph_construction"].append(timer.durations_ms["graph_construction"])
                lat["graph_traversal"].append(timer.durations_ms["graph_traversal"])
                lat["graph_heuristic_ranking"].append(timer.durations_ms["graph_ranking"])
                if "gnn_inference" in timer.durations_ms:
                    lat["gnn_inference"].append(timer.durations_ms["gnn_inference"])
                st = res.qgraph.graph.stats()
                sub_nodes.append(st["nodes"])
                sub_edges.append(st["edges"])
                candidates.append(len(res.qgraph.passages))
                if res.chains:
                    chain_lens.append(res.chains[0].length)
                strategy = "gnn" if "gnn" in res.rankings else "graph"
                for backend in ("extractive", "hf_reader", "hf_generator"):
                    if not c.generator.backends[backend].available():
                        continue
                    t = time.perf_counter()
                    c.pipeline.answer_from_result(q, corpus, res, strategy, backend, rt.max_evidence_passages,
                                                  StageTimer())
                    gen_lat.setdefault(backend, []).append((time.perf_counter() - t) * 1000)
        for i, key in enumerate(["query_embedding", "semantic_retrieval", "bm25_retrieval", "hybrid_retrieval"], start=6):
            stages[f"{i}_{key}"] = _stats(lat[key])
        stages["7_semantic_retrieval"]["note"] = "FAISS search only; query embedding measured separately"
        stages["10_question_graph_construction"] = {**_stats(lat["question_graph_construction"]),
                                                   "avg_nodes": round(statistics.mean(sub_nodes), 1),
                                                   "avg_edges": round(statistics.mean(sub_edges), 1),
                                                   "avg_candidate_passages": round(statistics.mean(candidates), 1)}
        stages["11_graph_traversal"] = {**_stats(lat["graph_traversal"]),
                                        "avg_top_chain_length": round(statistics.mean(chain_lens), 2) if chain_lens else None,
                                        "heuristic_ranking": _stats(lat["graph_heuristic_ranking"])}
        stages["12_gnn_inference"] = _stats(lat["gnn_inference"]) or {
            "note": "not measured: no trained GNN checkpoint"}
        stages["13_answer_generation"] = {b: _stats(v) for b, v in gen_lat.items()}
        return {
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "environment": {"python": platform.python_version(), "platform": platform.platform(),
                            "processor": platform.processor(), "cpu_count": os.cpu_count(),
                            "ram_gb": round(psutil.virtual_memory().total / 1073741824, 1), "device": c.settings.device,
                            "rss_mb_end": round(_rss_mb(), 1)},
            "config": {**rt.model_dump(), "questions": len(questions), "repeats": repeats,
                       "gnn_available": c.gnn_ranker.is_available},
            "stages": stages,
        }
