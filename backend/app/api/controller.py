"""APIController: the logic behind every REST endpoint.

Routers in ``app/api/routes_*.py`` stay thin: they validate requests with
Pydantic and delegate to this controller, which talks to the service
container. Keeping the logic here makes it testable without HTTP.
"""
from __future__ import annotations

import json
import os
import statistics
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from app.core.logging_setup import get_logger
from app.core.settings_service import GENERATOR_BACKENDS, KNOWN_EMBEDDING_MODELS
from app.evaluation.evaluator import Evaluator
from app.gnn.training_service import train_gnn
from app.graph.evidence_graph import EvidenceGraph
from app.graph.layout import force_layout
from app.preprocessing.chunker import chunk_statistics

if TYPE_CHECKING:  # pragma: no cover
    from app.core.container import Container

log = get_logger(__name__)

PIPELINE_STAGES = [
    ("ingestion", "Document ingestion", "PyMuPDF / python-docx text extraction"),
    ("chunking", "Chunking", "Sentence-aware, token budget with overlap"),
    ("ner", "Entity extraction", "spaCy NER + dependency relations"),
    ("embedding", "Embeddings", "Sentence-Transformers"),
    ("faiss", "Vector index", "FAISS IndexIDMap2(FlatIP)"),
    ("bm25", "Lexical index", "Okapi BM25"),
    ("graph", "Evidence graph", "NetworkX MultiDiGraph"),
    ("gnn", "GNN reranker", "PyTorch Geometric GAT"),
    ("generator", "Answer generation", "Grounded generation + citations"),
]


def _dir_size(path: Path) -> int:
    total = 0
    if path.exists():
        for root, _, files in os.walk(path):
            for f in files:
                try:
                    total += (Path(root) / f).stat().st_size
                except OSError:
                    pass
    return total


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    idx = min(len(values) - 1, max(0, int(round(q * (len(values) - 1)))))
    return values[idx]


class APIController:
    def __init__(self, container: "Container") -> None:
        self.c = container

    # ============================================================ documents
    def upload(self, files: list[tuple[str, bytes]], auto_process: bool = True) -> list[dict[str, Any]]:
        results = []
        for filename, data in files:
            try:
                doc, duplicate = self.c.processor.register_upload(filename, data)
            except ValueError as exc:
                results.append({"filename": filename, "error": str(exc)})
                continue
            job_id = None
            if auto_process and not duplicate:
                job_id = self.process(doc.id)["job_id"]
            results.append({"filename": filename, "document": self.c.repo.get_document(doc.id),
                            "duplicate": duplicate, "job_id": job_id})
        return results

    def process(self, doc_id: str) -> dict[str, Any]:
        if self.c.repo.get_document(doc_id) is None:
            raise KeyError(f"Document {doc_id} not found")
        self.c.repo.update_document(doc_id, status="queued", error=None)
        job_id = self.c.jobs.submit("process_document", self.c.ingestion.process_document, doc_id, pool="ingest",
                                    message=f"Processing {doc_id}")
        return {"document_id": doc_id, "job_id": job_id}

    def list_documents(self) -> dict[str, Any]:
        docs = self.c.repo.list_documents(source_type="upload")
        counts = self.c.repo.counts()
        return {"documents": docs, "stats": {
            "total_documents": counts["documents"], "processed_documents": counts["processed_documents"],
            "total_pages": counts["pages"], "total_chunks": counts["chunks"],
            "avg_chunk_tokens": round(counts["avg_chunk_tokens"] or 0, 1),
            "avg_chunk_chars": round(counts["avg_chunk_chars"] or 0, 1),
            "total_entities": counts["entities"], "entity_mentions": counts["entity_mentions"],
        }}

    def get_document(self, doc_id: str) -> dict[str, Any]:
        doc = self.c.repo.get_document(doc_id)
        if doc is None:
            raise KeyError(f"Document {doc_id} not found")
        chunks = self.c.repo.get_chunks(doc_id)
        return {
            "document": doc,
            "pages": [p.__dict__ for p in self.c.repo.get_pages(doc_id)],
            "sections": [s.__dict__ for s in self.c.repo.get_sections(doc_id)],
            "chunks": [c.to_dict() for c in chunks],
            "chunk_stats": chunk_statistics(chunks),
            "entities": self.c.repo.document_entity_summary(doc_id),
            "relations": [r.__dict__ for r in self.c.repo.get_relations(doc_id, only_processed=False)][:200],
        }

    def delete_document(self, doc_id: str) -> dict[str, Any]:
        self.c.ingestion.delete_document(doc_id)
        return {"deleted": doc_id}

    # ================================================================== QA
    def ask(self, question: str, mode: str, strategy: str, top_k: Optional[int], max_hops: Optional[int],
            backend: Optional[str]) -> dict[str, Any]:
        return self.c.pipeline.ask(question, mode=mode, strategy=strategy, top_k=top_k, max_hops=max_hops,
                                   backend=backend)

    def query(self, query_id: str) -> dict[str, Any]:
        payload = self.c.repo.get_query_payload(query_id)
        if payload is None:
            raise KeyError(f"Query {query_id} not found")
        return payload

    # =============================================================== graph
    def full_graph(self, limit: int = 300, search: Optional[str] = None,
                   node_types: Optional[list[str]] = None) -> dict[str, Any]:
        graph = self.c.corpus_manager.corpus.graph
        G = graph.G
        full_stats = graph.stats()
        types = set(node_types or ["document", "passage", "entity"])
        candidates = [n for n, d in G.nodes(data=True) if d.get("type") in types]
        if search:
            s = search.lower()
            hits = [n for n in candidates if s in str(G.nodes[n].get("label", "")).lower()]
            keep = set(hits)
            for n in hits:
                keep |= set(graph.neighbors(n))
            candidates = [n for n in candidates if n in keep]
        # documents and passages first, then entities by degree
        priority = {"document": 0, "passage": 1, "entity": 2}
        candidates.sort(key=lambda n: (priority.get(G.nodes[n].get("type"), 3), -G.degree(n), n))
        selected = candidates[:limit]
        sub = graph.subgraph(selected)
        sub.prune(remove_isolated=bool(search) is False and len(selected) > 50)
        positions = force_layout(sub)
        out = sub.to_dict(positions=positions)
        out["full_stats"] = full_stats
        out["truncated"] = len(candidates) > limit
        return out

    def neighbors(self, node_id: str) -> dict[str, Any]:
        graph = self.c.corpus_manager.corpus.graph
        if node_id not in graph.G:
            raise KeyError(f"Node {node_id} not found")
        nodes = [node_id, *graph.neighbors(node_id)[:60]]
        sub = graph.subgraph(nodes).to_dict()
        center = next(n for n in sub["nodes"] if n["id"] == node_id)  # same {id, type, label, data} shape
        return {"center": node_id, "node": center, **sub}

    def graph_query(self, query_id: Optional[str], question: Optional[str], view: str, top_k: Optional[int],
                    max_hops: Optional[int]) -> dict[str, Any]:
        if query_id:
            path = self.c.settings.query_graphs_dir / f"{query_id}.json"
            if not path.exists():
                raise KeyError(f"No saved graph for query {query_id}")
            data = json.loads(path.read_text(encoding="utf-8"))
            graph = data.get(view) or data.get("subgraph")
            return {"query_id": query_id, "question": data.get("question"), "view": view if data.get(view) else
                    "subgraph", **graph}
        if not question:
            raise ValueError("Provide query_id or question")
        corpus = self.c.corpus_manager.corpus
        if corpus.is_empty:
            raise ValueError("No processed documents to build a graph from")
        rt = self.c.runtime()
        result = self.c.reasoner.analyze(corpus, question, top_k=top_k or rt.top_k, max_hops=max_hops or rt.max_hops,
                                         alpha=rt.hybrid_alpha)
        chain = result.chains_by_strategy.get("gnn", result.chains)
        views = self.c.pipeline.graph_views(result, {"chain": chain[0] if chain else None, "citations": []})
        graph = views.get(view) or views["subgraph"]
        return {"query_id": None, "question": question, "view": view if views.get(view) else "subgraph",
                "chains": [c.to_dict() for c in chain[:5]], **graph}

    # ============================================================ analytics
    def overview(self) -> dict[str, Any]:
        counts = self.c.repo.counts()
        corpus = self.c.corpus_manager.corpus
        graph_stats = corpus.graph.stats()
        runs = self.c.repo.list_eval_runs(limit=20)
        latest = {}
        for r in runs:
            if r["status"] == "completed" and r["dataset"] not in latest:
                latest[r["dataset"]] = {"run_id": r["id"], "created_at": r["created_at"],
                                        "summary": (r["summary"] or {}).get("by_strategy", {})}
        query_rows = self.c.repo.metrics_rows("query", limit=2000)
        totals: dict[str, float] = {}
        for row in query_rows:
            totals[row["ref_id"]] = totals.get(row["ref_id"], 0.0) + row["duration_ms"]
        return {
            "project": {"name": "GraphRAG-QA",
                        "description": "Multi-hop question answering across documents and Wikipedia using "
                                       "evidence graphs and graph neural networks."},
            "counts": {
                "documents": counts["documents"], "processed_documents": counts["processed_documents"],
                "pages": counts["pages"], "passages": counts["chunks"], "entities": counts["entities"],
                "entity_mentions": counts["entity_mentions"], "relations": counts["relations"],
                "graph_nodes": graph_stats["nodes"], "graph_edges": graph_stats["edges"],
                "questions_answered": counts["questions_answered"],
                "wikipedia_cache_entries": counts["wikipedia_articles_cached"],
                "avg_chunk_tokens": round(counts["avg_chunk_tokens"] or 0, 1),
            },
            "graph": graph_stats,
            "recent_queries": self.c.repo.list_queries(limit=8),
            "latency": {"queries_measured": len(totals),
                        "avg_query_ms": round(statistics.mean(totals.values()), 1) if totals else None,
                        "p95_query_ms": round(_percentile(list(totals.values()), 0.95), 1) if totals else None},
            "latest_evaluations": latest,
            "pipeline": self.pipeline_status(),
            "health": self.health(check_wikipedia=False),
        }

    def pipeline_status(self) -> list[dict[str, Any]]:
        corpus = self.c.corpus_manager.corpus
        gen = self.c.generator.backend_status()
        gnn = self.c.gnn_ranker.status()
        state = {
            "ingestion": ("ready", f"{len(corpus.documents)} documents"),
            "chunking": ("ready", f"{len(corpus.chunks)} passages"),
            "ner": ("ready" if self.c.extractor.backend.startswith("spacy") else
                    "degraded" if self.c.extractor.backend == "regex_fallback" else "idle", self.c.extractor.backend),
            "embedding": ("ready" if self.c.embedder.backend == "sentence-transformers" else
                          "degraded" if self.c.embedder.backend == "hashing_fallback" else "idle",
                          self.c.embedder.model_id if self.c.embedder.backend != "not_loaded" else "not loaded"),
            "faiss": ("ready", f"{corpus.vector_store.size} vectors"),
            "bm25": ("ready", f"{corpus.bm25.size} passages"),
            "graph": ("ready", f"{corpus.graph.num_nodes} nodes / {corpus.graph.num_edges} edges"),
            "gnn": ("ready" if gnn["available"] else "untrained",
                    f"trained on {gnn['meta'].get('training_source')}" if gnn["available"] else gnn["reason"]),
            "generator": ("ready" if gen["hf_generator"]["available"] or gen["hf_reader"]["available"] else "degraded",
                          "local HF models" if gen["hf_generator"]["available"] else "extractive fallback"),
        }
        return [{"key": k, "label": label, "tech": tech, "status": state[k][0], "detail": state[k][1]}
                for k, label, tech in PIPELINE_STAGES]

    def retrieval_analytics(self, query_id: Optional[str]) -> dict[str, Any]:
        query_id = query_id or self.c.repo.latest_query_id()
        payload = self.c.repo.get_query_payload(query_id) if query_id else None
        runs = [r for r in self.c.repo.list_eval_runs(limit=30) if r["status"] == "completed"]
        return {
            "query_id": query_id,
            "question": payload["question"] if payload else None,
            "score_table": payload.get("score_table", []) if payload else [],
            "strategy_rankings": payload.get("strategy_rankings", {}) if payload else {},
            "chains": payload.get("chains", []) if payload else [],
            "effective_strategy": payload.get("effective_strategy") if payload else None,
            "evaluation_runs": [{"run_id": r["id"], "dataset": r["dataset"], "created_at": r["created_at"],
                                 "config": r["config"], "summary": r["summary"]} for r in runs],
            "gnn": self.gnn_status(),
            "recent_queries": self.c.repo.list_queries(limit=25),
        }

    def performance(self) -> dict[str, Any]:
        def aggregate(kind: str) -> list[dict[str, Any]]:
            by_stage: dict[str, list[float]] = {}
            for row in self.c.repo.metrics_rows(kind, limit=20000):
                by_stage.setdefault(row["stage"], []).append(row["duration_ms"])
            return [{"stage": st, "count": len(v), "mean_ms": round(statistics.mean(v), 2),
                     "p50_ms": round(_percentile(v, 0.5), 2), "p95_ms": round(_percentile(v, 0.95), 2),
                     "max_ms": round(max(v), 2)} for st, v in sorted(by_stage.items(), key=lambda kv: -statistics.mean(kv[1]))]

        bench_path = self.c.settings.results_dir / "latency_results.json"
        recent = []
        for q in self.c.repo.list_queries(limit=30):
            if q["status"] == "completed":
                recent.append({"query_id": q["id"], "question": q["question"], "mode": q["mode"],
                               "strategy": q["strategy"], "latency_ms": q["latency_ms"], "created_at": q["created_at"]})
        return {
            "query_stages": aggregate("query"),
            "ingestion_stages": aggregate("ingestion"),
            "recent_queries": recent,
            "benchmark": json.loads(bench_path.read_text(encoding="utf-8")) if bench_path.exists() else None,
            "warmup": self.c.warmup_state,
            "model_load_ms": {"embedding": self.c.embedder.load_ms, "spacy": self.c.extractor.load_ms,
                              "reader": self.c.generator.reader.load_ms, "generator": self.c.generator.seq2seq.load_ms},
        }

    # =========================================================== evaluation
    def start_evaluation(self, **kwargs: Any) -> dict[str, Any]:
        evaluator = Evaluator(self.c)
        run_id = f"eval-{os.urandom(5).hex()}"
        job_id = self.c.jobs.submit("evaluation", lambda progress: {
            k: v for k, v in evaluator.run(run_id=run_id, progress=progress, **kwargs).items() if k != "rows"},
            job_id=f"job-{run_id}", message=f"Evaluating on {kwargs.get('dataset')}")
        return {"run_id": run_id, "job_id": job_id}

    def evaluation_results(self, run_id: Optional[str]) -> dict[str, Any]:
        if run_id:
            run = self.c.repo.get_eval_run(run_id)
            if run is None:
                raise KeyError(f"Evaluation run {run_id} not found")
            return {"run": run, "rows": self.c.repo.get_eval_results(run_id),
                    "job": self.c.repo.get_job(f"job-{run_id}")}
        return {"runs": self.c.repo.list_eval_runs(limit=50), "datasets": self.datasets()}

    def evaluation_csv(self, run_id: str) -> str:
        run = self.c.repo.get_eval_run(run_id)
        if run is None:
            raise KeyError(f"Evaluation run {run_id} not found")
        return Evaluator.to_csv(run_id, run["dataset"], self.c.repo.get_eval_results(run_id))

    def datasets(self) -> list[dict[str, Any]]:
        samples = self.c.settings.samples_dir / "datasets"
        hp = samples / "hotpotqa" / "test.json"
        demo = json.loads((samples / "demo_qa.json").read_text(encoding="utf-8"))
        wiki = json.loads((samples / "wikipedia_demo.json").read_text(encoding="utf-8"))
        return [
            {"id": "demo", "name": demo["name"], "questions": len(demo["questions"]), "available": True,
             "description": demo["description"]},
            {"id": "hotpotqa", "name": "HotpotQA (distractor) test sample",
             "questions": len(json.loads(hp.read_text(encoding="utf-8"))["examples"]) if hp.exists() else 0,
             "available": hp.exists(),
             "description": "Real multi-hop questions with 10 Wikipedia paragraphs each (2 gold). Run "
                            "scripts/download_datasets.py to fetch the sample."},
            {"id": "wikipedia", "name": wiki["name"], "questions": len(wiki["questions"]), "available": True,
             "description": wiki["description"]},
        ]

    # ================================================================= GNN
    def gnn_status(self) -> dict[str, Any]:
        status = self.c.gnn_ranker.status()
        status["history"] = self.c.gnn_ranker.training_history()
        status["latest_job"] = self.c.repo.latest_job("gnn_training")
        status["hotpotqa_available"] = (self.c.settings.samples_dir / "datasets" / "hotpotqa" / "train.json").exists()
        return status

    def start_training(self, **kwargs: Any) -> dict[str, Any]:
        job_id = self.c.jobs.submit("gnn_training", lambda progress: train_gnn(self.c, progress=progress, **kwargs),
                                    message=f"Training GAT reranker on {kwargs.get('source')}")
        return {"job_id": job_id}

    # ======================================================== settings/health
    def get_settings(self) -> dict[str, Any]:
        s = self.c.settings
        storage = {name: _dir_size(path) for name, path in (
            ("uploads", s.uploads_dir), ("indexes", s.indexes_dir), ("graphs", s.graphs_dir),
            ("processed", s.processed_dir), ("models", s.models_dir), ("results", s.results_dir))}
        storage["database"] = s.db_path.stat().st_size if s.db_path.exists() else 0
        return {
            "runtime": self.c.runtime().model_dump(),
            "defaults": self.c.settings_service.defaults(),
            "options": {"generator_backends": GENERATOR_BACKENDS, "embedding_models": KNOWN_EMBEDDING_MODELS,
                        "strategies": ["faiss", "bm25", "hybrid", "graph", "gnn"]},
            "generator_status": self.c.generator.backend_status(),
            "gnn_status": self.c.gnn_ranker.status(),
            "models": {"embedding": self.c.embedder.model_id, "embedding_backend": self.c.embedder.backend,
                       "spacy": s.spacy_model, "ner_backend": self.c.extractor.backend,
                       "reader": s.hf_reader_model, "generator": s.hf_generator_model},
            "storage": {"paths": {"data": str(s.data_dir), "models": str(s.models_dir), "database": str(s.db_path),
                                  "results": str(s.results_dir)}, "bytes": storage},
            "limits": {"max_upload_mb": s.max_upload_mb, "allowed_extensions": list(s.allowed_extensions)},
        }

    def update_settings(self, changes: dict[str, Any]) -> dict[str, Any]:
        updated, effects = self.c.settings_service.update(changes)
        if effects["requires_reindex"]:
            self.c.reload_embedder()
            effects["reindex_job"] = self.c.jobs.submit("reindex", lambda progress: {
                "vectors": self.c.corpus_manager.rebuild_index()}, pool="ingest", message="Rebuilding FAISS index")
        return {"runtime": updated.model_dump(), **effects}

    def reprocess_all(self) -> dict[str, Any]:
        jobs = [self.process(d["id"])["job_id"] for d in self.c.repo.list_documents(source_type="upload")]
        return {"jobs": jobs}

    def health(self, check_wikipedia: bool = False) -> dict[str, Any]:
        e, x = self.c.embedder, self.c.extractor
        components = {
            "database": {"status": "ok", "path": str(self.c.settings.db_path)},
            "embedding_model": {"status": "ok" if e.backend == "sentence-transformers" else
                                "degraded" if e.backend == "hashing_fallback" else "loading",
                                "backend": e.backend, "model": e.model_name, "error": e.error},
            "ner_model": {"status": "ok" if x.backend.startswith("spacy") else
                          "degraded" if x.backend == "regex_fallback" else "loading", "backend": x.backend,
                          "error": x.error},
            "vector_index": {"status": "ok", "vectors": self.c.corpus_manager.store.size},
            "gnn": {"status": "ok" if self.c.gnn_ranker.is_available else "untrained",
                    "reason": self.c.gnn_ranker.reason},
            "generator": {"status": "ok", "backends": {k: v["available"] for k, v in
                                                       self.c.generator.backend_status().items()}},
        }
        if check_wikipedia:
            ok = self.c.open_domain.client.is_reachable()
            components["wikipedia"] = {"status": "ok" if ok else "unreachable"}
        statuses = {v["status"] for v in components.values()}
        overall = "ok" if statuses <= {"ok", "untrained"} else "degraded"
        return {"status": overall, "components": components, "warmup": self.c.warmup_state,
                "started_at": self.c.started_at}
