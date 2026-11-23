"""Dataset-based evaluation of all retrieval / reranking strategies.

For every question the expensive analysis (retrieval scores, subgraph, chains,
heuristic and GNN scores) is computed once; each strategy then selects its own
evidence and generates its own answer from that shared analysis. Strategies
are therefore compared on identical candidates.

Datasets
    demo      - bundled documents + data/samples/datasets/demo_qa.json
    hotpotqa  - HotpotQA distractor test sample (scripts/download_datasets.py)
    wikipedia - live open-domain questions (needs internet; article-level metrics)
"""
from __future__ import annotations

import csv
import io
import json
import time
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Iterator, Optional

from app.core.logging_setup import get_logger
from app.database.repository import utc_now
from app.core.timing import StageTimer
from app.evaluation import metrics as M
from app.gnn.dataset import (demo_corpus, gold_chunks_for_evidence, hotpot_corpus, load_demo_dataset,
                             load_hotpotqa)
from app.pipeline.graph_reasoner import STRATEGIES, GraphReasoner
from app.retrieval.corpus import Corpus

if TYPE_CHECKING:  # pragma: no cover
    from app.core.container import Container

log = get_logger(__name__)

CSV_COLUMNS = [
    "run_id", "dataset", "question_id", "question_type", "question", "strategy", "expected_answer",
    "predicted_answer", "insufficient", "exact_match", "f1", "precision_at_k", "recall_at_k", "mrr",
    "evidence_recall", "citation_coverage", "citation_precision", "answer_latency_ms", "graph_latency_ms",
    "generator_backend", "confidence",
]


@dataclass
class EvalItem:
    qid: str
    question: str
    qtype: str
    answers: list[str]
    corpus: Corpus
    gold: set[str]  # chunk ids, or document names for article-level datasets
    level: str = "chunk"  # chunk | document


class Evaluator:
    def __init__(self, container: "Container") -> None:
        self.c = container

    # ---------------------------------------------------------------- items
    def items(self, dataset: str, limit: Optional[int] = None) -> Iterator[EvalItem]:
        samples = self.c.settings.samples_dir
        if dataset == "demo":
            corpus = demo_corpus(self.c)
            data = load_demo_dataset(samples)
            for q in data["questions"][: limit or None]:
                gold = gold_chunks_for_evidence(corpus, q.get("gold_evidence", []), q.get("gold_documents"))
                yield EvalItem(q["id"], q["question"], q["type"], [q["answer"], *q.get("answer_aliases", [])],
                               corpus, gold)
        elif dataset == "hotpotqa":
            for ex in load_hotpotqa(samples, "test")[: limit or None]:
                corpus, gold = hotpot_corpus(self.c, ex)
                yield EvalItem(ex["id"], ex["question"], ex.get("type", "bridge"), [ex["answer"]], corpus, gold)
        elif dataset == "wikipedia":
            data = json.loads((samples / "datasets" / "wikipedia_demo.json").read_text(encoding="utf-8"))
            for q in data["questions"][: limit or None]:
                corpus, _ = self.c.open_domain.build_corpus(q["question"])
                yield EvalItem(q["id"], q["question"], q["type"], [q["answer"], *q.get("answer_aliases", [])],
                               corpus, set(q.get("gold_articles", [])), level="document")
        else:
            raise ValueError(f"Unknown dataset '{dataset}' (use demo, hotpotqa or wikipedia)")

    def dataset_size(self, dataset: str) -> int:
        samples = self.c.settings.samples_dir
        if dataset == "demo":
            return len(load_demo_dataset(samples)["questions"])
        if dataset == "hotpotqa":
            return len(load_hotpotqa(samples, "test"))
        data = json.loads((samples / "datasets" / "wikipedia_demo.json").read_text(encoding="utf-8"))
        return len(data["questions"])

    # ------------------------------------------------------------------ run
    def run(self, dataset: str = "demo", strategies: Optional[list[str]] = None, top_k: Optional[int] = None,
            max_hops: Optional[int] = None, limit: Optional[int] = None, backend: Optional[str] = None,
            k: int = 2, run_id: Optional[str] = None, persist: bool = True,
            progress: Optional[Callable[[float, str], None]] = None) -> dict[str, Any]:
        rt = self.c.runtime()
        strategies = [s for s in (strategies or list(STRATEGIES)) if s in STRATEGIES]
        top_k = top_k or (4 if dataset == "hotpotqa" else rt.top_k)
        max_hops = max_hops or (2 if dataset == "hotpotqa" else rt.max_hops)
        backend = backend or rt.generator_backend
        run_id = run_id or f"eval-{uuid.uuid4().hex[:10]}"
        config = {"dataset": dataset, "strategies": strategies, "top_k": top_k, "max_hops": max_hops, "k": k,
                  "limit": limit, "backend": backend, "hybrid_alpha": rt.hybrid_alpha,
                  "chunk_size": rt.chunk_size, "chunk_overlap": rt.chunk_overlap,
                  "embedding_model": self.c.embedder.model_id,
                  "gnn_checkpoint": self.c.gnn_ranker.status().get("meta", {}).get("trained_at")}
        if persist:
            self.c.repo.create_eval_run(run_id, dataset, config)
        total = min(limit, self.dataset_size(dataset)) if limit else self.dataset_size(dataset)
        rows: list[dict[str, Any]] = []
        skipped: list[str] = []
        try:
            for i, item in enumerate(self.items(dataset, limit)):
                rows.extend(self._evaluate_item(item, strategies, top_k, max_hops, backend, k, skipped))
                if progress:
                    progress((i + 1) / max(1, total), f"{i + 1}/{total} questions")
                if persist and (i + 1) % 10 == 0:
                    self.c.repo.update_eval_run(run_id, progress=round((i + 1) / max(1, total), 4))
            summary = self.summarize(rows, strategies)
            summary["skipped"] = sorted(set(skipped))
            summary["num_questions"] = len({r["question_id"] for r in rows})
            if persist:
                self.c.repo.add_eval_results(run_id, rows)
                self.c.repo.update_eval_run(run_id, status="completed", summary=summary, progress=1.0,
                                            finished_at=utc_now())
        except Exception as exc:
            if persist:
                self.c.repo.update_eval_run(run_id, status="failed", error=f"{exc.__class__.__name__}: {exc}")
            raise
        return {"run_id": run_id, "config": config, "summary": summary, "rows": rows}

    def _evaluate_item(self, item: EvalItem, strategies: list[str], top_k: int, max_hops: int, backend: str,
                       k: int, skipped: list[str]) -> list[dict[str, Any]]:
        timer = StageTimer()
        t0 = time.perf_counter()
        result = self.c.reasoner.analyze(item.corpus, item.question, top_k=top_k, max_hops=max_hops,
                                         alpha=self.c.runtime().hybrid_alpha, timer=timer)
        analysis_ms = (time.perf_counter() - t0) * 1000
        graph_ms = timer.durations_ms.get("graph_construction", 0.0) + timer.durations_ms.get("graph_traversal", 0.0)
        unanswerable = item.qtype == "unanswerable"
        rows = []
        for strategy in strategies:
            if strategy == "gnn" and "gnn" not in result.rankings:
                skipped.append("gnn (no trained checkpoint)")
                continue
            t1 = time.perf_counter()
            ans = self.c.pipeline.answer_from_result(item.question, item.corpus, result, strategy, backend,
                                                     self.c.runtime().max_evidence_passages, StageTimer())
            gen_ms = (time.perf_counter() - t1) * 1000
            insufficient = ans["confidence"].insufficient
            predicted = "" if insufficient else ans["final_answer"]
            ranking = GraphReasoner.full_ranking(result, item.corpus, strategy)
            evidence_ids = [ev.chunk.id for ev in ans["evidence"]]
            cited_ids = [] if insufficient else [c.chunk_id for c in ans["citations"]]
            if item.level == "document":
                to_doc = {c.id: c.document_name for c in item.corpus.chunks}
                ranking = list(dict.fromkeys(to_doc[r] for r in ranking))
                evidence_ids = list(dict.fromkeys(to_doc[r] for r in evidence_ids))
                cited_ids = list(dict.fromkeys(to_doc[r] for r in cited_ids))
            if unanswerable:
                em = f1 = 1.0 if insufficient else 0.0
            else:
                em, f1 = M.exact_match(predicted, item.answers), M.f1(predicted, item.answers)
            cov, prec = M.citation_scores(cited_ids, item.gold)
            rows.append({
                "question_id": item.qid, "question_type": item.qtype, "question": item.question,
                "strategy": strategy, "expected_answer": item.answers[0] if item.answers else "",
                "predicted_answer": predicted, "insufficient": int(insufficient), "exact_match": em, "f1": round(f1, 4),
                "precision_at_k": M.precision_at_k(ranking, item.gold, k),
                "recall_at_k": M.recall_at_k(ranking, item.gold, k), "mrr": M.mrr(ranking, item.gold),
                "evidence_recall": M.evidence_recall(evidence_ids, item.gold), "citation_coverage": cov,
                "citation_precision": prec, "answer_latency_ms": round(analysis_ms + gen_ms, 1),
                "graph_latency_ms": round(graph_ms, 1),
                "details": {"generator_backend": ans["generation"].backend,
                            "confidence": ans["confidence"].confidence,
                            "effective_strategy": ans["effective_strategy"],
                            "candidate_answer": ans["generation"].answer,
                            "top_ranked": ranking[:k], "gold": sorted(item.gold)},
            })
        return rows

    # -------------------------------------------------------------- summary
    @staticmethod
    def summarize(rows: list[dict[str, Any]], strategies: list[str]) -> dict[str, Any]:
        out: dict[str, Any] = {"by_strategy": {}, "by_type": {}}
        for s in strategies:
            rs = [r for r in rows if r["strategy"] == s]
            if not rs:
                continue
            answerable = [r for r in rs if r["question_type"] != "unanswerable"]
            unanswerable = [r for r in rs if r["question_type"] == "unanswerable"]
            out["by_strategy"][s] = {
                "questions": len(rs),
                "exact_match": M.mean(r["exact_match"] for r in answerable),
                "f1": M.mean(r["f1"] for r in answerable),
                "precision_at_k": M.mean(r["precision_at_k"] for r in rs),
                "recall_at_k": M.mean(r["recall_at_k"] for r in rs),
                "mrr": M.mean(r["mrr"] for r in rs),
                "evidence_recall": M.mean(r["evidence_recall"] for r in rs),
                "citation_coverage": M.mean(r["citation_coverage"] for r in rs),
                "citation_precision": M.mean(r["citation_precision"] for r in rs),
                "abstention_accuracy": M.mean(r["exact_match"] for r in unanswerable),
                "false_abstention_rate": M.mean(r["insufficient"] for r in answerable),
                "answer_latency_ms": M.mean(r["answer_latency_ms"] for r in rs),
                "graph_latency_ms": M.mean(r["graph_latency_ms"] for r in rs),
            }
            for qtype in sorted({r["question_type"] for r in rs}):
                tr = [r for r in rs if r["question_type"] == qtype]
                out["by_type"].setdefault(qtype, {})[s] = {
                    "questions": len(tr), "exact_match": M.mean(r["exact_match"] for r in tr),
                    "f1": M.mean(r["f1"] for r in tr), "recall_at_k": M.mean(r["recall_at_k"] for r in tr),
                    "mrr": M.mean(r["mrr"] for r in tr),
                }
        return out

    @staticmethod
    def to_csv(run_id: str, dataset: str, rows: list[dict[str, Any]]) -> str:
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            details = r.get("details") or {}
            writer.writerow({**r, "run_id": run_id, "dataset": dataset,
                             "generator_backend": details.get("generator_backend"),
                             "confidence": details.get("confidence")})
        return buf.getvalue()
