"""End-to-end question answering (multi-document and open-domain).

    question
      -> corpus (uploaded documents, or Wikipedia articles fetched for the question)
      -> GraphReasoner: FAISS / BM25 / hybrid retrieval, question subgraph,
         evidence chains, graph-heuristic and GNN scores
      -> evidence selection for the chosen strategy
      -> AnswerGenerator (grounded) -> CitationBuilder (validated)
      -> confidence + insufficient-evidence decision
      -> persisted query, answer, citations, subgraph JSON and stage timings
"""
from __future__ import annotations

import json
import uuid
from typing import TYPE_CHECKING, Any, Optional

from app.citations.citation_builder import CitationBuilder
from app.core.domain import EvidenceChain, RetrievedPassage
from app.core.logging_setup import get_logger
from app.core.timing import StageTimer
from app.generation.confidence import assess_confidence
from app.generation.generator import INSUFFICIENT_MESSAGE, EvidencePassage, GenerationResult
from app.graph.evidence_graph import QUESTION_NODE, passage_node
from app.graph.layout import chain_layout, force_layout
from app.graph.traversal import chain_highlight
from app.pipeline.graph_reasoner import STRATEGIES, ReasoningResult
from app.retrieval.corpus import Corpus

if TYPE_CHECKING:  # pragma: no cover
    from app.core.container import Container

log = get_logger(__name__)


class NoDocumentsError(RuntimeError):
    """Multi-document mode was used before any document was processed."""


class QAPipeline:
    def __init__(self, container: "Container") -> None:
        self.c = container
        self.citations = CitationBuilder()

    # ================================================================ public
    def ask(self, question: str, mode: str = "multi_document", strategy: str = "gnn", top_k: Optional[int] = None,
            max_hops: Optional[int] = None, backend: Optional[str] = None, persist: bool = True,
            corpus: Optional[Corpus] = None) -> dict[str, Any]:
        question = " ".join(question.split())
        if strategy not in STRATEGIES:
            raise ValueError(f"Unknown strategy '{strategy}'. Use one of {', '.join(STRATEGIES)}")
        rt = self.c.runtime()
        top_k = top_k or rt.top_k
        max_hops = max_hops or rt.max_hops
        backend = backend or rt.generator_backend
        timer = StageTimer()
        query_id = uuid.uuid4().hex[:12]
        open_domain: Optional[dict[str, Any]] = None
        if corpus is None:
            if mode == "open_domain":
                corpus, open_domain = self.c.open_domain.build_corpus(question, timer)
            else:
                corpus = self.c.corpus_manager.corpus
                if corpus.is_empty:
                    raise NoDocumentsError(
                        "No processed documents yet. Upload documents in the Document Workspace (or run "
                        "`python scripts/seed_demo.py`) before asking multi-document questions."
                    )
        result = self.c.reasoner.analyze(corpus, question, top_k=top_k, max_hops=max_hops, alpha=rt.hybrid_alpha,
                                         per_hop=self.c.settings.graph_expansion_per_hop,
                                         max_chains=self.c.settings.max_chains, timer=timer)
        answer = self.answer_from_result(question, corpus, result, strategy, backend, rt.max_evidence_passages, timer)
        payload = self._payload(query_id, question, mode, strategy, top_k, max_hops, corpus, result, answer, timer,
                                open_domain)
        if persist:
            self._persist(payload, result, answer)
        return payload

    # ========================================================== answering
    def select_evidence(self, result: ReasoningResult, ranking: list[RetrievedPassage],
                        chain: Optional[EvidenceChain], max_evidence: int) -> list[EvidencePassage]:
        scored = result.qgraph.passages
        evidence: list[EvidencePassage] = []
        seen: set[str] = set()

        def add(rp: RetrievedPassage, chain_step: Optional[int] = None, via: Optional[str] = None,
                reason: str = "") -> None:
            if rp.chunk.id in seen:
                return
            seen.add(rp.chunk.id)
            full = scored.get(rp.chunk.id, rp)
            evidence.append(EvidencePassage(
                index=len(evidence) + 1, chunk=rp.chunk, hybrid_score=full.hybrid_score,
                semantic_score=full.semantic_score, graph_score=full.graph_score, gnn_score=full.gnn_score,
                final_score=rp.final_score, chain_step=chain_step, via_entity_name=via, reason=reason,
            ))

        # the evidence chain (if any) comes first, then the strategy's ranking fills the remaining slots
        limit = max(max_evidence, chain.length if chain is not None else 0)
        if chain is not None:
            for i, step in enumerate(chain.steps, start=1):
                rp = scored.get(step.chunk_id)
                if rp is not None:
                    add(rp, chain_step=i, via=step.via_entity_name, reason=step.reason)
        for rp in ranking:
            if len(evidence) >= limit:
                break
            add(rp, reason=f"Ranked #{rp.rank} by the selected strategy")
        return evidence

    def answer_from_result(self, question: str, corpus: Corpus, result: ReasoningResult, strategy: str,
                           backend: str, max_evidence: int, timer: StageTimer) -> dict[str, Any]:
        notes = list(result.notes)
        effective = strategy
        if strategy == "gnn" and "gnn" not in result.rankings:
            effective = "graph"
            notes.append("Requested GNN reranking, but no trained checkpoint is available: the graph heuristic "
                         "ranking was used instead.")
        ranking = result.rankings[effective]
        chains = result.chains_by_strategy.get(effective, []) if effective in ("graph", "gnn") else []
        chain = chains[0] if chains else None
        evidence = self.select_evidence(result, ranking, chain, max_evidence)
        with timer.stage("answer_generation"):
            gen = self.c.generator.generate(question, evidence, chain, backend)
        with timer.stage("citation_validation"):
            citations, removed = self.citations.build(gen, evidence)
        settings = self.c.settings
        report = assess_confidence(question, evidence, chain, gen, settings.min_semantic_score,
                                   settings.min_term_coverage, settings.min_confidence,
                                   question_entities=result.question_entities)
        if not citations and not report.insufficient:
            report.insufficient = True
            report.reasons.append("No evidence passage could be cited for the answer.")
        final_answer = INSUFFICIENT_MESSAGE if report.insufficient else gen.answer
        return {
            "strategy": strategy,
            "effective_strategy": effective,
            "evidence": evidence,
            "chain": chain,
            "chains": chains,
            "generation": gen,
            "citations": citations,
            "removed_citations": removed,
            "confidence": report,
            "final_answer": final_answer,
            "notes": notes,
        }

    # ============================================================ payload
    @staticmethod
    def _evidence_chain_payload(chain: Optional[EvidenceChain], evidence: list[EvidencePassage],
                                corpus: Corpus) -> list[dict[str, Any]]:
        if chain is None:
            return []
        index_of = {ev.chunk.id: ev.index for ev in evidence}
        out = []
        for i, step in enumerate(chain.steps, start=1):
            chunk = corpus.chunk(step.chunk_id)
            out.append({
                "step": i, "chunk_id": step.chunk_id, "document_name": chunk.document_name,
                "page_number": chunk.page_number, "section": chunk.section, "url": chunk.url,
                "via_entity_id": step.via_entity, "via_entity_name": step.via_entity_name,
                "relevance": step.relevance, "reason": step.reason, "evidence_index": index_of.get(step.chunk_id),
                "text": chunk.text,
            })
        return out

    @staticmethod
    def score_table(result: ReasoningResult, top_k: int) -> list[dict[str, Any]]:
        """Per-passage scores and ranks of every strategy (for analytics)."""
        ranks: dict[str, dict[str, int]] = {}
        for name, ranking in result.rankings.items():
            for rp in ranking:
                ranks.setdefault(rp.chunk.id, {})[name] = rp.rank
        rows = []
        for cid, rp in result.qgraph.passages.items():
            if cid not in ranks and rp.origin != "retrieval":
                continue
            rows.append({
                "chunk_id": cid, "document_name": rp.chunk.document_name, "page_number": rp.chunk.page_number,
                "section": rp.chunk.section, "preview": rp.chunk.text[:220], "origin": rp.origin, "hop": rp.hop,
                "semantic_score": round(rp.semantic_score, 4), "bm25_score": round(rp.bm25_score, 4),
                "hybrid_score": round(rp.hybrid_score, 4),
                "graph_score": None if rp.graph_score is None else round(rp.graph_score, 4),
                "gnn_score": None if rp.gnn_score is None else round(rp.gnn_score, 4),
                "ranks": ranks.get(cid, {}),
            })
        for name, ranking in result.rankings.items():  # first-stage results outside the subgraph
            for rp in ranking:
                if rp.chunk.id not in result.qgraph.passages:
                    rows.append({**rp.to_dict(), "preview": rp.chunk.text[:220], "ranks": ranks.get(rp.chunk.id, {})})
        rows.sort(key=lambda r: -(r["hybrid_score"] or 0))
        return rows

    def _payload(self, query_id: str, question: str, mode: str, strategy: str, top_k: int, max_hops: int,
                 corpus: Corpus, result: ReasoningResult, answer: dict[str, Any], timer: StageTimer,
                 open_domain: Optional[dict[str, Any]]) -> dict[str, Any]:
        gen: GenerationResult = answer["generation"]
        report = answer["confidence"]
        evidence: list[EvidencePassage] = answer["evidence"]
        chain: Optional[EvidenceChain] = answer["chain"]
        effective = answer["effective_strategy"]
        cited = {c.index for c in answer["citations"]}
        insufficient = report.insufficient
        return {
            "query_id": query_id,
            "question": question,
            "mode": mode,
            "strategy": strategy,
            "effective_strategy": effective,
            "answer": answer["final_answer"],
            "candidate_answer": gen.answer or None,
            "answer_sentence": None if insufficient else gen.answer_sentence,
            "answer_citations": [] if insufficient else [i for i in gen.answer_citation_indices if i in cited],
            "reasoning_summary": "" if insufficient else gen.reasoning_summary,
            "reasoning_steps": [] if insufficient else [
                {"step": s.step, "kind": s.kind, "text": s.text, "citations": s.citation_indices}
                for s in gen.reasoning_steps
            ],
            "evidence_chain": self._evidence_chain_payload(chain, evidence, corpus),
            "chains": [c.to_dict() for c in answer["chains"][:5]],
            "citations": [] if insufficient else [c.to_dict() for c in answer["citations"]],
            "evidence_passages": [
                {"index": ev.index, "chunk_id": ev.chunk.id, "document_name": ev.chunk.document_name,
                 "page_number": ev.chunk.page_number, "section": ev.chunk.section, "url": ev.chunk.url,
                 "text": ev.chunk.text, "hybrid_score": round(ev.hybrid_score, 4),
                 "semantic_score": round(ev.semantic_score, 4),
                 "graph_score": None if ev.graph_score is None else round(ev.graph_score, 4),
                 "gnn_score": None if ev.gnn_score is None else round(ev.gnn_score, 4),
                 "chain_step": ev.chain_step, "reason": ev.reason, "cited": ev.index in cited}
                for ev in evidence
            ],
            "retrieved_passages": [rp.to_dict() for rp in result.rankings[effective]],
            "strategy_rankings": {name: [rp.chunk.id for rp in ranking] for name, ranking in result.rankings.items()},
            "score_table": self.score_table(result, top_k),
            "confidence": report.confidence,
            "confidence_components": report.components,
            "evidence_coverage": report.coverage,
            "insufficient_evidence": insufficient,
            "insufficient_reasons": report.reasons,
            "question_entities": result.question_entities,
            "generator_backend": gen.backend,
            "generator_warnings": gen.warnings,
            "removed_citations": answer["removed_citations"],
            "gnn_used": effective == "gnn",
            "notes": answer["notes"],
            "graph_stats": result.qgraph.graph.stats(),
            "expansion_log": result.qgraph.expansion_log,
            "timings_ms": timer.as_dict(),
            "total_ms": round(timer.total_ms, 1),
            "top_k": top_k,
            "max_hops": max_hops,
            "corpus": {"name": corpus.name, "documents": len(corpus.documents), "chunks": len(corpus.chunks)},
            "open_domain": open_domain,
        }

    def graph_views(self, result: ReasoningResult, answer: dict[str, Any]) -> dict[str, Any]:
        graph = result.qgraph.graph
        chain: Optional[EvidenceChain] = answer["chain"]
        cited_nodes = {passage_node(c.chunk_id) for c in answer["citations"]}
        nodes, edges = chain_highlight(chain) if chain else ({QUESTION_NODE}, set())
        positions = force_layout(graph, center=QUESTION_NODE)
        subgraph = graph.to_dict(positions=positions, highlight_nodes=nodes | cited_nodes, highlight_edges=edges)
        chain_view = None
        if chain:
            keep = set(nodes)
            for n in list(keep):
                if n.startswith("psg:") and n in graph.G:
                    doc = graph.G.nodes[n].get("document_id")
                    if doc:
                        keep.add(f"doc:{doc}")
            sub = graph.subgraph([n for n in keep if n in graph.G])
            chain_view = sub.to_dict(positions=chain_layout(sub, chain), highlight_nodes=nodes, highlight_edges=edges)
        return {"subgraph": subgraph, "chain": chain_view}

    def _persist(self, payload: dict[str, Any], result: ReasoningResult, answer: dict[str, Any]) -> None:
        query_id = payload["query_id"]
        try:
            views = self.graph_views(result, answer)
            path = self.c.settings.query_graphs_dir / f"{query_id}.json"
            path.write_text(json.dumps({"query_id": query_id, "question": payload["question"], **views}),
                            encoding="utf-8")
        except Exception as exc:  # graph view is best effort; never fail the answer
            log.warning("Could not save graph view for %s: %s", query_id, exc)
        self.c.repo.save_query_result(query_id, payload["question"], payload["mode"], payload["strategy"],
                                      payload["top_k"], payload["max_hops"], payload["total_ms"], payload)
        self.c.repo.log_metrics("query", query_id, payload["timings_ms"],
                                {"mode": payload["mode"], "strategy": payload["strategy"],
                                 "effective_strategy": payload["effective_strategy"],
                                 "graph_nodes": payload["graph_stats"]["nodes"],
                                 "graph_edges": payload["graph_stats"]["edges"],
                                 "chain_length": len(payload["evidence_chain"]),
                                 "insufficient": payload["insufficient_evidence"]})
