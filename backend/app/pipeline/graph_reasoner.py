"""Question analysis shared by the QA pipeline, the evaluator and GNN training.

For one question over one corpus it computes, with per-stage timings:

1. semantic / BM25 / hybrid scores for every passage and the three
   first-stage rankings (FAISS, BM25, hybrid);
2. question entities and their matches in the entity registry;
3. the question subgraph (hybrid seeds + multi-hop expansion);
4. evidence chains (multi-hop traversal);
5. graph-heuristic scores (PPR + chain support);
6. GNN scores (when a compatible trained checkpoint is available).

It returns rankings for every strategy so that strategies are always compared
on exactly the same candidates.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

from app.core.domain import EvidenceChain, RetrievedPassage
from app.core.timing import StageTimer
from app.graph.builder import GraphBuilder, QuestionGraph
from app.graph.heuristic_ranker import GraphHeuristicRanker
from app.graph.traversal import GraphTraversal
from app.retrieval.retrievers import BM25Retriever, HybridRetriever, QueryScores, SemanticRetriever

if TYPE_CHECKING:  # pragma: no cover
    from app.embeddings.embedder import EmbeddingGenerator
    from app.entities.extractor import EntityExtractor
    from app.gnn.ranker import GNNRanker
    from app.retrieval.corpus import Corpus

STRATEGIES = ("faiss", "bm25", "hybrid", "graph", "gnn")


@dataclass
class ReasoningResult:
    question: str
    scores: QueryScores
    rankings: dict[str, list[RetrievedPassage]]
    qgraph: QuestionGraph
    chains: list[EvidenceChain]
    chains_by_strategy: dict[str, list[EvidenceChain]]
    heuristic: dict[str, dict[str, float]]
    gnn_scores: Optional[dict[str, float]]
    question_entities: list[dict[str, Any]]
    timer: StageTimer
    notes: list[str] = field(default_factory=list)


class GraphReasoner:
    def __init__(self, embedder: "EmbeddingGenerator", extractor: "EntityExtractor", graph_builder: GraphBuilder,
                 traversal: Optional[GraphTraversal] = None, heuristic: Optional[GraphHeuristicRanker] = None,
                 gnn: Optional["GNNRanker"] = None) -> None:
        self.embedder = embedder
        self.extractor = extractor
        self.graph_builder = graph_builder
        self.traversal = traversal or GraphTraversal()
        self.heuristic = heuristic or GraphHeuristicRanker()
        self.gnn = gnn

    def match_question_entities(self, corpus: "Corpus", question: str) -> tuple[dict[str, float], list[dict[str, Any]]]:
        matched: dict[str, float] = {}
        details: list[dict[str, Any]] = []
        for qe in self.extractor.question_entities(question):
            matches = corpus.registry.match(qe.text)
            good = [(eid, s) for eid, s in matches if s >= 0.6]
            for eid, s in good:
                matched[eid] = max(matched.get(eid, 0.0), s)
            details.append({
                "text": qe.text,
                "label": qe.label,
                "matches": [{"entity_id": eid, "name": corpus.registry.entities[eid].name, "score": round(s, 3)}
                            for eid, s in good],
            })
        return matched, details

    def analyze(self, corpus: "Corpus", question: str, top_k: int = 8, max_hops: int = 3, alpha: float = 0.7,
                per_hop: int = 12, max_chains: int = 8, use_gnn: bool = True,
                timer: Optional[StageTimer] = None) -> ReasoningResult:
        timer = timer or StageTimer()
        notes: list[str] = []
        with timer.stage("query_embedding"):
            scores = QueryScores(corpus, self.embedder, question, alpha)
        rankings: dict[str, list[RetrievedPassage]] = {}
        with timer.stage("semantic_retrieval"):
            rankings["faiss"] = SemanticRetriever(corpus, self.embedder, alpha).search(question, top_k, scores)
        with timer.stage("bm25_retrieval"):
            rankings["bm25"] = BM25Retriever(corpus, self.embedder, alpha).search(question, top_k, scores)
        with timer.stage("hybrid_retrieval"):
            rankings["hybrid"] = HybridRetriever(corpus, self.embedder, alpha).search(question, top_k, scores)
        with timer.stage("question_analysis"):
            q_entities, q_entity_details = self.match_question_entities(corpus, question)
        with timer.stage("graph_construction"):
            seeds = [RetrievedPassage(**{**rp.__dict__}) for rp in rankings["hybrid"]]
            qgraph = self.graph_builder.build_question_graph(
                corpus, scores, question, q_entities, seeds, max_hops=max_hops, per_hop=per_hop,
                question_entity_mentions=q_entity_details,
            )
        with timer.stage("graph_traversal"):
            chains = self.traversal.find_chains(qgraph, corpus, question, max_hops=max_hops, max_chains=max_chains)
        with timer.stage("graph_ranking"):
            heuristic = self.heuristic.score(qgraph, corpus, chains)
            for cid, rp in qgraph.passages.items():
                rp.graph_score = heuristic[cid]["graph_score"]
        rankings["graph"] = self._rerank(qgraph, corpus, scores, top_k, key="graph")
        chains_by_strategy = {"graph": chains}

        gnn_scores: Optional[dict[str, float]] = None
        if use_gnn and self.gnn is not None and self.gnn.is_available:
            with timer.stage("gnn_inference"):
                gnn_scores = self.gnn.score(qgraph, corpus, scores.query_vec)
            for cid, rp in qgraph.passages.items():
                rp.gnn_score = gnn_scores.get(cid)
            rankings["gnn"] = self._rerank(qgraph, corpus, scores, top_k, key="gnn")
            chains_by_strategy["gnn"] = self.traversal.rescore(chains, gnn_scores)

        # attach graph/GNN scores to first-stage results for side-by-side display
        for name in ("faiss", "bm25", "hybrid"):
            for rp in rankings[name]:
                q = qgraph.passages.get(rp.chunk.id)
                if q is not None:
                    rp.graph_score, rp.gnn_score = q.graph_score, q.gnn_score
        return ReasoningResult(
            question=question, scores=scores, rankings=rankings, qgraph=qgraph, chains=chains,
            chains_by_strategy=chains_by_strategy, heuristic=heuristic, gnn_scores=gnn_scores,
            question_entities=q_entity_details, timer=timer, notes=notes,
        )

    @staticmethod
    def _rerank(qgraph: QuestionGraph, corpus: "Corpus", scores: QueryScores, top_k: int,
                key: str) -> list[RetrievedPassage]:
        """Rank subgraph passages by graph/GNN score; remaining passages follow by hybrid score."""
        attr = "graph_score" if key == "graph" else "gnn_score"
        ranked = sorted(qgraph.passages.values(), key=lambda rp: (-(getattr(rp, attr) or 0.0), rp.chunk.id))
        out = []
        for i, rp in enumerate(ranked[:top_k], start=1):
            copy = RetrievedPassage(**{**rp.__dict__})
            copy.rank = i
            copy.final_score = float(getattr(rp, attr) or 0.0)
            out.append(copy)
        return out

    @staticmethod
    def full_ranking(result: ReasoningResult, corpus: "Corpus", strategy: str) -> list[str]:
        """Complete ranking of all corpus passages for a strategy (used by evaluation)."""
        scores = result.scores
        if strategy == "faiss":
            order = sorted(range(len(corpus.chunks)), key=lambda p: (-scores.semantic[p], p))
        elif strategy == "bm25":
            order = sorted(range(len(corpus.chunks)), key=lambda p: (-scores.bm25_raw[p], p))
        elif strategy == "hybrid":
            order = sorted(range(len(corpus.chunks)), key=lambda p: (-scores.hybrid[p], p))
        else:
            attr = "graph_score" if strategy == "graph" else "gnn_score"
            sub = sorted(result.qgraph.passages.values(), key=lambda rp: (-(getattr(rp, attr) or 0.0), rp.chunk.id))
            head = [rp.chunk.id for rp in sub]
            rest = [corpus.chunks[p].id for p in sorted(range(len(corpus.chunks)), key=lambda p: (-scores.hybrid[p], p))
                    if corpus.chunks[p].id not in result.qgraph.passages]
            return head + rest
        return [corpus.chunks[p].id for p in order]
