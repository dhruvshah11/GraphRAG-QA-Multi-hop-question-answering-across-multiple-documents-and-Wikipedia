"""Non-neural graph reranking baseline (always available, no training needed).

    graph_score(p) = l_h * hybrid(p) + l_p * PPR(p) / max PPR + l_c * chain_support(p)

* PPR = Personalized PageRank on the question subgraph, restarting at the
  question node (damping 0.85). Edges are weighted by type; edges that touch
  an entity are scaled by the entity's specificity so that generic "hub"
  entities (years, countries) spread little relevance.
* chain_support(p) = best score of an evidence chain containing p, divided by
  the best chain score overall.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import networkx as nx

from app.core.domain import EvidenceChain
from app.graph.builder import QuestionGraph
from app.graph.evidence_graph import QUESTION_NODE, passage_node

if TYPE_CHECKING:  # pragma: no cover
    from app.retrieval.corpus import Corpus

EDGE_TYPE_WEIGHTS = {
    "SUPPORTS": 2.0,
    "MENTIONS": 1.0,
    "RELATED_TO": 1.0,
    "CO_OCCURS_WITH": 0.5,
    "CONNECTS_TO": 0.5,
    "CONTAINS": 0.2,
}


@dataclass
class HeuristicWeights:
    hybrid: float = 0.5
    ppr: float = 0.25
    chain: float = 0.25
    damping: float = 0.85


class GraphHeuristicRanker:
    def __init__(self, weights: HeuristicWeights | None = None) -> None:
        self.w = weights or HeuristicWeights()

    def personalized_pagerank(self, qg: QuestionGraph, corpus: "Corpus") -> dict[str, float]:
        H = qg.graph.to_weighted_undirected(EDGE_TYPE_WEIGHTS)
        for u, v, d in H.edges(data=True):
            for node in (u, v):
                if H.nodes[node].get("type") == "entity":
                    d["weight"] *= max(0.05, corpus.registry.specificity(node))
        if QUESTION_NODE not in H or H.number_of_edges() == 0:
            return {}
        try:
            return nx.pagerank(H, alpha=self.w.damping, personalization={QUESTION_NODE: 1.0}, weight="weight",
                               max_iter=200, tol=1e-8)
        except nx.PowerIterationFailedConvergence:  # pragma: no cover - tiny graphs converge
            return {}

    def score(self, qg: QuestionGraph, corpus: "Corpus", chains: list[EvidenceChain]) -> dict[str, dict[str, float]]:
        """Return ``{chunk_id: {"graph_score", "ppr", "chain_support"}}``."""
        ppr = self.personalized_pagerank(qg, corpus)
        passage_ppr = {cid: ppr.get(passage_node(cid), 0.0) for cid in qg.passages}
        max_ppr = max(passage_ppr.values(), default=0.0) or 1.0
        best_chain = max((c.score for c in chains), default=0.0)
        support: dict[str, float] = {}
        for chain in chains:
            for cid in chain.chunk_ids:
                support[cid] = max(support.get(cid, 0.0), chain.score)
        out: dict[str, dict[str, float]] = {}
        for cid, rp in qg.passages.items():
            ppr_norm = passage_ppr[cid] / max_ppr
            chain_support = (support.get(cid, 0.0) / best_chain) if best_chain > 0 else 0.0
            graph_score = self.w.hybrid * rp.hybrid_score + self.w.ppr * ppr_norm + self.w.chain * chain_support
            out[cid] = {"graph_score": graph_score, "ppr": ppr_norm, "chain_support": chain_support}
        return out
