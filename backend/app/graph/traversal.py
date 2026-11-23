"""Multi-hop evidence-chain search over the question subgraph.

An *evidence chain* is a simple path ``question -> p1 -> p2 -> ... -> pn`` in
the passage graph, where ``p1`` was retrieved (or mentions a question entity)
and consecutive passages share a bridge entity (or are adjacent in the same
document). Paths are enumerated with ``networkx.all_simple_paths`` (simple
paths never repeat a node, which prevents cycles) with ``cutoff = max_hops``.

Chain score (all terms in [0, 1]):

    score(C) = w_r * anchored_rel(C)            # 0.7 rel(p1) + 0.3 mean rel(p2..pn)
             + w_c * coverage(C)                # share of question terms covered
             + w_b * mean bridge specificity    # only for chains with > 1 passage
             - w_l * (|C| - 1)                  # length penalty

Pruning: a hop is kept only if it adds at least one new question term (an
uninformative hop cannot improve the evidence), passages need a minimum
relevance, and chains over the same passage set are de-duplicated.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import islice
from typing import TYPE_CHECKING, Optional

import networkx as nx

from app.core.domain import ChainStep, EvidenceChain, RetrievedPassage
from app.graph.builder import QuestionGraph
from app.graph.evidence_graph import QUESTION_NODE, chunk_id_of, passage_node
from app.preprocessing.tokenize import content_terms

if TYPE_CHECKING:  # pragma: no cover
    from app.retrieval.corpus import Corpus


@dataclass
class ChainWeights:
    relevance: float = 0.45
    coverage: float = 0.40
    bridge: float = 0.15
    length_penalty: float = 0.03
    min_relevance: float = 0.05
    max_neighbors: int = 6
    max_paths: int = 5000


def anchored_relevance(rels: list[float]) -> float:
    """Relevance of a chain: the entry passage counts most.

    A bridge passage is often only weakly similar to the question (it holds the
    *second* fact), so averaging would punish exactly the hops multi-hop
    questions need.
    """
    if len(rels) == 1:
        return rels[0]
    return 0.7 * rels[0] + 0.3 * sum(rels[1:]) / (len(rels) - 1)


class GraphTraversal:
    def __init__(self, weights: Optional[ChainWeights] = None) -> None:
        self.w = weights or ChainWeights()

    def question_terms(self, question: str) -> list[str]:
        return list(dict.fromkeys(content_terms(question, drop_question_words=True)))

    def _passage_graph(self, qg: QuestionGraph, corpus: "Corpus") -> nx.Graph:
        pg = nx.Graph()
        pg.add_node(QUESTION_NODE)
        for cid, rp in qg.passages.items():
            pg.add_node(cid)
            if rp.hop == 1:
                pg.add_edge(QUESTION_NODE, cid, bridge=None, bridge_name=None, spec=0.0)
        # CONNECTS_TO edges between included passages
        candidates: dict[str, list[tuple[float, str, Optional[str], Optional[str], float]]] = {}
        for u, v, d in qg.graph.G.edges(data=True):
            if d.get("type") != "CONNECTS_TO":
                continue
            a, b = chunk_id_of(u), chunk_id_of(v)
            if a not in qg.passages or b not in qg.passages:
                continue
            via = d.get("via") or []
            if via:
                # pick the shared entity that is best anchored to the question
                def quality(eid: str) -> float:
                    return 0.5 * corpus.registry.specificity(eid) + 0.5 * self._anchor(qg, eid)
                best = max(via, key=quality)
                names = dict(zip(via, d.get("via_names") or []))
                bridge, name, spec = best, names.get(best), quality(best)
            else:
                bridge, name, spec = None, None, 0.3  # adjacent passages of one document
            for x, y in ((a, b), (b, a)):
                priority = qg.passages[y].hybrid_score + 0.5 * spec
                candidates.setdefault(x, []).append((priority, y, bridge, name, spec))
        for x, lst in candidates.items():
            lst.sort(key=lambda t: -t[0])
            for _, y, bridge, name, spec in lst[: self.w.max_neighbors]:
                if not pg.has_edge(x, y):
                    pg.add_edge(x, y, bridge=bridge, bridge_name=name, spec=spec)
        return pg

    @staticmethod
    def _anchor(qg: QuestionGraph, entity_id: str) -> float:
        """How strongly an entity is tied to the question: 1 if it *is* a question
        entity, else the best match score of a question entity it co-occurs with
        (same sentence) or is related to; 0 otherwise."""
        if entity_id in qg.question_entities:
            return qg.question_entities[entity_id]
        G = qg.graph.G
        if entity_id not in G:
            return 0.0
        best = 0.0
        for other in set(G.successors(entity_id)) | set(G.predecessors(entity_id)):
            if other in qg.question_entities:
                types = {d.get("type") for d in (G.get_edge_data(entity_id, other) or {}).values()}
                types |= {d.get("type") for d in (G.get_edge_data(other, entity_id) or {}).values()}
                if types & {"CO_OCCURS_WITH", "RELATED_TO"}:
                    best = max(best, qg.question_entities[other])
        return best

    def find_chains(self, qg: QuestionGraph, corpus: "Corpus", question: str, max_hops: int = 3,
                    max_chains: int = 8) -> list[EvidenceChain]:
        if not qg.passages:
            return []
        q_terms = self.question_terms(question)
        q_term_set = set(q_terms)
        idf = getattr(getattr(corpus.bm25, "_bm25", None), "idf", {}) or {}
        max_idf = max([idf.get(t, 0.0) for t in q_terms] + [1e-6])
        # rare question terms matter more than generic ones ("nobel" vs "company")
        term_weight = {t: max(0.15, idf.get(t, max_idf) / max_idf) for t in q_terms}
        passage_terms = {cid: set(content_terms(rp.chunk.text)) & q_term_set for cid, rp in qg.passages.items()}
        q_entity_names = {
            eid: (corpus.registry.get(eid).name if corpus.registry.get(eid) else eid) for eid in qg.question_entities
        }
        pg = self._passage_graph(qg, corpus)
        targets = [cid for cid in qg.passages]
        best: dict[frozenset[str], EvidenceChain] = {}
        paths = nx.all_simple_paths(pg, QUESTION_NODE, target=targets, cutoff=max_hops)
        for path in islice(paths, self.w.max_paths):
            chain = self._score_path(path[1:], pg, qg, corpus, q_terms, passage_terms, q_entity_names, term_weight)
            if chain is None:
                continue
            key = frozenset(chain.chunk_ids)
            if key not in best or best[key].score < chain.score:
                best[key] = chain
        chains = sorted(best.values(), key=lambda c: (-c.score, c.length))
        return chains[:max_chains]

    def rescore(self, chains: list[EvidenceChain], relevance: dict[str, float]) -> list[EvidenceChain]:
        """Re-rank chains using another passage relevance (e.g. GNN probabilities)."""
        out = []
        for c in chains:
            rels = [float(relevance.get(s.chunk_id, 0.0)) for s in c.steps]
            n = len(rels)
            avg = anchored_relevance(rels)
            score = (
                self.w.relevance * avg
                + self.w.coverage * c.components["coverage"]
                + (self.w.bridge * c.components["bridge_specificity"] if n > 1 else 0.0)
                - self.w.length_penalty * (n - 1)
            )
            steps = [ChainStep(s.chunk_id, s.via_entity, s.via_entity_name, round(r, 4), s.reason)
                     for s, r in zip(c.steps, rels)]
            out.append(EvidenceChain(steps=steps, score=score, components={**c.components, "avg_relevance": avg},
                                     covered_terms=c.covered_terms, missing_terms=c.missing_terms))
        return sorted(out, key=lambda c: (-c.score, c.length))

    def _score_path(self, cids: list[str], pg: nx.Graph, qg: QuestionGraph, corpus: "Corpus", q_terms: list[str],
                    passage_terms: dict[str, set[str]], q_entity_names: dict[str, str],
                    term_weight: Optional[dict[str, float]] = None) -> Optional[EvidenceChain]:
        covered: set[str] = set()
        steps: list[ChainStep] = []
        rels: list[float] = []
        bridges: list[float] = []
        for i, cid in enumerate(cids):
            rp: RetrievedPassage = qg.passages[cid]
            if rp.hybrid_score < self.w.min_relevance:
                return None
            new_terms = passage_terms[cid] - covered
            if i > 0 and not new_terms:
                return None  # hop adds no new question information -> prune
            covered |= passage_terms[cid]
            rels.append(rp.hybrid_score)
            if i == 0:
                via, via_name = None, None
                q_ents = [q_entity_names[e] for e in qg.question_entities
                          if cid in (corpus.registry.get(e).chunk_ids if corpus.registry.get(e) else set())]
                if rp.origin == "retrieval":
                    reason = f"Retrieved directly (rank {rp.rank}, hybrid score {rp.hybrid_score:.2f})"
                else:
                    reason = f"Mentions a question entity (hybrid score {rp.hybrid_score:.2f})"
                if q_ents:
                    reason += "; mentions question entit" + ("ies " if len(q_ents) > 1 else "y ") + ", ".join(
                        f"'{n}'" for n in q_ents[:3])
            else:
                edge = pg.edges[cids[i - 1], cid]
                via, via_name = edge.get("bridge"), edge.get("bridge_name")
                bridges.append(float(edge.get("spec", 0.0)))
                if via:
                    reason = f"Linked to step {i} through the shared entity '{via_name}'"
                else:
                    reason = f"Next passage of the same document as step {i}"
                reason += f"; relevance {rp.hybrid_score:.2f}"
            if new_terms:
                reason += "; adds question terms: " + ", ".join(sorted(new_terms)[:6])
            steps.append(ChainStep(chunk_id=cid, via_entity=via, via_entity_name=via_name,
                                   relevance=round(rp.hybrid_score, 4), reason=reason))
        n = len(cids)
        term_weight = term_weight or {t: 1.0 for t in q_terms}
        total_w = sum(term_weight[t] for t in q_terms)
        coverage = sum(term_weight[t] for t in covered) / total_w if total_w else 0.0
        avg_rel = anchored_relevance(rels)
        avg_bridge = sum(bridges) / len(bridges) if bridges else 0.0
        score = (
            self.w.relevance * avg_rel
            + self.w.coverage * coverage
            + (self.w.bridge * avg_bridge if n > 1 else 0.0)
            - self.w.length_penalty * (n - 1)
        )
        return EvidenceChain(
            steps=steps,
            score=score,
            components={"avg_relevance": avg_rel, "coverage": coverage, "bridge_specificity": avg_bridge,
                        "length": float(n)},
            covered_terms=sorted(covered),
            missing_terms=sorted(set(q_terms) - covered),
        )


def chain_highlight(chain: EvidenceChain) -> tuple[set[str], set[tuple[str, str]]]:
    """Node ids and (undirected) edges that visualize a chain in the subgraph."""
    nodes: set[str] = {QUESTION_NODE}
    edges: set[tuple[str, str]] = set()
    prev: Optional[str] = None
    for i, step in enumerate(chain.steps):
        node = passage_node(step.chunk_id)
        nodes.add(node)
        if i == 0:
            edges.add((node, QUESTION_NODE))
        elif prev is not None:
            if step.via_entity:
                nodes.add(step.via_entity)
                edges.add((prev, step.via_entity))
                edges.add((node, step.via_entity))
            edges.add((prev, node))
        prev = node
    return nodes, edges
