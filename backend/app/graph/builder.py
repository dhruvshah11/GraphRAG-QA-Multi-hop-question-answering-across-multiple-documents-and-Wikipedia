"""Graph construction: the corpus evidence graph and question subgraphs."""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from itertools import combinations
from typing import TYPE_CHECKING, Any, Optional

import networkx as nx

from app.core.domain import EntityMention, Relation, RetrievedPassage, TextChunk
from app.entities.registry import EntityRegistry
from app.graph.evidence_graph import QUESTION_NODE, EvidenceGraph, doc_node, passage_node

if TYPE_CHECKING:  # pragma: no cover
    from app.retrieval.corpus import Corpus
    from app.retrieval.retrievers import QueryScores


NON_BRIDGE_CATEGORIES = frozenset({"DATE"})  # years/dates link passages only by coincidence


def is_bridge_entity(entity: Any, max_df: int) -> bool:
    """An entity may connect two passages if it is specific enough and not a date."""
    return entity is not None and entity.category not in NON_BRIDGE_CATEGORIES and 2 <= len(entity.chunk_ids) <= max_df


def bridge_df_limit(total_chunks: int, ratio: float, absolute: int) -> int:
    """Max number of passages an entity may appear in to still act as a bridge."""
    return max(4, min(absolute, int(ratio * total_chunks)))


@dataclass
class QuestionGraph:
    """The question-specific subgraph plus the scored passages it contains."""

    graph: EvidenceGraph
    passages: dict[str, RetrievedPassage]  # chunk id -> scores (retrieved + expanded)
    question_entities: dict[str, float]  # entity id -> match score
    question_entity_mentions: list[dict[str, Any]] = field(default_factory=list)
    hop_distance: dict[str, int] = field(default_factory=dict)
    expansion_log: list[dict[str, Any]] = field(default_factory=list)


class GraphBuilder:
    def __init__(self, max_df_ratio: float = 0.25, max_df_abs: int = 40) -> None:
        self.max_df_ratio = max_df_ratio
        self.max_df_abs = max_df_abs

    # ============================================================ corpus graph
    def build_corpus_graph(
        self,
        chunks: list[TextChunk],
        documents: dict[str, dict[str, Any]],
        registry: EntityRegistry,
        mentions: list[EntityMention],
        relations: list[Relation],
    ) -> EvidenceGraph:
        g = EvidenceGraph()
        chunk_doc = {c.id: c.document_id for c in chunks}

        # Document and passage nodes, CONTAINS edges -------------------------
        for doc_id, meta in documents.items():
            g.add_node(doc_node(doc_id), "document", meta.get("name", doc_id),
                       document_id=doc_id, file_type=meta.get("file_type"), source_type=meta.get("source_type"),
                       url=meta.get("url"), num_chunks=meta.get("num_chunks", 0))
        for c in chunks:
            if doc_node(c.document_id) not in g.G:
                g.add_node(doc_node(c.document_id), "document", c.document_name, document_id=c.document_id,
                           source_type=c.source_type, url=c.url)
            g.add_node(passage_node(c.id), "passage", f"{c.document_name} · #{c.chunk_index + 1}",
                       chunk_id=c.id, document_id=c.document_id, document_name=c.document_name,
                       chunk_index=c.chunk_index, page_number=c.page_number, page_end=c.page_end,
                       section=c.section, token_count=c.token_count, preview=c.text[:280])
            g.add_edge(doc_node(c.document_id), passage_node(c.id), "CONTAINS")

        # Entity nodes ---------------------------------------------------------
        for e in registry.entities.values():
            g.add_node(e.id, "entity", e.name, category=e.category, ner_label=e.label,
                       mention_count=e.mention_count, num_passages=len(e.chunk_ids),
                       num_documents=len(e.document_ids), aliases=sorted(e.aliases)[:12],
                       specificity=round(registry.specificity(e.id), 4))

        # MENTIONS: passage -> entity ----------------------------------------
        mention_counts = Counter((m.chunk_id, m.entity_id) for m in mentions if m.entity_id)
        for (cid, eid), n in mention_counts.items():
            if passage_node(cid) in g.G and eid in g.G:
                surface = sorted({m.text for m in mentions if m.chunk_id == cid and m.entity_id == eid})[:3]
                g.add_edge(passage_node(cid), eid, "MENTIONS", weight=n, mentions=surface)

        # CO_OCCURS_WITH: entities in the same sentence ----------------------
        by_sentence: dict[tuple[str, int], set[str]] = defaultdict(set)
        for m in mentions:
            if m.entity_id:
                by_sentence[(m.chunk_id, m.sentence_index)].add(m.entity_id)
        co_counts: Counter[tuple[str, str]] = Counter()
        co_chunks: dict[tuple[str, str], set[str]] = defaultdict(set)
        for (cid, _), ents in by_sentence.items():
            for a, b in combinations(sorted(ents), 2):
                co_counts[(a, b)] += 1
                co_chunks[(a, b)].add(cid)
        for (a, b), n in co_counts.items():
            g.add_edge(a, b, "CO_OCCURS_WITH", weight=n, chunk_ids=sorted(co_chunks[(a, b)])[:8])

        # RELATED_TO: verb relations stated in a sentence ---------------------
        for r in relations:
            s = registry.alias_to_id.get(r.source_normalized)
            t = registry.alias_to_id.get(r.target_normalized)
            if not s or not t or s == t or s not in g.G or t not in g.G:
                continue
            r.source_id, r.target_id = s, t
            g.add_edge(s, t, "RELATED_TO", key=f"RELATED_TO:{r.relation}", relation=r.relation,
                       sentence=r.sentence[:400], chunk_id=r.chunk_id)

        # CONNECTS_TO: passages sharing a bridge entity -----------------------
        max_df = bridge_df_limit(len(chunks), self.max_df_ratio, self.max_df_abs)
        shared: dict[tuple[str, str], list[str]] = defaultdict(list)
        for e in registry.entities.values():
            if is_bridge_entity(e, max_df):
                for a, b in combinations(sorted(e.chunk_ids), 2):
                    shared[(a, b)].append(e.id)
        for (a, b), ent_ids in shared.items():
            if passage_node(a) not in g.G or passage_node(b) not in g.G:
                continue
            ent_ids.sort(key=lambda x: -registry.specificity(x))
            g.add_edge(passage_node(a), passage_node(b), "CONNECTS_TO",
                       weight=round(sum(registry.specificity(x) for x in ent_ids), 4),
                       via=ent_ids[:6], via_names=[registry.entities[x].name for x in ent_ids[:6]],
                       cross_document=chunk_doc.get(a) != chunk_doc.get(b), adjacent=False)

        # CONNECTS_TO: consecutive passages of the same document (explicit order)
        by_doc: dict[str, list[TextChunk]] = defaultdict(list)
        for c in chunks:
            by_doc[c.document_id].append(c)
        for doc_chunks in by_doc.values():
            doc_chunks.sort(key=lambda c: c.chunk_index)
            for prev, nxt in zip(doc_chunks, doc_chunks[1:]):
                u, v = passage_node(prev.id), passage_node(nxt.id)
                if g.G.has_edge(u, v, "CONNECTS_TO"):
                    g.G.edges[u, v, "CONNECTS_TO"]["adjacent"] = True
                else:
                    g.add_edge(u, v, "CONNECTS_TO", weight=0.3, via=[], via_names=[],
                               cross_document=False, adjacent=True)
        return g

    # ======================================================== question graph
    def build_question_graph(
        self,
        corpus: "Corpus",
        scores: "QueryScores",
        question: str,
        question_entities: dict[str, float],
        seeds: list[RetrievedPassage],
        max_hops: int = 3,
        per_hop: int = 12,
        question_entity_mentions: Optional[list[dict[str, Any]]] = None,
    ) -> QuestionGraph:
        registry = corpus.registry
        max_df = bridge_df_limit(len(corpus.chunks), self.max_df_ratio, self.max_df_abs)
        included: dict[str, RetrievedPassage] = {}
        expansion_log: list[dict[str, Any]] = []

        for rp in seeds:
            rp.hop = 1
            included[rp.chunk.id] = rp

        # Hop 1 (continued): passages that mention a question entity.
        entity_passages: dict[str, tuple[float, str]] = {}
        for eid, match_score in question_entities.items():
            entity = registry.get(eid)
            if not entity:
                continue
            for cid in entity.chunk_ids:
                if cid not in included and cid in corpus.pos_by_chunk:
                    score = float(scores.hybrid[corpus.pos_by_chunk[cid]]) + 0.1 * match_score
                    if cid not in entity_passages or entity_passages[cid][0] < score:
                        entity_passages[cid] = (score, eid)
        for cid, (score, eid) in sorted(entity_passages.items(), key=lambda kv: -kv[1][0])[:per_hop]:
            rp = scores.passage_by_id(cid, origin="entity_match", hop=1)
            included[cid] = rp
            expansion_log.append({"chunk_id": cid, "hop": 1, "via": eid, "reason": "mentions question entity"})

        # Hops 2..max_hops: follow bridge entities and document adjacency.
        frontier = list(included)
        for hop in range(2, max_hops + 1):
            candidates: dict[str, tuple[float, Optional[str], str]] = {}
            for cid in frontier:
                for eid in corpus.entity_ids_by_chunk.get(cid, ()):
                    entity = registry.get(eid)
                    if not is_bridge_entity(entity, max_df):
                        continue
                    spec = registry.specificity(eid)
                    for other in entity.chunk_ids:
                        if other in included or other not in corpus.pos_by_chunk:
                            continue
                        score = float(scores.hybrid[corpus.pos_by_chunk[other]]) + 0.2 * spec
                        if other not in candidates or candidates[other][0] < score:
                            candidates[other] = (score, eid, cid)
                for other in corpus.adjacent_chunks(cid):
                    if other in included:
                        continue
                    score = float(scores.hybrid[corpus.pos_by_chunk[other]])
                    if other not in candidates or candidates[other][0] < score:
                        candidates[other] = (score, None, cid)
            if not candidates:
                break
            ranked = sorted(candidates.items(), key=lambda kv: (-kv[1][0], kv[0]))[:per_hop]
            frontier = []
            for cid, (score, eid, src) in ranked:
                included[cid] = scores.passage_by_id(cid, origin="graph_expansion", hop=hop)
                frontier.append(cid)
                expansion_log.append({"chunk_id": cid, "hop": hop, "via": eid or "adjacent", "from": src,
                                      "reason": "shares bridge entity" if eid else "adjacent passage"})

        graph = self._assemble(corpus, question, question_entities, included, seeds)
        # BFS hop distance from the question node (undirected)
        und = graph.G.to_undirected(as_view=True)
        hop_distance = dict(nx.single_source_shortest_path_length(und, QUESTION_NODE))
        return QuestionGraph(
            graph=graph,
            passages=included,
            question_entities=question_entities,
            question_entity_mentions=question_entity_mentions or [],
            hop_distance=hop_distance,
            expansion_log=expansion_log,
        )

    def _assemble(
        self,
        corpus: "Corpus",
        question: str,
        question_entities: dict[str, float],
        included: dict[str, RetrievedPassage],
        seeds: list[RetrievedPassage],
    ) -> EvidenceGraph:
        full = corpus.graph
        registry = corpus.registry
        g = EvidenceGraph()
        g.add_node(QUESTION_NODE, "question", question)

        included_nodes = {passage_node(cid) for cid in included}
        # entities to keep: question entities, bridges, and the most specific per passage
        mention_count: Counter[str] = Counter()
        for cid in included:
            for eid in corpus.entity_ids_by_chunk.get(cid, ()):
                mention_count[eid] += 1
        keep: set[str] = {eid for eid in question_entities if eid in full.G}
        keep |= {eid for eid, n in mention_count.items() if n >= 2}
        for cid in included:
            ents = sorted(corpus.entity_ids_by_chunk.get(cid, ()), key=lambda e: -registry.specificity(e))
            keep |= set(ents[:4])

        for cid, rp in included.items():
            c = rp.chunk
            dn = doc_node(c.document_id)
            if dn not in g.G:
                meta = full.G.nodes[dn] if dn in full.G else {"label": c.document_name}
                g.add_node(dn, "document", meta.get("label", c.document_name),
                           **{k: v for k, v in meta.items() if k not in ("type", "label")})
            pn = passage_node(cid)
            meta = full.G.nodes[pn] if pn in full.G else {}
            g.add_node(pn, "passage", meta.get("label", f"{c.document_name} · #{c.chunk_index + 1}"),
                       **{k: v for k, v in meta.items() if k not in ("type", "label")},
                       origin=rp.origin, hop=rp.hop, hybrid_score=round(rp.hybrid_score, 4),
                       semantic_score=round(rp.semantic_score, 4), bm25_score=round(rp.bm25_score, 4))
            g.add_edge(dn, pn, "CONTAINS")
        for eid in keep:
            if eid in full.G:
                d = full.G.nodes[eid]
                g.add_node(eid, "entity", d.get("label", eid), **{k: v for k, v in d.items() if k not in ("type", "label")},
                           is_question_entity=eid in question_entities)

        # induced edges from the corpus graph (only edges whose evidence is in the subgraph)
        included_ids = set(included)
        for u in list(g.G.nodes):
            if u not in full.G:
                continue
            for _, v, k, d in full.G.out_edges(u, keys=True, data=True):
                etype = d.get("type")
                if v not in g.G or etype == "CONTAINS":
                    continue
                if etype == "CO_OCCURS_WITH" and not set(d.get("chunk_ids", [])) & included_ids:
                    continue
                if etype == "RELATED_TO" and d.get("chunk_id") not in included_ids:
                    continue
                g.G.add_edge(u, v, key=k, **d)

        for rp in seeds:
            g.add_edge(passage_node(rp.chunk.id), QUESTION_NODE, "SUPPORTS", weight=max(rp.hybrid_score, 1e-3),
                       rank=rp.rank)
        for eid, score in question_entities.items():
            if eid in g.G:
                g.add_edge(QUESTION_NODE, eid, "MENTIONS", weight=score)
        return g
