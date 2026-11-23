"""Evidence graph construction, serialization and multi-hop traversal."""
from __future__ import annotations

from app.gnn.dataset import gold_chunks_for_evidence
from app.graph.evidence_graph import EDGE_TYPES, NODE_TYPES, QUESTION_NODE, EvidenceGraph, passage_node


def test_graph_nodes(demo_corpus):
    g = demo_corpus.graph
    stats = g.stats()
    assert stats["node_types"]["document"] == 5
    assert stats["node_types"]["passage"] == len(demo_corpus.chunks)
    assert stats["node_types"]["entity"] == len(demo_corpus.registry.entities)
    assert "ent:geoffrey_hinton" in g.G
    for n, d in g.G.nodes(data=True):
        assert d["type"] in NODE_TYPES and n.split(":", 1)[0] in ("doc", "psg", "ent")


def test_graph_edges_are_typed_and_evidence_based(demo_corpus):
    g = demo_corpus.graph
    types = {d["type"] for _, _, d in g.G.edges(data=True)}
    assert types <= set(EDGE_TYPES)
    contains = [e for e in g.G.edges(data=True) if e[2]["type"] == "CONTAINS"]
    assert len(contains) == len(demo_corpus.chunks)
    for u, v, d in g.G.edges(data=True):
        if d["type"] == "MENTIONS":
            assert g.G.nodes[u]["type"] == "passage" and g.G.nodes[v]["type"] == "entity"
        if d["type"] == "RELATED_TO":
            assert d["sentence"] and d["relation"]  # every relation carries its source sentence
    # cross-document connection through a shared entity (Hinton: Nobel PDF <-> profile TXT)
    nobel = next(c.id for c in demo_corpus.chunks if c.document_name.startswith("nobel"))
    hinton = next(c.id for c in demo_corpus.chunks if c.document_name.startswith("geoffrey"))
    edges = g.edges_between(passage_node(nobel), passage_node(hinton))
    assert any(e["type"] == "CONNECTS_TO" and e["cross_document"] and "ent:geoffrey_hinton" in e["via"] for e in edges)


def test_graph_serialization_roundtrip(demo_corpus, tmp_path):
    g = demo_corpus.graph
    g.save_json(tmp_path / "g.json")
    loaded = EvidenceGraph.load_json(tmp_path / "g.json")
    assert loaded.stats() == g.stats()
    g.save_graphml(tmp_path / "g.graphml")
    assert (tmp_path / "g.graphml").stat().st_size > 1000


def test_question_subgraph_and_multihop_chain(container, demo_corpus, demo_questions):
    q = demo_questions["demo-01"]
    res = container.reasoner.analyze(demo_corpus, q["question"], top_k=4, max_hops=3, use_gnn=False)
    G = res.qgraph.graph.G
    assert QUESTION_NODE in G
    assert any(d["type"] == "SUPPORTS" for _, _, d in G.edges(data=True))
    gold = gold_chunks_for_evidence(demo_corpus, q["gold_evidence"], q["gold_documents"])
    best = res.chains[0]
    assert set(best.chunk_ids) == gold, "top chain should connect the Nobel PDF and the Hinton profile"
    assert best.steps[1].via_entity == "ent:geoffrey_hinton"
    for chain in res.chains:
        assert len(chain.chunk_ids) == len(set(chain.chunk_ids))  # no cycles
        assert chain.length <= 3
        assert all(s.reason for s in chain.steps)


def test_max_hops_limits_expansion(container, demo_corpus):
    res = container.reasoner.analyze(demo_corpus, "Who founded DeepMind?", top_k=2, max_hops=1, use_gnn=False)
    assert all(rp.hop == 1 for rp in res.qgraph.passages.values())
    assert all(c.length == 1 for c in res.chains)


def test_graph_heuristic_scores(container, demo_corpus):
    res = container.reasoner.analyze(demo_corpus, "Which company acquired DeepMind?", top_k=4, max_hops=2,
                                     use_gnn=False)
    assert res.rankings["graph"]
    for cid, h in res.heuristic.items():
        assert 0 <= h["ppr"] <= 1 and 0 <= h["chain_support"] <= 1
