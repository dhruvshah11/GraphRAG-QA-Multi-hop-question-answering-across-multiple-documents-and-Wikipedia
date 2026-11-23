"""Deterministic node positions for the React Flow graph explorer.

Layouts are computed on the server with NetworkX (seeded), so the explorer
always shows the same picture for the same graph.
"""
from __future__ import annotations

import math
from typing import Optional

import networkx as nx

from app.core.domain import EvidenceChain
from app.graph.evidence_graph import QUESTION_NODE, EvidenceGraph, doc_node, passage_node

SCALE = 900.0


def force_layout(graph: EvidenceGraph, center: Optional[str] = None, seed: int = 42) -> dict[str, tuple[float, float]]:
    G = graph.G
    if G.number_of_nodes() == 0:
        return {}
    H = graph.to_weighted_undirected()
    n = H.number_of_nodes()
    k = 2.2 / math.sqrt(max(n, 1))
    if center and center in H:
        pos = nx.spring_layout(H, seed=seed, k=k, iterations=120, pos={center: (0.0, 0.0)}, fixed=[center],
                               weight=None)
    else:
        pos = nx.spring_layout(H, seed=seed, k=k, iterations=120, weight=None)
    scale = SCALE * max(1.0, math.sqrt(n) / 6)
    return {node: (float(x) * scale, float(y) * scale) for node, (x, y) in pos.items()}


def chain_layout(graph: EvidenceGraph, chain: EvidenceChain) -> dict[str, tuple[float, float]]:
    """Left-to-right layout: question -> passage -> bridge entity -> passage ..."""
    pos: dict[str, tuple[float, float]] = {QUESTION_NODE: (0.0, 0.0)}
    x = 0.0
    for i, step in enumerate(chain.steps):
        if i > 0 and step.via_entity and step.via_entity in graph.G:
            x += 260.0
            pos[step.via_entity] = (x, 150.0)
        x += 260.0
        pn = passage_node(step.chunk_id)
        pos[pn] = (x, 0.0)
        doc = graph.G.nodes[pn].get("document_id") if pn in graph.G else None
        if doc and doc_node(doc) in graph.G and doc_node(doc) not in pos:
            pos[doc_node(doc)] = (x, -170.0)
    return pos
