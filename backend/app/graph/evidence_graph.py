"""Heterogeneous evidence graph (NetworkX ``MultiDiGraph``) and serialization.

Node types : question, document, passage, entity
Edge types : CONTAINS       document -> passage
             MENTIONS       passage  -> entity   (also question -> entity)
             RELATED_TO     entity   -> entity   (verb relation stated in one sentence)
             CO_OCCURS_WITH entity   -> entity   (same sentence)
             SUPPORTS       passage  -> question (passage retrieved for the question)
             CONNECTS_TO    passage  -> passage  (shared bridge entity / adjacent chunks)

Every edge is derived from extracted text or explicit metadata; nothing is
invented. Node ids are stable: ``doc:<document id>``, ``psg:<chunk id>``,
``ent:<normalized name>``, ``q:question``.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Optional

import networkx as nx

NODE_TYPES = ("question", "document", "passage", "entity")
EDGE_TYPES = ("CONTAINS", "MENTIONS", "RELATED_TO", "CO_OCCURS_WITH", "SUPPORTS", "CONNECTS_TO")
QUESTION_NODE = "q:question"


def doc_node(doc_id: str) -> str:
    return f"doc:{doc_id}"


def passage_node(chunk_id: str) -> str:
    return f"psg:{chunk_id}"


def chunk_id_of(node_id: str) -> str:
    return node_id[4:] if node_id.startswith("psg:") else node_id


class EvidenceGraph:
    def __init__(self, graph: Optional[nx.MultiDiGraph] = None) -> None:
        self.G: nx.MultiDiGraph = graph if graph is not None else nx.MultiDiGraph()

    # --------------------------------------------------------------- building
    def add_node(self, node_id: str, node_type: str, label: str, **attrs: Any) -> None:
        if node_type not in NODE_TYPES:
            raise ValueError(f"Unknown node type {node_type}")
        if node_id in self.G:
            self.G.nodes[node_id].update(attrs)
        else:
            self.G.add_node(node_id, type=node_type, label=label, **attrs)

    def add_edge(self, u: str, v: str, edge_type: str, weight: float = 1.0, key: Optional[str] = None,
                 **attrs: Any) -> None:
        if edge_type not in EDGE_TYPES:
            raise ValueError(f"Unknown edge type {edge_type}")
        if u not in self.G or v not in self.G:
            raise KeyError(f"Both endpoints must exist before adding an edge ({u} -> {v})")
        key = key or edge_type
        if self.G.has_edge(u, v, key):
            data = self.G.edges[u, v, key]
            data["weight"] = data.get("weight", 0.0) + weight
            data.update({k: v_ for k, v_ in attrs.items() if k not in data})
        else:
            self.G.add_edge(u, v, key=key, type=edge_type, weight=float(weight), **attrs)

    # ---------------------------------------------------------------- queries
    def nodes_of_type(self, node_type: str) -> list[str]:
        return [n for n, d in self.G.nodes(data=True) if d.get("type") == node_type]

    def node(self, node_id: str) -> dict[str, Any]:
        return dict(self.G.nodes[node_id])

    def neighbors(self, node_id: str) -> list[str]:
        return list(dict.fromkeys([*self.G.successors(node_id), *self.G.predecessors(node_id)]))

    def edges_between(self, u: str, v: str) -> list[dict[str, Any]]:
        out = []
        for a, b in ((u, v), (v, u)):
            if self.G.has_edge(a, b):
                out.extend(dict(d, source=a, target=b) for d in self.G.get_edge_data(a, b).values())
        return out

    @property
    def num_nodes(self) -> int:
        return self.G.number_of_nodes()

    @property
    def num_edges(self) -> int:
        return self.G.number_of_edges()

    def stats(self) -> dict[str, Any]:
        node_types = Counter(d.get("type") for _, d in self.G.nodes(data=True))
        edge_types = Counter(d.get("type") for _, _, d in self.G.edges(data=True))
        return {
            "nodes": self.num_nodes,
            "edges": self.num_edges,
            "node_types": {t: node_types.get(t, 0) for t in NODE_TYPES},
            "edge_types": {t: edge_types.get(t, 0) for t in EDGE_TYPES},
        }

    def to_weighted_undirected(self, type_weights: Optional[dict[str, float]] = None) -> nx.Graph:
        """Collapse to a simple undirected graph, summing (type-scaled) weights."""
        type_weights = type_weights or {}
        H = nx.Graph()
        H.add_nodes_from(self.G.nodes(data=True))
        for u, v, d in self.G.edges(data=True):
            w = float(d.get("weight", 1.0)) * type_weights.get(d.get("type"), 1.0)
            if w <= 0:
                continue
            if H.has_edge(u, v):
                H[u][v]["weight"] += w
            else:
                H.add_edge(u, v, weight=w)
        return H

    def subgraph(self, nodes: Iterable[str]) -> "EvidenceGraph":
        return EvidenceGraph(self.G.subgraph(list(nodes)).copy())

    def prune(self, remove_isolated: bool = True, max_entity_degree: Optional[int] = None) -> int:
        """Graph pruning: drop isolated nodes and (optionally) hub entities."""
        to_remove = []
        for n, d in self.G.nodes(data=True):
            degree = self.G.degree(n)
            if remove_isolated and degree == 0 and d.get("type") != "question":
                to_remove.append(n)
            elif max_entity_degree and d.get("type") == "entity" and degree > max_entity_degree:
                to_remove.append(n)
        self.G.remove_nodes_from(to_remove)
        return len(to_remove)

    # ---------------------------------------------------------- serialization
    def to_dict(self, positions: Optional[dict[str, tuple[float, float]]] = None,
                highlight_nodes: Optional[set[str]] = None,
                highlight_edges: Optional[set[tuple[str, str]]] = None) -> dict[str, Any]:
        highlight_nodes = highlight_nodes or set()
        highlight_edges = highlight_edges or set()
        nodes = []
        for n, d in self.G.nodes(data=True):
            item = {
                "id": n,
                "type": d.get("type"),
                "label": d.get("label", n),
                "data": {k: v for k, v in d.items() if k not in ("type", "label")},
                "highlighted": n in highlight_nodes,
            }
            if positions and n in positions:
                item["position"] = {"x": round(float(positions[n][0]), 2), "y": round(float(positions[n][1]), 2)}
            nodes.append(item)
        edges = []
        for u, v, k, d in self.G.edges(keys=True, data=True):
            edges.append(
                {
                    "id": f"{u}|{v}|{k}",
                    "source": u,
                    "target": v,
                    "type": d.get("type"),
                    "weight": round(float(d.get("weight", 1.0)), 4),
                    "data": {kk: vv for kk, vv in d.items() if kk not in ("type", "weight")},
                    "highlighted": (u, v) in highlight_edges or (v, u) in highlight_edges,
                }
            )
        return {"nodes": nodes, "edges": edges, "stats": self.stats()}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvidenceGraph":
        g = cls()
        for n in data.get("nodes", []):
            g.G.add_node(n["id"], type=n["type"], label=n.get("label", n["id"]), **n.get("data", {}))
        for e in data.get("edges", []):
            key = e["id"].split("|", 2)[-1] if "|" in e["id"] else e["type"]
            g.G.add_edge(e["source"], e["target"], key=key, type=e["type"], weight=e.get("weight", 1.0),
                         **e.get("data", {}))
        return g

    def save_json(self, path: Path, **kwargs: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(**kwargs)), encoding="utf-8")

    @classmethod
    def load_json(cls, path: Path) -> "EvidenceGraph":
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def save_graphml(self, path: Path) -> None:
        """GraphML export (list/dict attributes are stored as JSON strings)."""
        H = nx.MultiDiGraph()
        for n, d in self.G.nodes(data=True):
            H.add_node(n, **{k: (json.dumps(v) if isinstance(v, (list, dict, set, tuple)) else v)
                             for k, v in d.items() if v is not None})
        for u, v, k, d in self.G.edges(keys=True, data=True):
            H.add_edge(u, v, key=k, **{kk: (json.dumps(vv) if isinstance(vv, (list, dict, set, tuple)) else vv)
                                       for kk, vv in d.items() if vv is not None})
        path.parent.mkdir(parents=True, exist_ok=True)
        nx.write_graphml(H, str(path))
