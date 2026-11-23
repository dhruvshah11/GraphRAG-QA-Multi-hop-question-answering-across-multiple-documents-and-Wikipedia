"""Conversion of a question subgraph into a PyTorch Geometric ``Data`` object.

Node feature vector (dimension = embedding_dim + 8):

* ``e(v) * q``   element-wise product of the node's text embedding and the
                 question embedding (question-aware semantic signal)
* 8 scalars      [cosine(e(v), q), BM25 (norm.), hybrid score, is_seed,
                  hop distance from question / 6, is_question_entity,
                  log-degree (norm.), entity specificity]

Experiment (not adopted): adding the graph-heuristic score as a 9th scalar
("v2") gave a lower best validation MRR (0.922) than this specification
(0.928), so v1 is kept. The heuristic score is still stored on each graph
(``data.graph``) so training reports it as a second validation baseline.

Node text embeddings: question -> question embedding; passage -> passage
embedding from the index; entity -> embedding of its canonical name;
document -> mean of its passages' embeddings in the subgraph.

A learned node-type embedding is added inside the model; edge types are
one-hot encoded as ``edge_attr`` and used by the attention of GATConv.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING, Optional

import numpy as np
import torch
from torch_geometric.data import Data

from app.graph.builder import QuestionGraph
from app.graph.evidence_graph import EDGE_TYPES, chunk_id_of

if TYPE_CHECKING:  # pragma: no cover
    from app.embeddings.embedder import EmbeddingGenerator
    from app.retrieval.corpus import Corpus

FEATURE_SPEC_VERSION = "gat-features-v1"
NODE_TYPE_IDS = {"question": 0, "document": 1, "passage": 2, "entity": 3}
EDGE_TYPE_IDS = {t: i for i, t in enumerate(EDGE_TYPES)}
SCALAR_FEATURES = [
    "cosine_to_question", "bm25", "hybrid", "is_seed", "hop_distance", "is_question_entity", "log_degree",
    "specificity",
]


def feature_dim(embedding_dim: int) -> int:
    return embedding_dim + len(SCALAR_FEATURES)


class FeatureBuilder:
    def __init__(self, embedder: "EmbeddingGenerator") -> None:
        self.embedder = embedder

    def build(self, qg: QuestionGraph, corpus: "Corpus", question_vec: np.ndarray,
              gold_chunk_ids: Optional[set[str]] = None) -> tuple[Data, list[str]]:
        G = qg.graph.G
        node_ids = sorted(G.nodes, key=lambda n: (NODE_TYPE_IDS.get(G.nodes[n].get("type"), 9), n))
        index = {n: i for i, n in enumerate(node_ids)}
        dim = question_vec.shape[0]
        emb = np.zeros((len(node_ids), dim), dtype=np.float32)
        scalars = np.zeros((len(node_ids), len(SCALAR_FEATURES)), dtype=np.float32)
        graph_scores = np.zeros(len(node_ids), dtype=np.float32)  # baseline only, not a model input
        node_type = np.zeros(len(node_ids), dtype=np.int64)

        entity_nodes = [n for n in node_ids if G.nodes[n].get("type") == "entity"]
        entity_vecs = self.embedder.encode_names([G.nodes[n].get("label", n) for n in entity_nodes]) \
            if entity_nodes else np.zeros((0, dim), dtype=np.float32)
        entity_vec = {n: entity_vecs[i] for i, n in enumerate(entity_nodes)}

        doc_members: dict[str, list[np.ndarray]] = {}
        for n in node_ids:
            if G.nodes[n].get("type") == "passage":
                cid = chunk_id_of(n)
                doc_members.setdefault(f"doc:{corpus.chunk(cid).document_id}", []).append(corpus.embedding(cid))

        degrees = dict(G.degree())
        max_log_deg = math.log1p(max(degrees.values(), default=1)) or 1.0

        for n in node_ids:
            i = index[n]
            d = G.nodes[n]
            t = d.get("type")
            node_type[i] = NODE_TYPE_IDS[t]
            if t == "question":
                vec = question_vec
            elif t == "passage":
                cid = chunk_id_of(n)
                vec = corpus.embedding(cid)
                rp = qg.passages.get(cid)
                if rp is not None:
                    scalars[i, 1] = rp.bm25_score
                    scalars[i, 2] = rp.hybrid_score
                    scalars[i, 3] = 1.0 if rp.origin == "retrieval" else 0.0
                    graph_scores[i] = float(rp.graph_score or 0.0)
            elif t == "entity":
                vec = entity_vec[n]
                scalars[i, 5] = 1.0 if n in qg.question_entities else 0.0
                scalars[i, 7] = corpus.registry.specificity(n)
            else:  # document
                members = doc_members.get(n)
                if members:
                    m = np.mean(members, axis=0)
                    vec = m / (np.linalg.norm(m) or 1.0)
                else:
                    vec = np.zeros(dim, dtype=np.float32)
            emb[i] = vec
            scalars[i, 0] = float(np.dot(vec, question_vec))
            hop = qg.hop_distance.get(n)
            scalars[i, 4] = min(hop, 6) / 6.0 if hop is not None else 1.0
            scalars[i, 6] = math.log1p(degrees.get(n, 0)) / max_log_deg

        x = np.concatenate([emb * question_vec[None, :], scalars], axis=1)

        src, dst, etype = [], [], []
        for u, v, d in G.edges(data=True):
            t = EDGE_TYPE_IDS.get(d.get("type"), 0)
            for a, b in ((u, v), (v, u)):
                src.append(index[a])
                dst.append(index[b])
                etype.append(t)
        edge_index = torch.tensor([src, dst], dtype=torch.long) if src else torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((len(etype), len(EDGE_TYPES)), dtype=torch.float32)
        if etype:
            edge_attr[torch.arange(len(etype)), torch.tensor(etype)] = 1.0

        passage_mask = torch.tensor(node_type == NODE_TYPE_IDS["passage"], dtype=torch.bool)
        y = torch.zeros(len(node_ids), dtype=torch.float32)
        if gold_chunk_ids:
            for n in node_ids:
                if G.nodes[n].get("type") == "passage" and chunk_id_of(n) in gold_chunk_ids:
                    y[index[n]] = 1.0
        hybrid = torch.tensor(scalars[:, 2], dtype=torch.float32)
        graph_score = torch.from_numpy(graph_scores)
        data = Data(
            x=torch.from_numpy(x),
            edge_index=edge_index,
            edge_attr=edge_attr,
            node_type=torch.from_numpy(node_type),
            passage_mask=passage_mask,
            y=y,
            hybrid=hybrid,
            graph=graph_score,
        )
        data.num_nodes = len(node_ids)
        return data, node_ids
