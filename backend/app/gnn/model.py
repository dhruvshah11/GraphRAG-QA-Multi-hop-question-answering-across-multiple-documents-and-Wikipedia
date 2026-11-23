"""Graph Attention Network for evidence reranking.

Architecture (all sizes configurable):

    x (emb*q || 8 scalars) --Linear--> h0 (+ node-type embedding), ReLU
    h1 = ReLU(GATConv_1(h0, edges, edge-type one-hot)) + h0      (4 heads, concat)
    h2 = ReLU(GATConv_2(h1, edges, edge-type one-hot)) + h1      (4 heads, concat)
    score = MLP([h0 || h2]) + w^T s  -> one logit per node (passage logits are used)

The "wide" term w^T s is a linear path from the 8 scalar relevance features
(cosine, BM25, hybrid, ...) straight to the logit (wide & deep). Without it the
strongest single signal is diluted inside the 392-d input projection.

Dropout is applied after the input projection and after each GAT layer.
Residual connections keep the direct relevance signal (h0) available while the
attention layers add information from neighbouring passages and entities
(e.g. a passage that shares a bridge entity with a highly relevant passage).
"""
from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import GATConv


class GATReranker(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 64, heads: int = 4, dropout: float = 0.2,
                 num_node_types: int = 4, edge_dim: int = 6, n_scalars: int = 8) -> None:
        super().__init__()
        if hidden_dim % heads != 0:
            raise ValueError("hidden_dim must be divisible by heads")
        self.config = {"in_dim": in_dim, "hidden_dim": hidden_dim, "heads": heads, "dropout": dropout,
                       "num_node_types": num_node_types, "edge_dim": edge_dim,
                       "n_scalars": n_scalars}
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.type_embedding = nn.Embedding(num_node_types, hidden_dim)
        self.gat1 = GATConv(hidden_dim, hidden_dim // heads, heads=heads, edge_dim=edge_dim, dropout=dropout)
        self.gat2 = GATConv(hidden_dim, hidden_dim // heads, heads=heads, edge_dim=edge_dim, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.scorer = nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
                                    nn.Linear(hidden_dim, 1))
        self.n_scalars = n_scalars
        self.wide = nn.Linear(n_scalars, 1) if n_scalars > 0 else None

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, node_type: torch.Tensor,
                edge_attr: torch.Tensor | None = None, return_attention: bool = False):
        h0 = torch.relu(self.input_proj(x) + self.type_embedding(node_type))
        h = self.dropout(h0)
        if return_attention:
            h1_raw, att1 = self.gat1(h, edge_index, edge_attr, return_attention_weights=True)
        else:
            h1_raw = self.gat1(h, edge_index, edge_attr)
        h1 = self.dropout(torch.relu(h1_raw) + h0)
        if return_attention:
            h2_raw, att2 = self.gat2(h1, edge_index, edge_attr, return_attention_weights=True)
        else:
            h2_raw = self.gat2(h1, edge_index, edge_attr)
        h2 = torch.relu(h2_raw) + h1
        logits = self.scorer(torch.cat([h0, h2], dim=-1)).squeeze(-1)
        if self.wide is not None:
            logits = logits + self.wide(x[:, -self.n_scalars:]).squeeze(-1)
        if return_attention:
            return logits, (att1, att2)
        return logits

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
