"""Training loop, validation, negative sampling and checkpointing for the GAT reranker."""
from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from app.core.logging_setup import get_logger
from app.core.seeds import set_seed
from app.gnn.model import GATReranker

log = get_logger(__name__)


@dataclass
class TrainConfig:
    epochs: int = 40
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 16
    neg_ratio: int = 3  # sampled negatives per positive (half hard, half random)
    hidden_dim: int = 64
    heads: int = 4
    dropout: float = 0.2
    patience: int = 8
    seed: int = 42
    k: int = 2


def sample_training_mask(data: Data, neg_ratio: int, rng: random.Random) -> torch.Tensor:
    """Positives + sampled negatives among passage nodes.

    Half of the negatives are *hard* (highest hybrid retrieval score among
    non-gold passages), half are random, so the model learns to beat the
    retrieval baseline rather than to separate trivially irrelevant passages.
    """
    passage_idx = torch.nonzero(data.passage_mask, as_tuple=False).flatten().tolist()
    pos = [i for i in passage_idx if data.y[i] > 0.5]
    neg = [i for i in passage_idx if data.y[i] <= 0.5]
    n_neg = min(len(neg), max(1, neg_ratio * max(1, len(pos))))
    hard_sorted = sorted(neg, key=lambda i: -float(data.hybrid[i]))
    n_hard = n_neg // 2
    chosen = set(hard_sorted[:n_hard])
    remaining = [i for i in neg if i not in chosen]
    rng.shuffle(remaining)
    chosen |= set(remaining[: n_neg - n_hard])
    mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    for i in pos + sorted(chosen):
        mask[i] = True
    return mask


def ranking_metrics(scores: np.ndarray, labels: np.ndarray, k: int) -> Optional[dict[str, float]]:
    """P@k, R@k and reciprocal rank for one graph; None if no positive exists."""
    if labels.sum() == 0:
        return None
    order = np.argsort(-scores, kind="stable")
    ranked = labels[order]
    hits_k = ranked[:k].sum()
    first = int(np.argmax(ranked > 0.5))
    return {
        "precision_at_k": float(hits_k / k),
        "recall_at_k": float(hits_k / labels.sum()),
        "mrr": 1.0 / (first + 1),
        "all_gold_in_top_k": float(hits_k == labels.sum()),
    }


@torch.no_grad()
def evaluate_model(model: Optional[GATReranker], dataset: list[Data], k: int = 2,
                   use_baseline: bool = False, baseline_attr: str = "hybrid") -> dict[str, float]:
    """Mean ranking metrics over graphs (passage nodes only) + BCE loss."""
    if model is not None:
        model.eval()
    loss_fn = nn.BCEWithLogitsLoss()
    per_graph, losses = [], []
    for data in dataset:
        mask = data.passage_mask
        labels = data.y[mask].numpy()
        if use_baseline or model is None:
            scores = getattr(data, baseline_attr)[mask].numpy()
        else:
            logits = model(data.x, data.edge_index, data.node_type, data.edge_attr)
            losses.append(float(loss_fn(logits[mask], data.y[mask])))
            scores = torch.sigmoid(logits[mask]).numpy()
        m = ranking_metrics(scores, labels, k)
        if m:
            per_graph.append(m)
    out = {key: float(np.mean([m[key] for m in per_graph])) if per_graph else 0.0
           for key in ("precision_at_k", "recall_at_k", "mrr", "all_gold_in_top_k")}
    out["graphs_with_gold"] = len(per_graph)
    if losses:
        out["loss"] = float(np.mean(losses))
    return out


class GNNTrainer:
    def __init__(self, config: TrainConfig, checkpoint_path: Path, history_path: Path) -> None:
        self.config = config
        self.checkpoint_path = Path(checkpoint_path)
        self.history_path = Path(history_path)

    def train(self, train_set: list[Data], val_set: list[Data], meta: dict[str, Any],
              progress: Optional[Callable[[int, dict[str, Any]], None]] = None) -> dict[str, Any]:
        cfg = self.config
        if not train_set:
            raise ValueError("Training set is empty")
        set_seed(cfg.seed)
        rng = random.Random(cfg.seed)
        in_dim = int(train_set[0].x.shape[1])
        from app.gnn.features import SCALAR_FEATURES

        model = GATReranker(in_dim, cfg.hidden_dim, cfg.heads, cfg.dropout,
                            edge_dim=int(train_set[0].edge_attr.shape[1]), n_scalars=len(SCALAR_FEATURES))
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        loss_fn = nn.BCEWithLogitsLoss()

        baseline_val = evaluate_model(None, val_set, cfg.k, use_baseline=True)
        baseline_graph = (evaluate_model(None, val_set, cfg.k, use_baseline=True, baseline_attr="graph")
                          if hasattr(val_set[0], "graph") else None)
        history: list[dict[str, Any]] = []
        best_mrr, best_loss = -1.0, float("inf")
        best_epoch, stale = 0, 0
        started = time.perf_counter()
        log.info("Training GAT reranker: %d train / %d val graphs, %d parameters",
                 len(train_set), len(val_set), model.num_parameters())

        for epoch in range(1, cfg.epochs + 1):
            model.train()
            for data in train_set:
                data.train_mask = sample_training_mask(data, cfg.neg_ratio, rng)
            loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True)
            total, count = 0.0, 0
            for batch in loader:
                optimizer.zero_grad()
                logits = model(batch.x, batch.edge_index, batch.node_type, batch.edge_attr)
                mask = batch.train_mask
                loss = loss_fn(logits[mask], batch.y[mask])
                loss.backward()
                optimizer.step()
                total += float(loss) * int(mask.sum())
                count += int(mask.sum())
            val = evaluate_model(model, val_set, cfg.k)
            record = {"epoch": epoch, "train_loss": total / max(1, count), **{f"val_{k}": v for k, v in val.items()}}
            history.append(record)
            val_loss = val.get("loss", float("inf"))
            # model selection: best validation MRR, ties broken by lower validation loss
            improved = val["mrr"] > best_mrr + 1e-6 or (abs(val["mrr"] - best_mrr) <= 1e-6 and val_loss < best_loss)
            if improved:
                best_mrr, best_loss = val["mrr"], val_loss
                best_epoch, stale = epoch, 0
                self._save(model, meta, cfg, history, baseline_val, record, len(train_set), len(val_set))
            else:
                stale += 1
            if progress:
                progress(epoch, record)
            log.info("epoch %02d  train_loss=%.4f  val_loss=%.4f  val_mrr=%.4f  val_R@%d=%.4f",
                     epoch, record["train_loss"], val.get("loss", float("nan")), val["mrr"], cfg.k, val["recall_at_k"])
            if stale >= cfg.patience:
                log.info("Early stopping at epoch %d (best epoch %d)", epoch, best_epoch)
                break

        summary = {
            "best_epoch": best_epoch,
            "epochs_run": len(history),
            "train_seconds": round(time.perf_counter() - started, 2),
            "best_val": next(h for h in history if h["epoch"] == best_epoch),
            "baseline_val_hybrid": baseline_val,
            "baseline_val_graph_heuristic": baseline_graph,
            "num_parameters": model.num_parameters(),
        }
        history_payload = {"config": asdict(cfg), "meta": meta, "history": history, "summary": summary}
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        self.history_path.write_text(json.dumps(history_payload, indent=2), encoding="utf-8")
        # update the checkpoint meta with the final summary
        ckpt = torch.load(str(self.checkpoint_path), map_location="cpu", weights_only=True)
        ckpt["meta"]["summary"] = summary
        torch.save(ckpt, str(self.checkpoint_path))
        return summary

    def _save(self, model: GATReranker, meta: dict[str, Any], cfg: TrainConfig, history: list[dict[str, Any]],
              baseline_val: dict[str, float], best: dict[str, Any], n_train: int, n_val: int) -> None:
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        full_meta = {
            **meta,
            "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "train_config": asdict(cfg),
            "num_train_graphs": n_train,
            "num_val_graphs": n_val,
            "best_epoch": best["epoch"],
            "best_val": best,
            "baseline_val_hybrid": baseline_val,
        }
        torch.save({"state_dict": model.state_dict(), "config": model.config, "meta": full_meta},
                   str(self.checkpoint_path))
