"""End-to-end GNN training job (used by the API and scripts/train_gnn.py)."""
from __future__ import annotations

import json
import random
import shutil
import time
from typing import TYPE_CHECKING, Any, Callable, Optional

import torch

from app.core.logging_setup import get_logger
from app.gnn.dataset import demo_graphs, hotpotqa_graphs
from app.gnn.features import FEATURE_SPEC_VERSION, SCALAR_FEATURES
from app.gnn.trainer import GNNTrainer, TrainConfig

if TYPE_CHECKING:  # pragma: no cover
    from app.core.container import Container

log = get_logger(__name__)

SOURCE_LABELS = {
    "hotpotqa": "HotpotQA distractor setting - real multi-hop questions (train/val sampled from the official "
                "train split, CC BY-SA 4.0)",
    "demo": "RULE-DERIVED DEMONSTRATION DATA - hand-written demo questions plus template questions generated "
            "from extracted relations of the 5 bundled documents. Tiny; proves the pipeline runs, not that the "
            "model generalizes.",
}


def _cached_graphs(container: "Container", name: str, build: Callable[[], list[Any]]) -> list[Any]:
    path = container.settings.processed_dir / f"gnn_{name}_{FEATURE_SPEC_VERSION}.pt"
    model_tag = container.embedder.model_id
    if path.exists():
        payload = torch.load(str(path), weights_only=False)
        if payload.get("embedding_model") == model_tag:
            log.info("Loaded cached graph dataset %s (%d graphs)", path.name, len(payload["graphs"]))
            return payload["graphs"]
    graphs = build()
    torch.save({"embedding_model": model_tag, "graphs": graphs}, str(path))
    return graphs


def train_gnn(container: "Container", source: str = "hotpotqa", epochs: int = 40, train_limit: Optional[int] = None,
              val_limit: Optional[int] = None, lr: float = 1e-3, seed: int = 42,
              progress: Optional[Callable[[float, str], None]] = None,
              extra_train: bool = False) -> dict[str, Any]:
    started = time.perf_counter()

    def report(fraction: float, message: str) -> None:
        if progress:
            progress(fraction, message)
        log.info("[train %.0f%%] %s", fraction * 100, message)

    if source == "hotpotqa":
        def build(split: str, limit: Optional[int], lo: float, hi: float) -> list[Any]:
            return hotpotqa_graphs(container, split, limit,
                                   progress=lambda i, n: report(lo + (hi - lo) * i / n, f"building {split} graphs {i}/{n}"))

        train = _cached_graphs(container, f"hotpotqa_train_{train_limit or 'all'}",
                               lambda: build("train", train_limit, 0.0, 0.45))
        val = _cached_graphs(container, f"hotpotqa_val_{val_limit or 'all'}",
                             lambda: build("val", val_limit, 0.45, 0.55))
        info: dict[str, Any] = {"extra_train": extra_train}
        if extra_train:  # additional questions from scripts/download_datasets.py --extra-train
            extra = _cached_graphs(container, "hotpotqa_train_extra_all",
                                   lambda: build("train_extra", None, 0.0, 0.45))
            train = list(train) + list(extra)
            info["extra_train_graphs"] = len(extra)
    elif source == "demo":
        graphs, info = demo_graphs(container)
        rng = random.Random(seed)
        rng.shuffle(graphs)
        cut = max(1, int(0.8 * len(graphs)))
        train, val = graphs[:cut], graphs[cut:] or graphs[:1]
    else:
        raise ValueError("source must be 'hotpotqa' or 'demo'")
    report(0.55, f"training on {len(train)} graphs, validating on {len(val)}")

    settings = container.settings
    config = TrainConfig(epochs=epochs, lr=lr, seed=seed, hidden_dim=settings.gnn_hidden_dim,
                         heads=settings.gnn_heads, dropout=settings.gnn_dropout)
    history_path = settings.gnn_dir / "training_history.json"
    trainer = GNNTrainer(config, settings.gnn_checkpoint_path, history_path)
    meta = {
        "training_source": source,
        "dataset_label": SOURCE_LABELS[source] + (" + train_extra.json" if extra_train else ""),
        "embedding_model": container.embedder.model_id,
        "feature_spec_version": FEATURE_SPEC_VERSION,
        "scalar_features": SCALAR_FEATURES,
        "source_info": info,
    }
    summary = trainer.train(train, val, meta, progress=lambda ep, rec: report(
        0.55 + 0.45 * ep / epochs, f"epoch {ep}: train_loss={rec['train_loss']:.4f} val_mrr={rec['val_mrr']:.4f}"))
    settings.results_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(history_path, settings.results_dir / "training_history.json")
    summary["total_seconds"] = round(time.perf_counter() - started, 1)
    summary["training_source"] = source
    summary["num_train_graphs"] = len(train)
    summary["num_val_graphs"] = len(val)
    container.repo.log_metrics("training", source, {"gnn_training": summary["total_seconds"] * 1000}, summary)
    (settings.gnn_dir / "last_training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
