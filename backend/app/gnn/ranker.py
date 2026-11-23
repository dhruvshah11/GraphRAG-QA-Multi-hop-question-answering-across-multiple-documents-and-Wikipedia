"""GNN reranker inference and status reporting.

The ranker only reports itself as available when a checkpoint produced by
``scripts/train_gnn.py`` exists *and* was trained with the same feature
specification and embedding model. A randomly initialised model is never used
for ranking.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
import torch

from app.core.logging_setup import get_logger
from app.gnn.features import FEATURE_SPEC_VERSION, FeatureBuilder
from app.gnn.model import GATReranker
from app.graph.builder import QuestionGraph
from app.graph.evidence_graph import chunk_id_of

if TYPE_CHECKING:  # pragma: no cover
    from app.embeddings.embedder import EmbeddingGenerator
    from app.retrieval.corpus import Corpus

log = get_logger(__name__)


def load_checkpoint(path: Path) -> dict[str, Any]:
    return torch.load(str(path), map_location="cpu", weights_only=True)


class GNNRanker:
    def __init__(self, checkpoint_path: Path, embedder: "EmbeddingGenerator") -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.embedder = embedder
        self.features = FeatureBuilder(embedder)
        self.model: Optional[GATReranker] = None
        self.meta: dict[str, Any] = {}
        self.reason = "not loaded"
        self._mtime: Optional[float] = None
        self._lock = threading.Lock()

    def _maybe_reload(self) -> None:
        if not self.checkpoint_path.exists():
            self.model, self.meta, self._mtime = None, {}, None
            self.reason = "No trained checkpoint found. Run `python scripts/train_gnn.py` or use the GNN panel."
            return
        mtime = self.checkpoint_path.stat().st_mtime
        if self._mtime == mtime:
            return
        with self._lock:
            try:
                ckpt = load_checkpoint(self.checkpoint_path)
                meta = ckpt.get("meta", {})
                if meta.get("feature_spec_version") != FEATURE_SPEC_VERSION:
                    raise ValueError(f"feature spec {meta.get('feature_spec_version')} != {FEATURE_SPEC_VERSION}")
                if meta.get("embedding_model") != self.embedder.model_id:
                    raise ValueError(
                        f"trained with embeddings '{meta.get('embedding_model')}', current is '{self.embedder.model_id}'"
                    )
                model = GATReranker(**ckpt["config"])
                model.load_state_dict(ckpt["state_dict"])
                model.eval()
                self.model, self.meta = model, meta
                self.reason = "ready"
                log.info("Loaded GNN checkpoint (%s, trained on %s)", self.checkpoint_path.name,
                         meta.get("training_source"))
            except Exception as exc:
                self.model, self.meta = None, {}
                self.reason = f"Checkpoint incompatible or unreadable: {exc}"
                log.warning("GNN checkpoint not usable: %s", exc)
            self._mtime = mtime

    @property
    def is_available(self) -> bool:
        self._maybe_reload()
        return self.model is not None

    @torch.no_grad()
    def score(self, qg: QuestionGraph, corpus: "Corpus", question_vec: np.ndarray) -> dict[str, float]:
        """Probability that each passage of the subgraph is supporting evidence."""
        if not self.is_available:
            raise RuntimeError(self.reason)
        data, node_ids = self.features.build(qg, corpus, question_vec)
        assert self.model is not None
        logits = self.model(data.x, data.edge_index, data.node_type, data.edge_attr)
        probs = torch.sigmoid(logits).cpu().numpy()
        return {
            chunk_id_of(n): float(probs[i])
            for i, n in enumerate(node_ids)
            if qg.graph.G.nodes[n].get("type") == "passage"
        }

    def status(self) -> dict[str, Any]:
        self._maybe_reload()
        meta = dict(self.meta)
        history_path = self.checkpoint_path.with_name("training_history.json")
        return {
            "available": self.model is not None,
            "reason": self.reason,
            "checkpoint_path": str(self.checkpoint_path),
            "checkpoint_exists": self.checkpoint_path.exists(),
            "feature_spec_version": FEATURE_SPEC_VERSION,
            "architecture": "GATReranker: Linear -> 2x GATConv(4 heads, edge-type aware) -> MLP scorer",
            "num_parameters": self.model.num_parameters() if self.model is not None else None,
            "meta": meta,
            "history_available": history_path.exists(),
        }

    def training_history(self) -> Optional[dict[str, Any]]:
        path = self.checkpoint_path.with_name("training_history.json")
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        return None
