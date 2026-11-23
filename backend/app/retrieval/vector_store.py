"""FAISS vector store with persistence and incremental updates.

``IndexIDMap2(IndexFlatIP)`` is used: exact inner-product search over
L2-normalized vectors (= cosine similarity), with stable 64-bit ids so that
documents can be added and removed incrementally, and vectors can be
reconstructed by id.
"""
from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import faiss
import numpy as np

from app.core.logging_setup import get_logger

log = get_logger(__name__)


class VectorStore:
    def __init__(self, dim: int, model_id: str, index_path: Optional[Path] = None) -> None:
        self.dim = dim
        self.model_id = model_id
        self.index_path = Path(index_path) if index_path else None
        self.meta_path = self.index_path.with_suffix(".meta.json") if self.index_path else None
        self._lock = threading.RLock()
        self.index = self._new_index()

    def _new_index(self) -> faiss.IndexIDMap2:
        return faiss.IndexIDMap2(faiss.IndexFlatIP(self.dim))

    # ------------------------------------------------------------ persistence
    def load(self) -> bool:
        """Load the persisted index. Returns False if missing or incompatible."""
        if not self.index_path or not self.index_path.exists() or not self.meta_path.exists():
            return False
        meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
        if meta.get("model_id") != self.model_id or meta.get("dim") != self.dim:
            log.warning(
                "FAISS index was built with %s (dim %s); current model is %s (dim %s) -> rebuild required",
                meta.get("model_id"), meta.get("dim"), self.model_id, self.dim,
            )
            return False
        with self._lock:
            self.index = faiss.read_index(str(self.index_path))
        log.info("Loaded FAISS index with %d vectors", self.index.ntotal)
        return True

    def save(self) -> None:
        if not self.index_path:
            return
        with self._lock:
            self.index_path.parent.mkdir(parents=True, exist_ok=True)
            faiss.write_index(self.index, str(self.index_path))
            self.meta_path.write_text(
                json.dumps(
                    {
                        "model_id": self.model_id,
                        "dim": self.dim,
                        "count": int(self.index.ntotal),
                        "index_type": "IndexIDMap2(IndexFlatIP)",
                        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

    # -------------------------------------------------------------- mutation
    def add(self, vectors: np.ndarray, ids: list[int]) -> None:
        if len(ids) == 0:
            return
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        if vectors.shape[1] != self.dim:
            raise ValueError(f"Vector dim {vectors.shape[1]} != index dim {self.dim}")
        with self._lock:
            self.index.add_with_ids(vectors, np.asarray(ids, dtype=np.int64))

    def remove(self, ids: list[int]) -> int:
        if not ids:
            return 0
        with self._lock:
            return int(self.index.remove_ids(np.asarray(ids, dtype=np.int64)))

    def reset(self) -> None:
        with self._lock:
            self.index = self._new_index()

    # ---------------------------------------------------------------- search
    def search(self, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(scores, ids)`` of the ``k`` most similar vectors."""
        with self._lock:
            n = int(self.index.ntotal)
            if n == 0 or k <= 0:
                return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.int64)
            q = np.ascontiguousarray(query.reshape(1, -1), dtype=np.float32)
            scores, ids = self.index.search(q, min(k, n))
        mask = ids[0] >= 0
        return scores[0][mask], ids[0][mask]

    def reconstruct(self, vector_id: int) -> np.ndarray:
        with self._lock:
            return self.index.reconstruct(int(vector_id))

    @property
    def size(self) -> int:
        return int(self.index.ntotal)
