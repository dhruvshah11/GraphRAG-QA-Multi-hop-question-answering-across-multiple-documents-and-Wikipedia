"""Sentence-embedding generation.

Default model: ``sentence-transformers/all-MiniLM-L6-v2`` (384-d, ~90 MB,
fast on CPU). Vectors are L2-normalized so that inner product == cosine.

all-MiniLM-L6-v2 reads at most 256 word pieces. Passages longer than
``window_words`` words are therefore split into sentence windows, each window
is embedded and the passage vector is the normalized mean of its windows, so
no part of a 400-token chunk is silently truncated.

If the model cannot be loaded (no internet on first run and no local copy), a
deterministic hashing embedder is used instead and reported as a degraded
fallback in ``/api/health``.
"""
from __future__ import annotations

import hashlib
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Optional

import numpy as np

from app.core.logging_setup import get_logger
from app.preprocessing.sentences import split_sentences
from app.preprocessing.tokenize import content_terms

log = get_logger(__name__)


class HashingEmbedder:
    """Lexical fallback: signed feature hashing of stemmed unigrams + bigrams."""

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            terms = content_terms(text)
            features = terms + [f"{a}_{b}" for a, b in zip(terms, terms[1:])]
            for feat in features:
                h = int.from_bytes(hashlib.md5(feat.encode("utf-8")).digest()[:8], "little")
                out[i, h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return out / norms


class EmbeddingGenerator:
    def __init__(
        self,
        model_name: str,
        cache_dir: Path,
        device: str = "cpu",
        batch_size: int = 32,
        window_words: int = 180,
    ) -> None:
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self.device = device
        self.batch_size = batch_size
        self.window_words = window_words
        self._model: Any = None
        self._fallback: Optional[HashingEmbedder] = None
        self._lock = threading.Lock()
        self._query_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._name_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self.backend = "not_loaded"
        self.dim: Optional[int] = None
        self.load_ms: Optional[float] = None
        self.error: Optional[str] = None

    # ----------------------------------------------------------------- loading
    def load(self) -> None:
        if self._model is not None or self._fallback is not None:
            return
        with self._lock:
            if self._model is not None or self._fallback is not None:
                return
            start = time.perf_counter()
            try:
                from sentence_transformers import SentenceTransformer

                from app.generation.hf_models import hf_model_cached

                self._model = SentenceTransformer(
                    self.model_name, cache_folder=str(self.cache_dir), device=self.device,
                    local_files_only=hf_model_cached(self.model_name, self.cache_dir),
                )
                getter = getattr(self._model, "get_embedding_dimension", None) or getattr(
                    self._model, "get_sentence_embedding_dimension"
                )
                self.dim = int(getter())
                self.backend = "sentence-transformers"
            except Exception as exc:  # offline on first run, corrupted cache, ...
                self.error = str(exc)
                self._fallback = HashingEmbedder()
                self.dim = self._fallback.dim
                self.backend = "hashing_fallback"
                log.warning("Embedding model unavailable (%s); using hashing fallback", exc)
            self.load_ms = (time.perf_counter() - start) * 1000
            log.info("Embedding backend %s ready (dim=%s, %.0f ms)", self.backend, self.dim, self.load_ms)

    @property
    def model_id(self) -> str:
        """Identifier stored with the FAISS index to detect model changes."""
        self.load()
        return self.model_name if self._model is not None else f"hashing-{self.dim}"

    # ---------------------------------------------------------------- encoding
    def encode(self, texts: list[str]) -> np.ndarray:
        """Embed short texts (questions, entity names, sentences)."""
        self.load()
        if not texts:
            return np.zeros((0, self.dim or 0), dtype=np.float32)
        if self._model is None:
            return self._fallback.encode(texts)  # type: ignore[union-attr]
        with self._lock:
            vecs = self._model.encode(
                texts,
                batch_size=self.batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        return np.asarray(vecs, dtype=np.float32)

    def _windows(self, text: str) -> list[str]:
        words = text.split()
        if len(words) <= self.window_words:
            return [text]
        windows, current, count = [], [], 0
        for start, end in split_sentences(text):
            sentence = text[start:end]
            n = len(sentence.split())
            if current and count + n > self.window_words:
                windows.append(" ".join(current))
                current, count = [], 0
            current.append(sentence)
            count += n
        if current:
            windows.append(" ".join(current))
        return windows or [text]

    def encode_passages(self, texts: list[str]) -> np.ndarray:
        """Embed passages with window averaging for long texts."""
        self.load()
        if not texts:
            return np.zeros((0, self.dim or 0), dtype=np.float32)
        if self._model is None:
            return self._fallback.encode(texts)  # type: ignore[union-attr]
        all_windows: list[str] = []
        owners: list[int] = []
        for i, text in enumerate(texts):
            for w in self._windows(text):
                all_windows.append(w)
                owners.append(i)
        window_vecs = self.encode(all_windows)
        out = np.zeros((len(texts), window_vecs.shape[1]), dtype=np.float32)
        np.add.at(out, np.asarray(owners), window_vecs)
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return out / norms

    def encode_query(self, text: str) -> np.ndarray:
        """Embed a question (LRU-cached)."""
        key = text.strip()
        cached = self._query_cache.get(key)
        if cached is not None:
            self._query_cache.move_to_end(key)
            return cached
        vec = self.encode([key])[0]
        self._query_cache[key] = vec
        if len(self._query_cache) > 512:
            self._query_cache.popitem(last=False)
        return vec

    def encode_names(self, names: list[str]) -> np.ndarray:
        """Embed entity names, caching vectors across questions (bounded LRU)."""
        self.load()
        if not names:
            return np.zeros((0, self.dim or 0), dtype=np.float32)
        unique = list(dict.fromkeys(names))
        found = {n: self._name_cache[n] for n in unique if n in self._name_cache}
        missing = [n for n in unique if n not in found]
        if missing:
            found.update(zip(missing, self.encode(missing)))
        for n in unique:  # refresh recency, then evict the oldest entries
            self._name_cache[n] = found[n]
            self._name_cache.move_to_end(n)
        while len(self._name_cache) > 20_000:
            self._name_cache.popitem(last=False)
        return np.stack([found[n] for n in names])
