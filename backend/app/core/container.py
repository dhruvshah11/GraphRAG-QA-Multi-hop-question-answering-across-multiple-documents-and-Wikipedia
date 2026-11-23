"""Composition root: creates every service once and wires them together.

Models (embeddings, spaCy, reader, GNN) are loaded lazily and exactly once per
process; the API, the scripts and the tests all obtain services from here.
"""
from __future__ import annotations

import threading
import time
from datetime import datetime, timezone
from typing import Optional

from app.core.config import Settings, get_settings
from app.core.jobs import JobManager
from app.core.logging_setup import configure_logging, get_logger
from app.core.seeds import set_seed
from app.core.settings_service import RuntimeSettings, SettingsService
from app.database.db import Database
from app.database.repository import Repository
from app.embeddings.embedder import EmbeddingGenerator
from app.entities.extractor import EntityExtractor
from app.generation.generator import AnswerGenerator
from app.gnn.ranker import GNNRanker
from app.graph.builder import GraphBuilder
from app.ingestion.document_processor import DocumentProcessor
from app.ingestion.service import IngestionService
from app.pipeline.graph_reasoner import GraphReasoner
from app.pipeline.qa_pipeline import QAPipeline
from app.preprocessing.chunker import Chunker
from app.retrieval.corpus import CorpusManager
from app.retrieval.wikipedia import OpenDomainRetriever

log = get_logger(__name__)


class Container:
    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()
        self.settings.ensure_dirs()
        configure_logging(self.settings.log_level)
        set_seed(self.settings.seed)
        self.started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self.db = Database(self.settings.db_path)
        self.repo = Repository(self.db)
        self.settings_service = SettingsService(self.settings, self.repo)
        rt = self.runtime()
        self.embedder = EmbeddingGenerator(rt.embedding_model, self.settings.embedding_cache_dir, self.settings.device,
                                           self.settings.embedding_batch_size, self.settings.embedding_window_words)
        self.extractor = EntityExtractor(self.settings.spacy_model)
        self.graph_builder = GraphBuilder(self.settings.max_bridge_entity_df_ratio, self.settings.max_bridge_entity_df_abs)
        self.gnn_ranker = GNNRanker(self.settings.gnn_checkpoint_path, self.embedder)
        self.reasoner = GraphReasoner(self.embedder, self.extractor, self.graph_builder, gnn=self.gnn_ranker)
        self.generator = AnswerGenerator(self.settings, self.embedder, self.extractor)
        self.corpus_manager = CorpusManager(self.settings, self.repo, self.embedder, self.extractor, self.graph_builder)
        self.processor = DocumentProcessor(self.settings, self.repo)
        self.ingestion = IngestionService(self)
        self.open_domain = OpenDomainRetriever(self)
        self.pipeline = QAPipeline(self)
        self.jobs = JobManager(self.repo)
        self.warmup_state = {"status": "not_started", "started_at": None, "finished_at": None, "ms": {}}

    def runtime(self) -> RuntimeSettings:
        return self.settings_service.get()

    def make_chunker(self) -> Chunker:
        rt = self.runtime()
        return Chunker(rt.chunk_size, rt.chunk_overlap)

    def reload_embedder(self) -> None:
        """Switch to the embedding model selected in the runtime settings."""
        rt = self.runtime()
        if rt.embedding_model == self.embedder.model_name:
            return
        self.embedder = EmbeddingGenerator(rt.embedding_model, self.settings.embedding_cache_dir, self.settings.device,
                                           self.settings.embedding_batch_size, self.settings.embedding_window_words)
        for obj in (self.corpus_manager, self.reasoner, self.generator):
            obj.embedder = self.embedder
        self.gnn_ranker = GNNRanker(self.settings.gnn_checkpoint_path, self.embedder)
        self.reasoner.gnn = self.gnn_ranker
        self.generator.backends["extractive"].embedder = self.embedder  # type: ignore[attr-defined]
        self.generator.backends["hf_generator"].embedder = self.embedder  # type: ignore[attr-defined]
        self.corpus_manager._store = None
        self.corpus_manager._corpus = None

    def warm_up(self) -> None:
        """Load models and the corpus snapshot (called in a background thread at startup)."""
        self.warmup_state.update(status="running", started_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))
        timings = {}
        try:
            for name, fn in (
                ("embedding_model", self.embedder.load),
                ("spacy_model", self.extractor.load),
                ("reader_model", self.generator.reader.load),
                ("corpus_snapshot", lambda: self.corpus_manager.corpus),
                ("gnn_checkpoint", lambda: self.gnn_ranker.is_available),
            ):
                start = time.perf_counter()
                fn()
                timings[name] = round((time.perf_counter() - start) * 1000, 1)
            self.warmup_state["status"] = "ready"
        except Exception as exc:  # pragma: no cover - reported in /api/health
            self.warmup_state["status"] = f"error: {exc}"
            log.exception("Warm-up failed")
        self.warmup_state["ms"] = timings
        self.warmup_state["finished_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        log.info("Warm-up finished: %s", timings)


_container: Optional[Container] = None
_lock = threading.Lock()


def get_container() -> Container:
    global _container
    if _container is None:
        with _lock:
            if _container is None:
                _container = Container()
    return _container


def set_container(container: Optional[Container]) -> None:
    """Replace the global container (used by tests)."""
    global _container
    _container = container
