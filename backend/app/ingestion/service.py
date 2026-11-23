"""Document processing workflow: extract -> chunk -> embed -> NER -> index -> graph."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from app.core.logging_setup import get_logger
from app.core.timing import StageTimer
from app.database.repository import utc_now
from app.ingestion.document_processor import DocumentProcessor
from app.preprocessing.chunker import chunk_statistics

if TYPE_CHECKING:  # pragma: no cover
    from app.core.container import Container

log = get_logger(__name__)


class IngestionService:
    def __init__(self, container: "Container") -> None:
        self.c = container

    def process_document(self, doc_id: str, progress: Optional[Any] = None) -> dict[str, Any]:
        repo = self.c.repo
        row = repo.get_document(doc_id)
        if row is None:
            raise KeyError(f"Document {doc_id} not found")
        doc = DocumentProcessor.document_from_row(row)
        repo.update_document(doc_id, status="processing", error=None)
        timer = StageTimer()

        def step(fraction: float, msg: str) -> None:
            if progress:
                progress(fraction, msg)

        try:
            step(0.05, "extracting text")
            with timer.stage("extraction"):
                doc = self.c.processor.extract(doc)
            repo.save_document_content(doc)
            step(0.2, "chunking")
            with timer.stage("chunking"):
                chunks = self.c.make_chunker().chunk_document(doc)
            step(0.35, "embedding + entity extraction")
            stats = self.c.corpus_manager.index_document(doc, chunks, timer)
            repo.update_document(doc_id, status="processed", num_pages=doc.num_pages, num_chunks=len(chunks),
                                 num_entities=stats["distinct_entities"], processed_at=utc_now(),
                                 processing_ms=round(timer.total_ms, 1))
            step(0.85, "rebuilding evidence graph")
            with timer.stage("graph_refresh"):
                corpus = self.c.corpus_manager.refresh()
            repo.update_document(doc_id, processing_ms=round(timer.total_ms, 1))
            summary = {
                "document_id": doc_id,
                "pages": doc.num_pages,
                **stats,
                "chunk_stats": chunk_statistics(chunks),
                "timings_ms": timer.as_dict(),
                "graph": corpus.graph.stats(),
            }
            repo.log_metrics("ingestion", doc_id, timer.as_dict(),
                             {"pages": doc.num_pages, "chunks": len(chunks), "file_type": doc.file_type,
                              "size_bytes": doc.size_bytes})
            log.info("Processed %s: %d pages, %d chunks, %d mentions (%.0f ms)", doc.name, doc.num_pages,
                     len(chunks), stats["mentions"], timer.total_ms)
            return summary
        except Exception as exc:
            repo.update_document(doc_id, status="failed", error=f"{exc.__class__.__name__}: {exc}")
            log.error("Processing of %s failed: %s", doc_id, exc)
            raise

    def delete_document(self, doc_id: str) -> None:
        row = self.c.repo.get_document(doc_id)
        if row is None:
            raise KeyError(f"Document {doc_id} not found")
        self.c.corpus_manager.remove_document(doc_id)
        self.c.repo.delete_document(doc_id)
        if row.get("path"):
            path = Path(row["path"])
            if path.exists() and self.c.settings.uploads_dir in path.parents:
                path.unlink()
        self.c.corpus_manager.refresh()
