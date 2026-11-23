"""Document ingestion: validation, safe storage, extraction and normalization."""
from __future__ import annotations

import hashlib
import re
import unicodedata
from pathlib import Path

from app.core.config import Settings
from app.core.domain import Document, PageText, Section
from app.core.logging_setup import get_logger
from app.database.repository import Repository, utc_now
from app.ingestion.extractors import ExtractionError, RawPage, extract_pages
from app.ingestion.text_normalizer import normalize_text

log = get_logger(__name__)


class DocumentValidationError(ValueError):
    """The uploaded file is not acceptable (type, size, emptiness)."""


def safe_filename(name: str) -> str:
    """Return a filesystem-safe version of a user supplied filename.

    Directory components are removed (prevents path traversal such as
    ``../../etc/passwd``), unicode is transliterated to ASCII and characters
    outside ``[A-Za-z0-9._-]`` are replaced.
    """
    name = Path(name.replace("\\", "/")).name
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"[^A-Za-z0-9._-]", "_", name).strip("._")
    return name[:120] or "document"


def build_document_text(raw_pages: list[RawPage]) -> tuple[str, list[PageText], list[Section]]:
    """Join normalized blocks into one text, recording page offsets and sections."""
    text = ""
    pages: list[PageText] = []
    sections: list[Section] = []
    section_title: str | None = None
    section_start = 0

    for raw_page in raw_pages:
        page_start: int | None = None
        for block in raw_page.blocks:
            normalized = normalize_text(block.text)
            if not normalized:
                continue
            if text:
                text += "\n\n"
            block_start = len(text)
            if block.heading is not None:
                if block_start > section_start:
                    sections.append(Section(section_title, section_start, block_start))
                section_title = normalize_text(block.heading)
                section_start = block_start
            if page_start is None:
                page_start = block_start
            text += normalized
        if page_start is not None:
            pages.append(PageText(raw_page.page_number, text[page_start:], page_start))
    if len(text) > section_start:
        sections.append(Section(section_title, section_start, len(text)))
    return text, pages, sections


class DocumentProcessor:
    """Validates uploads, stores them safely and extracts normalized text."""

    def __init__(self, settings: Settings, repo: Repository) -> None:
        self.settings = settings
        self.repo = repo

    # ------------------------------------------------------------ validation
    def validate(self, filename: str, size_bytes: int) -> str:
        suffix = Path(filename).suffix.lower()
        if suffix not in self.settings.allowed_extensions:
            allowed = ", ".join(self.settings.allowed_extensions)
            raise DocumentValidationError(f"Unsupported file type '{suffix or '?'}'. Allowed: {allowed}")
        if size_bytes == 0:
            raise DocumentValidationError("File is empty")
        max_bytes = self.settings.max_upload_mb * 1024 * 1024
        if size_bytes > max_bytes:
            raise DocumentValidationError(
                f"File is {size_bytes / 1048576:.1f} MB; the limit is {self.settings.max_upload_mb} MB"
            )
        return suffix

    # --------------------------------------------------------------- storage
    def register_upload(self, filename: str, data: bytes) -> tuple[Document, bool]:
        """Persist an uploaded file and create its database record.

        Returns ``(document, is_duplicate)``. Uploading identical bytes twice
        returns the existing document instead of indexing it again.
        """
        suffix = self.validate(filename, len(data))
        checksum = hashlib.sha256(data).hexdigest()
        existing = self.repo.find_document_by_checksum(checksum)
        if existing:
            return self.document_from_row(existing), True

        doc_id = f"doc_{checksum[:12]}"
        clean_name = safe_filename(filename)
        stored_path = self.settings.uploads_dir / f"{doc_id}_{clean_name}"
        stored_path.write_bytes(data)
        doc = Document(
            id=doc_id,
            name=clean_name,
            file_type=suffix.lstrip("."),
            source_type="upload",
            path=str(stored_path),
            size_bytes=len(data),
            status="uploaded",
            checksum=checksum,
            created_at=utc_now(),
        )
        self.repo.insert_document(doc)
        log.info("Registered upload %s as %s (%d bytes)", clean_name, doc_id, len(data))
        return doc, False

    @staticmethod
    def document_from_row(row: dict) -> Document:
        return Document(
            id=row["id"], name=row["name"], file_type=row["file_type"], source_type=row["source_type"],
            path=row["path"], url=row["url"], size_bytes=row["size_bytes"], num_pages=row["num_pages"],
            num_chunks=row["num_chunks"], num_entities=row["num_entities"], status=row["status"],
            error=row["error"], checksum=row["checksum"], created_at=row["created_at"],
            processed_at=row["processed_at"],
        )

    # ------------------------------------------------------------ extraction
    def extract(self, doc: Document) -> Document:
        """Extract, normalize and attach page/section structure to ``doc``."""
        if not doc.path or not Path(doc.path).exists():
            raise ExtractionError(f"Stored file for {doc.id} is missing")
        raw_pages = extract_pages(Path(doc.path))
        text, pages, sections = build_document_text(raw_pages)
        if not text.strip():
            raise ExtractionError("No text could be extracted from the document")
        doc.text = text
        doc.pages = pages
        doc.sections = sections
        doc.num_pages = len(raw_pages)
        return doc
