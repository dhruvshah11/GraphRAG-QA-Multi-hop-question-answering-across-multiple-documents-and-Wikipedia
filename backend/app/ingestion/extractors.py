"""Format-specific text extraction (PDF, TXT, DOCX).

Each extractor returns a list of ``RawPage`` objects. Page numbers are kept for
PDFs; for TXT (form-feed separated) and DOCX (explicit / last-rendered page
breaks) logical page numbers are derived from page-break markers when present.
DOCX headings are returned as section markers so that chunks can carry the
section title.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from app.core.logging_setup import get_logger

log = get_logger(__name__)


class ExtractionError(Exception):
    """Raised when a document cannot be read or contains no text."""


@dataclass
class RawBlock:
    text: str
    heading: Optional[str] = None  # set when this block starts a new section


@dataclass
class RawPage:
    page_number: int
    blocks: list[RawBlock] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n\n".join(b.text for b in self.blocks if b.text.strip())


def extract_pdf(path: Path) -> list[RawPage]:
    try:
        import pymupdf  # PyMuPDF >= 1.24 (the legacy module name is ``fitz``)
    except ImportError:  # pragma: no cover
        try:
            import fitz as pymupdf
        except ImportError as exc:
            raise ExtractionError("PyMuPDF is not installed") from exc
    try:
        pdf = pymupdf.open(str(path))
    except Exception as exc:
        raise ExtractionError(f"Could not open PDF: {exc}") from exc
    pages: list[RawPage] = []
    try:
        if pdf.needs_pass:
            raise ExtractionError("PDF is password protected")
        for index, page in enumerate(pdf):
            text = page.get_text("text", sort=True) or ""
            pages.append(RawPage(page_number=index + 1, blocks=[RawBlock(text=text)]))
    finally:
        pdf.close()
    if not any(p.text.strip() for p in pages):
        raise ExtractionError("PDF contains no extractable text (scanned images are not supported; OCR is out of scope)")
    return pages


def _decode_bytes(data: bytes) -> str:
    """Decode text files: UTF-16 (BOM), then UTF-8, then Windows-1252."""
    if data[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return data.decode("utf-16", errors="replace")
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        pass
    try:
        return data.decode("cp1252")
    except UnicodeDecodeError:
        return data.decode("latin-1", errors="replace")


def extract_txt(path: Path) -> list[RawPage]:
    text = _decode_bytes(path.read_bytes())
    parts = text.split("\f")  # form feed = explicit page break
    pages = [RawPage(page_number=i + 1, blocks=[RawBlock(text=part)]) for i, part in enumerate(parts)]
    pages = [p for p in pages if p.text.strip()] or []
    if not pages:
        raise ExtractionError("Text file is empty")
    # renumber after dropping empty pages
    for i, p in enumerate(pages):
        p.page_number = i + 1
    return pages


def extract_docx(path: Path) -> list[RawPage]:
    try:
        import docx  # python-docx
        from docx.oxml.ns import qn
    except ImportError as exc:  # pragma: no cover
        raise ExtractionError("python-docx is not installed") from exc
    try:
        document = docx.Document(str(path))
    except Exception as exc:
        raise ExtractionError(f"Could not open DOCX: {exc}") from exc

    pages: list[RawPage] = [RawPage(page_number=1)]
    body = document.element.body
    para_by_element = {p._element: p for p in document.paragraphs}
    table_by_element = {t._element: t for t in document.tables}

    for child in body.iterchildren():
        if child.tag == qn("w:p"):
            paragraph = para_by_element.get(child)
            if paragraph is None:
                continue
            xml = child.xml
            # A page break before/inside this paragraph starts a new logical page.
            breaks = xml.count('w:type="page"') + xml.count("lastRenderedPageBreak")
            text = paragraph.text.strip()
            if breaks and pages[-1].blocks:
                pages.append(RawPage(page_number=len(pages) + 1))
            if not text:
                continue
            style = (paragraph.style.name if paragraph.style is not None else "") or ""
            is_heading = style.lower().startswith("heading") or style.lower() == "title"
            pages[-1].blocks.append(RawBlock(text=text, heading=text if is_heading else None))
        elif child.tag == qn("w:tbl"):
            table = table_by_element.get(child)
            if table is None:
                continue
            rows = []
            for row in table.rows:
                cells = []
                for cell in row.cells:
                    cell_text = cell.text.strip()
                    if cell_text and (not cells or cells[-1] != cell_text):  # merged cells repeat
                        cells.append(cell_text)
                if cells:
                    rows.append(" | ".join(cells))
            if rows:
                pages[-1].blocks.append(RawBlock(text="\n".join(rows)))
    pages = [p for p in pages if p.blocks]
    if not pages:
        raise ExtractionError("DOCX contains no text")
    for i, p in enumerate(pages):
        p.page_number = i + 1
    return pages


EXTRACTORS = {".pdf": extract_pdf, ".txt": extract_txt, ".docx": extract_docx}


def extract_pages(path: Path) -> list[RawPage]:
    suffix = path.suffix.lower()
    if suffix not in EXTRACTORS:
        raise ExtractionError(f"Unsupported file type: {suffix}")
    pages = EXTRACTORS[suffix](path)
    log.info("Extracted %d page(s) from %s", len(pages), path.name)
    return pages
