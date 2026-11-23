"""Document ingestion: PDF / TXT / DOCX extraction, normalization, validation."""
from __future__ import annotations

from pathlib import Path

import pytest

from app.ingestion.document_processor import DocumentValidationError, build_document_text, safe_filename
from app.ingestion.extractors import ExtractionError, extract_pages
from app.ingestion.text_normalizer import normalize_text
from tests.conftest import DOCS


def test_pdf_extraction_preserves_pages():
    pages = extract_pages(DOCS / "nobel_prizes_2024.pdf")
    assert [p.page_number for p in pages] == [1, 2]
    text, page_texts, _ = build_document_text(pages)
    assert "Geoffrey E. Hinton of the University of Toronto" in text
    assert "Nobel Prize in Chemistry 2024" in page_texts[1].text
    # page offsets point into the full text
    for p in page_texts:
        assert text[p.char_offset:p.char_offset + 20] == p.text[:20]


def test_txt_ingestion_and_form_feed_pages(tmp_path: Path):
    pages = extract_pages(DOCS / "geoffrey_hinton_profile.txt")
    assert len(pages) == 1 and "University of Edinburgh" in pages[0].text
    f = tmp_path / "two_pages.txt"
    f.write_bytes("Café page one.\fPage two text.".encode("cp1252"))  # non-UTF-8 + form feed
    pages = extract_pages(f)
    assert [p.page_number for p in pages] == [1, 2]
    assert "Café" in pages[0].text


def test_docx_ingestion_headings_become_sections():
    pages = extract_pages(DOCS / "deepmind_history.docx")
    _, _, sections = build_document_text(pages)
    titles = [s.title for s in sections]
    assert "Founding" in titles and "AlphaGo" in titles and "Google DeepMind" in titles


def test_empty_and_unsupported_files(tmp_path: Path):
    empty = tmp_path / "empty.txt"
    empty.write_text("   ")
    with pytest.raises(ExtractionError):
        extract_pages(empty)
    with pytest.raises(ExtractionError):
        extract_pages(tmp_path / "file.xyz")


def test_normalization_repairs_pdf_artifacts():
    raw = "Machine learn-\ning uses ﬁne\ntuned models.\n\n\nNew   paragraph."
    assert normalize_text(raw) == "Machine learning uses fine tuned models.\n\nNew paragraph."


def test_safe_filename_blocks_path_traversal():
    assert safe_filename("../../etc/passwd") == "passwd"
    assert safe_filename("C:\\Windows\\evil name?.pdf") == "evil_name_.pdf"
    assert safe_filename("résumé.docx") == "resume.docx"


def test_upload_validation_and_duplicates(container):
    proc = container.processor
    with pytest.raises(DocumentValidationError):
        proc.validate("malware.exe", 10)
    with pytest.raises(DocumentValidationError):
        proc.validate("big.pdf", (container.settings.max_upload_mb + 1) * 1024 * 1024)
    with pytest.raises(DocumentValidationError):
        proc.validate("empty.txt", 0)
    data = b"A short document about Alan Turing and the University of Manchester."
    doc, dup = proc.register_upload("turing.txt", data)
    doc2, dup2 = proc.register_upload("turing copy.txt", data)
    assert not dup and dup2 and doc.id == doc2.id
