"""Chunking utility: token budget, overlap, sentence boundaries, exact offsets."""
from __future__ import annotations

import pytest

from app.core.domain import Document, PageText, Section
from app.preprocessing.chunker import Chunker, chunk_statistics
from app.preprocessing.sentences import split_sentences
from app.preprocessing.tokenize import count_tokens

TEXT = " ".join(f"Sentence number {i} talks about topic {i % 7} in some detail." for i in range(60))


def make_doc(text: str, pages: list[PageText] | None = None) -> Document:
    return Document(id="doc_test", name="t.txt", file_type="txt", text=text,
                    pages=pages or [PageText(1, text, 0)])


def test_chunks_respect_budget_and_are_exact_slices():
    chunker = Chunker(chunk_size=50, chunk_overlap=10)
    doc = make_doc(TEXT)
    chunks = chunker.chunk_document(doc)
    assert len(chunks) > 3
    for c in chunks:
        assert c.token_count <= 50
        assert doc.text[c.char_start:c.char_end] == c.text
        assert c.text.endswith(".")  # never cut inside a sentence here


def test_consecutive_chunks_overlap():
    chunker = Chunker(chunk_size=50, chunk_overlap=12)
    chunks = chunker.chunk_document(make_doc(TEXT))
    for a, b in zip(chunks, chunks[1:]):
        assert b.char_start < a.char_end, "expected overlapping sentences"
        overlap = TEXT[b.char_start:a.char_end]
        assert count_tokens(overlap) <= 12


def test_no_overlap_option():
    chunks = Chunker(chunk_size=50, chunk_overlap=0).chunk_document(make_doc(TEXT))
    for a, b in zip(chunks, chunks[1:]):
        assert b.char_start >= a.char_end


def test_overlong_sentence_is_split():
    long_sentence = " ".join(["word"] * 130) + "."
    chunks = Chunker(chunk_size=50, chunk_overlap=0).chunk_document(make_doc(long_sentence))
    assert len(chunks) == 3 and all(c.token_count <= 50 for c in chunks)


def test_page_numbers_and_sections():
    p1, p2 = "First page sentence one. First page sentence two.", "Second page sentence."
    text = p1 + "\n\n" + p2
    doc = make_doc(text, [PageText(1, p1, 0), PageText(2, p2, len(p1) + 2)])
    doc.sections = [Section("Intro", 0, len(p1)), Section("Body", len(p1) + 2, len(text))]
    chunks = Chunker(chunk_size=200, chunk_overlap=0).chunk_document(doc)
    # sections are merged when tiny, but page metadata stays correct
    assert chunks[0].page_number == 1 and chunks[-1].page_end == 2


def test_sentence_splitter_handles_abbreviations():
    text = "Dr. Smith works at the U.S. office. He met J. K. Rowling in 2001. Prices rose e.g. sharply."
    sents = [text[s:e] for s, e in split_sentences(text)]
    assert sents == ["Dr. Smith works at the U.S. office.", "He met J. K. Rowling in 2001.",
                     "Prices rose e.g. sharply."]


def test_invalid_parameters():
    with pytest.raises(ValueError):
        Chunker(chunk_size=100, chunk_overlap=100)
    with pytest.raises(ValueError):
        Chunker(chunk_size=5)


def test_statistics():
    chunks = Chunker(60, 10).chunk_document(make_doc(TEXT))
    stats = chunk_statistics(chunks)
    assert stats["total_chunks"] == len(chunks) and stats["max_tokens"] <= 60
