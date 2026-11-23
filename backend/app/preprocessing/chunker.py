"""Sentence-aware chunking with token budget and overlap.

Algorithm (per section of a document):

1. Split the text into sentences (character spans).
2. A sentence longer than ``chunk_size`` tokens is split into token windows
   (the only case where a sentence is cut).
3. Greedily pack consecutive sentences until adding the next one would exceed
   ``chunk_size`` tokens -> emit a chunk.
4. The next chunk starts with the trailing sentences of the previous chunk
   whose total length is at most ``chunk_overlap`` tokens (overlap), so facts
   that straddle a boundary appear complete in at least one chunk.

Chunk text is always an exact slice of the document text, so character
offsets in citations point at the original (normalized) text.
"""
from __future__ import annotations

from dataclasses import dataclass

from app.core.domain import Document, Section, TextChunk
from app.preprocessing.sentences import split_sentences
from app.preprocessing.tokenize import count_tokens, token_spans


@dataclass
class _Unit:
    start: int
    end: int
    tokens: int


class Chunker:
    def __init__(self, chunk_size: int = 400, chunk_overlap: int = 80) -> None:
        if chunk_size < 20:
            raise ValueError("chunk_size must be at least 20 tokens")
        if not 0 <= chunk_overlap < chunk_size:
            raise ValueError("chunk_overlap must be >= 0 and smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    # ------------------------------------------------------------------ spans
    def _units(self, text: str) -> list[_Unit]:
        units: list[_Unit] = []
        for start, end in split_sentences(text):
            n = count_tokens(text[start:end])
            if n <= self.chunk_size:
                units.append(_Unit(start, end, n))
                continue
            # Hard split of an over-long sentence into windows of chunk_size tokens.
            spans = [(s + start, e + start) for s, e in token_spans(text[start:end])]
            for i in range(0, len(spans), self.chunk_size):
                window = spans[i : i + self.chunk_size]
                units.append(_Unit(window[0][0], window[-1][1], len(window)))
        return units

    def split_text(self, text: str) -> list[tuple[int, int, int]]:
        """Return ``(start, end, token_count)`` spans of chunks within ``text``."""
        units = self._units(text)
        chunks: list[tuple[int, int, int]] = []
        i = 0
        while i < len(units):
            j, tokens = i, 0
            while j < len(units) and (j == i or tokens + units[j].tokens <= self.chunk_size):
                tokens += units[j].tokens
                j += 1
            chunks.append((units[i].start, units[j - 1].end, tokens))
            if j >= len(units):
                break
            # step back to create the overlap (never back to the current start)
            k, overlap = j, 0
            while k - 1 > i and overlap + units[k - 1].tokens <= self.chunk_overlap:
                k -= 1
                overlap += units[k].tokens
            i = k
        return chunks

    # -------------------------------------------------------------- documents
    @staticmethod
    def merge_small_sections(sections: list[Section], text: str, min_tokens: int = 25) -> list[Section]:
        """Merge tiny sections into a neighbour so that no chunk is just a heading.

        A tiny leading section (e.g. a title) is merged *forward* into the next
        section; any other tiny section is appended to the *previous* section,
        whose title is extended ("Founding / Acquisition by Google").
        """
        merged: list[Section] = []
        pending_start: int | None = None
        for i, sec in enumerate(sections):
            start = pending_start if pending_start is not None else sec.char_start
            tiny = count_tokens(text[sec.char_start:sec.char_end]) < min_tokens
            if tiny and not merged and i < len(sections) - 1:
                pending_start = start
                continue
            if tiny and merged:
                prev = merged[-1]
                title = " / ".join(t for t in (prev.title, sec.title) if t) or None
                merged[-1] = Section(title, prev.char_start, sec.char_end)
                continue
            merged.append(Section(sec.title, start, sec.char_end))
            pending_start = None
        return merged

    def chunk_document(self, doc: Document) -> list[TextChunk]:
        sections = self.merge_small_sections(doc.sections, doc.text) if doc.sections else [Section(None, 0, len(doc.text))]
        chunks: list[TextChunk] = []
        for section in sections:
            section_text = doc.text[section.char_start : section.char_end]
            for start, end, n_tokens in self.split_text(section_text):
                abs_start, abs_end = section.char_start + start, section.char_start + end
                has_pages = bool(doc.pages)
                chunks.append(
                    TextChunk(
                        id=f"{doc.id}:c{len(chunks):04d}",
                        document_id=doc.id,
                        document_name=doc.name,
                        chunk_index=len(chunks),
                        text=doc.text[abs_start:abs_end],
                        page_number=doc.page_for_offset(abs_start) if has_pages else None,
                        page_end=doc.page_for_offset(max(abs_start, abs_end - 1)) if has_pages else None,
                        char_start=abs_start,
                        char_end=abs_end,
                        token_count=n_tokens,
                        section=section.title,
                        url=doc.url,
                        source_type=doc.source_type,
                    )
                )
        return chunks


def chunk_statistics(chunks: list[TextChunk]) -> dict[str, float]:
    if not chunks:
        return {"total_chunks": 0, "avg_tokens": 0.0, "min_tokens": 0, "max_tokens": 0, "avg_chars": 0.0}
    tokens = [c.token_count for c in chunks]
    return {
        "total_chunks": len(chunks),
        "avg_tokens": round(sum(tokens) / len(tokens), 1),
        "min_tokens": min(tokens),
        "max_tokens": max(tokens),
        "avg_chars": round(sum(len(c.text) for c in chunks) / len(chunks), 1),
    }
