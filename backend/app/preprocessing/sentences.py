"""Rule-based sentence splitter that returns character spans.

A model-free splitter keeps chunking fast and deterministic. It protects common
abbreviations (Dr., U.S., e.g.) and initials (J. K. Rowling) from being treated
as sentence ends.
"""
from __future__ import annotations

import re

ABBREVIATIONS = frozenset(
    """mr mrs ms dr prof sr jr st vs etc e.g i.e inc ltd co corp no nos fig figs al approx dept est gen gov
    lt mt rev sgt col capt cf ed eds vol vols pp jan feb mar apr jun jul aug sep sept oct nov dec u.s u.k
    u.s.a u.n e.u ph.d b.tech m.tech a.m p.m""".split()
)

_BOUNDARY = re.compile(r"[.!?]+[\"'”’)\]]*(?=\s+)")
_PARAGRAPH = re.compile(r"\n\s*\n")


def _is_false_boundary(text: str, punct_start: int, next_start: int) -> bool:
    # word immediately before the punctuation
    m = re.search(r"([\w.]+)$", text[:punct_start + 1])
    word = (m.group(1) if m else "").lower().rstrip(".")
    if word in ABBREVIATIONS:
        return True
    # single capital initial, e.g. "J. K. Rowling"
    if re.fullmatch(r"[a-z]", word) and punct_start > 0 and text[punct_start - 1].isupper():
        return True
    # next sentence must not start with a lower-case letter
    if next_start < len(text) and text[next_start].islower():
        return True
    return False


def split_sentences(text: str) -> list[tuple[int, int]]:
    """Split ``text`` into sentences, returning ``(start, end)`` character spans."""
    spans: list[tuple[int, int]] = []
    para_start = 0
    paragraph_bounds = []
    for m in _PARAGRAPH.finditer(text):
        paragraph_bounds.append((para_start, m.start()))
        para_start = m.end()
    paragraph_bounds.append((para_start, len(text)))

    for p_start, p_end in paragraph_bounds:
        para = text[p_start:p_end]
        sent_start = 0
        for m in _BOUNDARY.finditer(para):
            end = m.end()
            next_start = end
            while next_start < len(para) and para[next_start].isspace():
                next_start += 1
            if next_start >= len(para):
                break
            if _is_false_boundary(para, m.start(), next_start):
                continue
            segment = para[sent_start:end]
            if segment.strip():
                lead = len(segment) - len(segment.lstrip())
                spans.append((p_start + sent_start + lead, p_start + end))
            sent_start = next_start
        tail = para[sent_start:]
        if tail.strip():
            lead = len(tail) - len(tail.lstrip())
            spans.append((p_start + sent_start + lead, p_start + len(para.rstrip())))
    return spans


def sentences(text: str) -> list[str]:
    return [text[s:e] for s, e in split_sentences(text)]
