"""Text cleaning applied to every extracted page before chunking."""
from __future__ import annotations

import re
import unicodedata

_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f​﻿]")
_LINEBREAK_HYPHEN = re.compile(r"([a-z])-\n([a-z])")
_SPACES = re.compile(r"[ \t  - 　]+")
_BLANK_LINES = re.compile(r"\n\s*\n+")


def normalize_text(text: str) -> str:
    """Normalize unicode, repair PDF line wrapping and collapse whitespace.

    * NFKC unicode normalization (ligatures such as "ﬁ" become "fi")
    * removal of control / zero-width characters
    * de-hyphenation of words broken across lines ("infor-\\nmation")
    * lines inside a paragraph are joined; paragraph breaks are preserved
    * runs of spaces collapse to one space
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _CONTROL.sub(" ", text)
    text = _LINEBREAK_HYPHEN.sub(r"\1\2", text)
    paragraphs = []
    for para in _BLANK_LINES.split(text):
        lines = [_SPACES.sub(" ", ln).strip() for ln in para.split("\n")]
        joined = " ".join(ln for ln in lines if ln)
        if joined:
            paragraphs.append(joined)
    return "\n\n".join(paragraphs)


def clean_inline(text: str) -> str:
    """Collapse all whitespace (including newlines) to single spaces."""
    return _SPACES.sub(" ", text.replace("\n", " ")).strip()
