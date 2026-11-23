"""Lightweight tokenization utilities (no model required).

* ``word_tokens`` / ``count_tokens`` – word-level tokens used for chunk sizes.
* ``content_terms`` – lower-cased, stop-word-filtered, lightly stemmed terms used
  by BM25, question-term coverage and citation support checks.
"""
from __future__ import annotations

import re
from functools import lru_cache

TOKEN_RE = re.compile(r"\w+(?:[-'’]\w+)*|[^\w\s]")
WORD_RE = re.compile(r"\w+(?:[-'’]\w+)*")

STOPWORDS = frozenset(
    """a about above after again against all also am an and any are aren't as at be because been before being
    below between both but by can could did didn't do does doesn't doing don't down during each few for from
    further had hadn't has hasn't have haven't having he he'd he'll he's her here here's hers herself him himself
    his how how's i i'd i'll i'm i've if in into is isn't it it's its itself let's me more most mustn't my myself
    no nor not of off on once only or other ought our ours ourselves out over own same shan't she she'd she'll
    she's should shouldn't so some such than that that's the their theirs them themselves then there there's
    these they they'd they'll they're they've this those through to too under until up very was wasn't we we'd
    we'll we're we've were weren't what what's when when's where where's which while who who's whom why why's
    with won't would wouldn't you you'd you'll you're you've your yours yourself yourselves
    many much whose one ones s t also within upon via per among""".split()
)

# Words that make up the *form* of a question but are not information needs.
QUESTION_WORDS = frozenset(
    "what which who whom whose when where why how name named list tell give describe explain does did do "
    "is are was were kind type year".split()
)


@lru_cache(maxsize=50_000)
def simple_stem(word: str) -> str:
    """A tiny suffix-stripping stemmer (consistent for queries and documents)."""
    if len(word) <= 3 or not word.isalpha():
        return word
    for suffix, replacement in (
        ("ational", "ate"), ("ization", "ize"), ("iveness", "ive"), ("fulness", "ful"),
        ("ies", "y"), ("sses", "ss"), ("ingly", ""), ("edly", ""), ("ing", ""), ("ed", ""),
        ("ly", ""), ("es", ""), ("s", ""),
    ):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            if suffix == "s" and word.endswith(("ss", "us", "is")):
                return word
            return word[: -len(suffix)] + replacement
    return word


def word_tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(text)


def count_tokens(text: str) -> int:
    return len(TOKEN_RE.findall(text))


def token_spans(text: str) -> list[tuple[int, int]]:
    return [(m.start(), m.end()) for m in TOKEN_RE.finditer(text)]


def content_terms(text: str, drop_question_words: bool = False) -> list[str]:
    """Lower-cased, stemmed tokens without stop words or punctuation."""
    terms = []
    for raw in WORD_RE.findall(text.lower()):
        word = raw.replace("’", "'")
        if word in STOPWORDS or (drop_question_words and word in QUESTION_WORDS):
            continue
        if len(word) == 1 and not word.isdigit():
            continue
        terms.append(simple_stem(word))
    return terms
