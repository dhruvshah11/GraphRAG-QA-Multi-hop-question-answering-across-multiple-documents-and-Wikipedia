"""Entity name normalization and label/category mapping."""
from __future__ import annotations

import re
import unicodedata

# spaCy labels kept for the evidence graph. Numeric labels (CARDINAL, MONEY,
# PERCENT, QUANTITY, ORDINAL, TIME) are dropped: they rarely identify the same
# real-world thing across passages and would create meaningless bridges.
KEEP_LABELS = frozenset(
    {"PERSON", "ORG", "GPE", "LOC", "FAC", "NORP", "EVENT", "WORK_OF_ART", "LAW", "PRODUCT", "LANGUAGE", "DATE"}
)

CATEGORY_BY_LABEL = {
    "PERSON": "PERSON",
    "ORG": "ORGANIZATION",
    "GPE": "LOCATION",
    "LOC": "LOCATION",
    "FAC": "LOCATION",
    "NORP": "GROUP",
    "EVENT": "EVENT",
    "WORK_OF_ART": "WORK_OF_ART",
    "LAW": "LAW",
    "PRODUCT": "PRODUCT",
    "LANGUAGE": "LANGUAGE",
    "DATE": "DATE",
    "MISC": "MISC",
}

_MONTHS = "january|february|march|april|may|june|july|august|september|october|november|december"
_YEAR = re.compile(r"\b(1[0-9]{3}|20[0-9]{2})s?\b")
_MONTH = re.compile(rf"\b({_MONTHS})\b", re.IGNORECASE)
_ACRONYM_WITH_DOTS = re.compile(r"^(?:[A-Za-z]\.){2,}[A-Za-z]?\.?$")
_LEADING_ARTICLE = re.compile(r"^(the|a|an)\s+", re.IGNORECASE)
_POSSESSIVE = re.compile(r"(?:'s|')$")
_PUNCT = re.compile(r"[\"“”‘’`´(),;:!?\[\]{}<>|*_#]")


def category_for(label: str) -> str:
    return CATEGORY_BY_LABEL.get(label, "MISC")


def clean_surface(text: str) -> str:
    """Tidy a mention for display: unicode, whitespace, article, possessive."""
    text = unicodedata.normalize("NFKC", text).replace("’", "'").strip()
    text = re.sub(r"\s+", " ", text)
    text = _LEADING_ARTICLE.sub("", text)
    text = _POSSESSIVE.sub("", text)
    return text.strip(" .,-")


def normalize_entity(text: str) -> str:
    """Canonical key used for entity identity (case/punctuation insensitive)."""
    raw = unicodedata.normalize("NFKC", text).strip()
    if _ACRONYM_WITH_DOTS.match(raw):  # "U.S." -> "us"
        return raw.replace(".", "").casefold()
    text = clean_surface(text)
    if _ACRONYM_WITH_DOTS.match(text):
        text = text.replace(".", "")
    text = _PUNCT.sub(" ", text)
    text = re.sub(r"\.(?=\s|$)", "", text)
    text = text.replace("&", " and ")
    text = re.sub(r"\s+", " ", text).strip()
    return text.casefold()


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", text.casefold()).strip("_")
    return slug[:80] or "x"


def entity_id_for(normalized: str) -> str:
    return f"ent:{slugify(normalized)}"


def is_valid_mention(text: str, label: str) -> bool:
    """Filter out mentions that are too short, too long or non-informative."""
    cleaned = clean_surface(text)
    if len(cleaned) < 2 or len(cleaned) > 80:
        return False
    if len(cleaned.split()) > 8:
        return False
    if label == "DATE":
        # keep calendar dates with a year or month; drop "today", "last week", "two years"
        return bool(_YEAR.search(cleaned) or _MONTH.search(cleaned))
    if not any(ch.isalpha() for ch in cleaned):
        return False
    if cleaned.lower() in {"he", "she", "it", "they", "this", "that", "these", "those"}:
        return False
    return True
