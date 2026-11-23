"""Evaluation metrics (all computed from system outputs, nothing estimated).

Answer quality
    exact_match  - 1 if the normalized prediction equals a normalized gold answer
    f1           - token-level F1 (SQuAD definition), max over gold aliases
Retrieval / ranking (passage level)
    precision_at_k, recall_at_k - share of the top-k that is gold / share of gold in the top-k
    mrr                         - 1 / rank of the first gold passage (0 if none retrieved)
    evidence_recall             - share of gold passages contained in the evidence given to the generator
Citations
    citation_coverage  - share of gold passages that are cited
    citation_precision - share of cited passages that are gold
Unanswerable questions are scored as correct (EM = F1 = 1) when the system
abstains with the insufficient-evidence message, and 0 otherwise.
"""
from __future__ import annotations

from typing import Iterable, Optional

from app.generation.generator import normalize_answer, token_f1


def exact_match(prediction: str, golds: Iterable[str]) -> float:
    p = normalize_answer(prediction)
    return float(any(p == normalize_answer(g) for g in golds if g))


def f1(prediction: str, golds: Iterable[str]) -> float:
    golds = [g for g in golds if g]
    return max((token_f1(prediction, g) for g in golds), default=0.0)


def precision_at_k(ranked: list[str], gold: set[str], k: int) -> Optional[float]:
    if not gold:
        return None
    top = ranked[:k]
    return sum(1 for r in top if r in gold) / k


def recall_at_k(ranked: list[str], gold: set[str], k: int) -> Optional[float]:
    if not gold:
        return None
    return sum(1 for r in ranked[:k] if r in gold) / len(gold)


def mrr(ranked: list[str], gold: set[str]) -> Optional[float]:
    if not gold:
        return None
    for i, r in enumerate(ranked, start=1):
        if r in gold:
            return 1.0 / i
    return 0.0


def evidence_recall(evidence_ids: list[str], gold: set[str]) -> Optional[float]:
    if not gold:
        return None
    return len(set(evidence_ids) & gold) / len(gold)


def citation_scores(cited_ids: list[str], gold: set[str]) -> tuple[Optional[float], Optional[float]]:
    if not gold:
        return None, None
    cited = set(cited_ids)
    coverage = len(cited & gold) / len(gold)
    precision = (len(cited & gold) / len(cited)) if cited else 0.0
    return coverage, precision


def mean(values: Iterable[Optional[float]]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    return round(sum(vals) / len(vals), 4) if vals else None
