"""Confidence estimation and insufficient-evidence detection.

The confidence is computed from measurable retrieval and evidence signals - it
is never produced by a language model:

    confidence = 0.30 * S_ret + 0.25 * S_cov + 0.15 * S_chain + 0.30 * S_ans

* S_ret   - best cosine similarity between the question and an evidence
            passage, rescaled from [0.10, 0.70] to [0, 1]
* S_cov   - share of question content terms that occur in the evidence
* S_chain - score of the best evidence chain (0 when no chain is used)
* S_ans   - answer support from the generator (reader span probability,
            extractive sentence score, ...); halved if a model abstained;
            0 if the answer could not be verified against the evidence

The answer is replaced by the insufficient-evidence message when any hard
check fails or the confidence is below ``min_confidence``. Hard checks:

* best passage similarity below ``min_semantic``;
* question-term coverage below ``min_coverage``;
* entity grounding: the question names entities (people, organisations,
  events, ...) but *none* of them occurs in the evidence - the evidence is
  then about something else, however similar it looks;
* the generator abstained or its answer could not be verified.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from app.core.domain import EvidenceChain
from app.generation.generator import EvidencePassage, GenerationResult
from app.preprocessing.tokenize import content_terms

WEIGHTS = {"retrieval": 0.30, "coverage": 0.25, "chain": 0.15, "answer_support": 0.30}


@dataclass
class ConfidenceReport:
    confidence: float
    components: dict[str, float]
    insufficient: bool
    reasons: list[str] = field(default_factory=list)
    coverage: dict[str, Any] = field(default_factory=dict)


def assess_confidence(question: str, evidence: list[EvidencePassage], chain: Optional[EvidenceChain],
                      result: GenerationResult, min_semantic: float, min_coverage: float,
                      min_confidence: float,
                      question_entities: Optional[list[dict[str, Any]]] = None) -> ConfidenceReport:
    q_terms = list(dict.fromkeys(content_terms(question, drop_question_words=True)))
    ev_terms: set[str] = set()
    for ev in evidence:
        ev_terms |= set(content_terms(ev.chunk.text))
    covered = [t for t in q_terms if t in ev_terms]
    coverage = len(covered) / len(q_terms) if q_terms else 0.0
    s_ret_raw = max((ev.semantic_score for ev in evidence), default=0.0)
    s_ret = min(1.0, max(0.0, (s_ret_raw - 0.10) / 0.60))
    s_chain = min(1.0, max(0.0, chain.score)) if chain is not None else 0.0
    s_ans = min(1.0, max(0.0, result.support_score)) if result.supported and not result.abstained else 0.0
    if result.model_abstained:
        s_ans *= 0.5
    components = {"retrieval": s_ret, "coverage": coverage, "chain": s_chain, "answer_support": s_ans}
    confidence = sum(WEIGHTS[k] * v for k, v in components.items())

    reasons: list[str] = []
    if not evidence:
        reasons.append("No relevant passages were retrieved.")
    else:
        if s_ret_raw < min_semantic:
            reasons.append(f"The best passage similarity ({s_ret_raw:.2f}) is below the threshold ({min_semantic:.2f}).")
        if coverage < min_coverage:
            missing = [t for t in q_terms if t not in ev_terms]
            reasons.append(f"The evidence covers only {coverage:.0%} of the question terms "
                           f"(missing: {', '.join(missing[:8])}).")
    # entity grounding: at least one named (non-date) question entity must appear in the evidence
    named = [e for e in (question_entities or []) if e.get("label") not in ("DATE", "PROPN")]
    if named and evidence:
        ev_text = " ".join(ev.chunk.text for ev in evidence).lower()
        def present(e: dict[str, Any]) -> bool:
            names = [e["text"], *[m["name"] for m in e.get("matches", [])]]
            return any(n and n.lower() in ev_text for n in names)
        if not any(present(e) for e in named):
            reasons.append("None of the entities named in the question ("
                           + ", ".join(f"'{e['text']}'" for e in named[:4]) + ") appears in the evidence.")
    if result.abstained or not result.answer:
        reasons.append("The answer generator found no answer in the evidence.")
    elif not result.supported:
        reasons.append("The generated answer could not be verified against the evidence.")
    if not reasons and confidence < min_confidence:
        reasons.append(f"Overall confidence ({confidence:.2f}) is below the threshold ({min_confidence:.2f}).")
    return ConfidenceReport(
        confidence=round(confidence, 4),
        components={k: round(v, 4) for k, v in components.items()},
        insufficient=bool(reasons),
        reasons=reasons,
        coverage={"question_terms": q_terms, "covered": covered,
                  "missing": [t for t in q_terms if t not in ev_terms], "ratio": round(coverage, 4)},
    )
