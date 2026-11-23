"""Citation construction and validation.

Every claim of the answer (the answer itself and every ``fact`` reasoning
step) carries citation numbers. For each (claim, passage) pair the builder
locates the exact supporting sentence and computes a support score:

* 1.0  if the claim (e.g. the answer span) occurs verbatim in the sentence;
* otherwise the share of the claim's content terms found in the sentence.

Citations whose support is below ``min_support`` are removed - a passage is
never cited for a claim it does not support.
"""
from __future__ import annotations

from typing import Optional

from app.core.domain import Citation
from app.generation.generator import (EvidencePassage, GenerationResult, best_sentence_for, evidence_sentences,
                                      normalize_answer)


class CitationBuilder:
    def __init__(self, min_support: float = 0.3) -> None:
        self.min_support = min_support

    def build(self, result: GenerationResult, evidence: list[EvidencePassage]) -> tuple[list[Citation], list[str]]:
        by_index = {ev.index: ev for ev in evidence}
        sentences = evidence_sentences(evidence)
        citations: dict[int, Citation] = {}
        removed: list[str] = []

        def cite(claim: str, idx: int, is_answer: bool) -> bool:
            ev = by_index.get(idx)
            if ev is None:
                removed.append(f"Citation [{idx}] does not refer to a retrieved passage.")
                return False
            sent, support = best_sentence_for(claim, sentences, {idx})
            if is_answer and normalize_answer(result.answer) and normalize_answer(result.answer) in normalize_answer(
                    ev.chunk.text):
                sent_ans, _ = best_sentence_for(result.answer, sentences, {idx})
                if sent_ans is not None:
                    sent, support = sent_ans, 1.0
            if sent is None or support < self.min_support:
                removed.append(f"Passage [{idx}] was not cited: it does not support \"{claim[:80]}\" "
                               f"(support {support:.2f}).")
                return False
            if idx not in citations:
                citations[idx] = Citation(
                    index=idx, chunk_id=ev.chunk.id, document_id=ev.chunk.document_id,
                    document_name=ev.chunk.document_name, page_number=ev.chunk.page_number,
                    page_end=ev.chunk.page_end, section=ev.chunk.section, url=ev.chunk.url,
                    source_type=ev.chunk.source_type, passage_text=ev.chunk.text, supporting_sentence=sent.text,
                    sentence_start=sent.start, sentence_end=sent.end, retrieval_score=ev.hybrid_score,
                    graph_score=ev.graph_score, gnn_score=ev.gnn_score, support_score=round(support, 4),
                    claim=claim,
                )
            return True

        if result.answer and not result.abstained:
            answer_claim = result.answer if result.answer.lower() not in ("yes", "no") else (
                result.answer_sentence or result.answer)
            result.answer_citation_indices = [
                i for i in result.answer_citation_indices
                if cite(answer_claim, i, is_answer=result.answer.lower() not in ("yes", "no"))
            ]
        for step in result.reasoning_steps:
            if step.kind != "fact":
                continue
            step.citation_indices = [i for i in step.citation_indices if cite(step.text, i, is_answer=False)]
        # inference steps may only reference passages that are actually cited
        for step in result.reasoning_steps:
            if step.kind == "inference":
                step.citation_indices = [i for i in step.citation_indices if i in citations]
        return [citations[i] for i in sorted(citations)], removed


def answer_has_citation(citations: list[Citation], answer: Optional[str]) -> bool:
    if not answer:
        return False
    return any(normalize_answer(answer) in normalize_answer(c.passage_text) for c in citations) or bool(citations)
