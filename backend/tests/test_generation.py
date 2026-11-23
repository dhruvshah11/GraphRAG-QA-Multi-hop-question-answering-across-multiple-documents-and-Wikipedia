"""Answer generation, grounding, citations and insufficient-evidence handling."""
from __future__ import annotations

import pytest

from app.core.domain import TextChunk
from app.generation.answer_types import expected_answer_type
from app.generation.generator import INSUFFICIENT_MESSAGE, EvidencePassage, GenerationResult, token_f1
from app.evaluation import metrics as M


def ask(container, corpus, question, strategy="graph", backend="auto"):
    return container.pipeline.ask(question, strategy=strategy, backend=backend, corpus=corpus, persist=False)


@pytest.mark.parametrize("qid", ["demo-01", "demo-02", "demo-03", "demo-06", "demo-07"])
def test_multi_document_answers(container, demo_corpus, demo_questions, qid):
    q = demo_questions[qid]
    out = ask(container, demo_corpus, q["question"])
    assert not out["insufficient_evidence"], out["insufficient_reasons"]
    assert M.exact_match(out["answer"], [q["answer"], *q["answer_aliases"]]) == 1.0, out["answer"]
    assert 0 < out["confidence"] <= 1


def test_citations_point_at_exact_supporting_text(container, demo_corpus, demo_questions):
    out = ask(container, demo_corpus, demo_questions["demo-02"]["question"])
    assert out["citations"]
    for c in out["citations"]:
        assert c["passage_text"][c["sentence_start"]:c["sentence_end"]].strip() == c["supporting_sentence"].strip()
        assert c["support_score"] >= 0.3
    # the answer itself is cited by a passage that contains it
    answer_cites = [c for c in out["citations"] if c["index"] in out["answer_citations"]]
    assert answer_cites and all("london" in c["passage_text"].lower() for c in answer_cites)
    # facts and inferences are labelled; facts always carry citations
    kinds = {s["kind"] for s in out["reasoning_steps"]}
    assert kinds == {"fact", "inference"}
    assert all(s["citations"] for s in out["reasoning_steps"] if s["kind"] == "fact")


@pytest.mark.parametrize("question", ["Who won the 2030 FIFA World Cup final?",
                                      "What is the boiling point of liquid nitrogen in Kelvin?"])
def test_insufficient_evidence(container, demo_corpus, question):
    out = ask(container, demo_corpus, question)
    assert out["insufficient_evidence"] is True
    assert out["answer"] == INSUFFICIENT_MESSAGE
    assert out["citations"] == [] and out["insufficient_reasons"]


def test_extractive_backend_works_without_models(container, demo_corpus):
    out = ask(container, demo_corpus, "In which city was the AlphaGo match against Lee Sedol held?", backend="extractive")
    assert out["generator_backend"] == "extractive"
    assert "seoul" in (out["answer"] + (out["answer_sentence"] or "")).lower()


def test_ungrounded_answers_are_rejected(container):
    chunk = TextChunk("d:c0", "d", "d.txt", 0, "Paris is the capital of France.", 1, 1, 0, 31, 7)
    ev = [EvidencePassage(1, chunk, 0.8, 0.8, None, None, 0.8)]
    fake = GenerationResult("Berlin", None, [1], [], 0.9, "test", True)
    grounded = container.generator._ground("What is the capital of France?", fake, ev)
    assert grounded.supported is False
    ok = container.generator._ground("What is the capital of France?", GenerationResult("Paris", None, [], [], 0.9, "t", True), ev)
    assert ok.supported and ok.answer_citation_indices == [1]


def test_answer_types_and_metrics():
    assert expected_answer_type("When was X founded?").kind == "date"
    assert expected_answer_type("Which university did she attend?").kind == "organization"
    assert expected_answer_type("Were A and B both American?").kind == "yesno"
    assert token_f1("the University of Edinburgh", "University of Edinburgh") == 1.0
    assert M.precision_at_k(["a", "b", "c"], {"a", "c"}, 2) == 0.5
    assert M.recall_at_k(["a", "b", "c"], {"a", "c"}, 3) == 1.0
    assert M.mrr(["x", "a"], {"a"}) == 0.5
    assert M.citation_scores(["a", "z"], {"a", "b"}) == (0.5, 0.5)
