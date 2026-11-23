"""Evidence-grounded answer generation with pluggable backends.

Backends (all subclasses of :class:`GenerationBackend`):

=================  ==========================================================
``extractive``     model-free: scores evidence sentences (semantic similarity,
                   question-term overlap, expected answer type) and extracts
                   the answer entity. Always available.
``hf_reader``      local extractive QA model (SQuAD 2.0, can abstain).
``hf_generator``   local Flan-T5 (short generative answers, yes/no).
``ollama``         optional local LLM through Ollama's /api/chat (JSON mode).
``anthropic``      optional Claude API backend (official SDK, JSON schema).
=================  ==========================================================

Whatever the backend, the answer is accepted only if it is *grounded*: a span
answer must occur in a cited evidence passage (or match an evidence sentence
with token-F1 >= 0.6). Ungrounded output is rejected and the pipeline falls
back to the extractive backend. Reasoning steps are labelled ``fact``
(a quoted evidence sentence with its citation) or ``inference`` (a step that
combines facts), so retrieved evidence is always distinguishable from
generated explanation.
"""
from __future__ import annotations

import json
import re
import string
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

import numpy as np

from app.core.domain import EvidenceChain, TextChunk
from app.core.logging_setup import get_logger
from app.generation.answer_types import AnswerType, expected_answer_type
from app.generation.hf_models import ExtractiveReader, Seq2SeqGenerator
from app.preprocessing.sentences import split_sentences
from app.preprocessing.tokenize import content_terms

if TYPE_CHECKING:  # pragma: no cover
    from app.core.config import Settings
    from app.embeddings.embedder import EmbeddingGenerator
    from app.entities.extractor import EntityExtractor

log = get_logger(__name__)

INSUFFICIENT_MESSAGE = "Insufficient evidence found to answer this question reliably."


# ================================================================ data types
@dataclass
class EvidencePassage:
    index: int  # 1-based citation number
    chunk: TextChunk
    hybrid_score: float
    semantic_score: float
    graph_score: Optional[float]
    gnn_score: Optional[float]
    final_score: float
    chain_step: Optional[int] = None
    via_entity_name: Optional[str] = None
    reason: str = ""

    @property
    def title(self) -> str:
        page = f", p. {self.chunk.page_number}" if self.chunk.page_number else ""
        section = f" — {self.chunk.section}" if self.chunk.section else ""
        return f"{self.chunk.document_name}{section}{page}"


@dataclass
class ReasoningStep:
    step: int
    kind: str  # fact | inference
    text: str
    citation_indices: list[int] = field(default_factory=list)


@dataclass
class GenerationResult:
    answer: str
    answer_sentence: Optional[str]
    answer_citation_indices: list[int]
    reasoning_steps: list[ReasoningStep]
    support_score: float
    backend: str
    supported: bool
    abstained: bool = False
    warnings: list[str] = field(default_factory=list)
    raw_output: Optional[str] = None
    model_abstained: bool = False  # a model backend said "no answer" before a fallback answered

    @property
    def reasoning_summary(self) -> str:
        parts = []
        for s in self.reasoning_steps:
            cites = "".join(f"[{i}]" for i in s.citation_indices)
            label = "Evidence" if s.kind == "fact" else "Inference"
            parts.append(f"{label} {cites}: {s.text}".replace(" :", ":"))
        return " ".join(parts)


# ============================================================ text helpers
def normalize_answer(text: str) -> str:
    """SQuAD-style normalization (lower, no punctuation/articles/extra spaces)."""
    text = text.lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def token_f1(pred: str, gold: str) -> float:
    p, g = normalize_answer(pred).split(), normalize_answer(gold).split()
    if not p or not g:
        return float(p == g)
    common = Counter(p) & Counter(g)
    same = sum(common.values())
    if same == 0:
        return 0.0
    precision, recall = same / len(p), same / len(g)
    return 2 * precision * recall / (precision + recall)


@dataclass
class EvidenceSentence:
    evidence_index: int  # 1-based
    start: int
    end: int
    text: str


def evidence_sentences(evidence: list[EvidencePassage]) -> list[EvidenceSentence]:
    out = []
    for ev in evidence:
        for s, e in split_sentences(ev.chunk.text):
            text = ev.chunk.text[s:e].strip()
            if len(text) >= 3:
                out.append(EvidenceSentence(ev.index, s, e, text))
    return out


def locate_answer(answer: str, evidence: list[EvidencePassage]) -> Optional[tuple[int, int, int]]:
    """Find ``answer`` in the evidence: (evidence index, start, end) or None."""
    if not answer.strip():
        return None
    pattern = re.compile(re.escape(answer.strip()), re.IGNORECASE)
    for ev in evidence:
        m = pattern.search(ev.chunk.text)
        if m:
            return ev.index, m.start(), m.end()
    target = normalize_answer(answer)
    if not target:
        return None
    for ev in evidence:
        if target in normalize_answer(ev.chunk.text):
            return ev.index, 0, 0
    return None


def best_sentence_for(claim: str, sentences: list[EvidenceSentence],
                      restrict_to: Optional[set[int]] = None) -> tuple[Optional[EvidenceSentence], float]:
    """Evidence sentence that best supports ``claim`` (substring > token overlap)."""
    claim_norm = normalize_answer(claim)
    claim_terms = set(content_terms(claim))
    best, best_score = None, 0.0
    for s in sentences:
        if restrict_to and s.evidence_index not in restrict_to:
            continue
        if claim_norm and claim_norm in normalize_answer(s.text):
            score = 1.0
        elif claim_terms:
            score = len(claim_terms & set(content_terms(s.text))) / len(claim_terms)
        else:
            score = 0.0
        if score > best_score:
            best, best_score = s, score
    return best, best_score


# ================================================================ backends
class GenerationBackend(ABC):
    name = "base"

    @abstractmethod
    def available(self) -> bool: ...

    @abstractmethod
    def generate(self, question: str, evidence: list[EvidencePassage], chain: Optional[EvidenceChain],
                 atype: AnswerType) -> GenerationResult: ...


class ExtractiveBackend(GenerationBackend):
    """Model-free sentence scoring + answer-type entity extraction."""

    name = "extractive"

    def __init__(self, embedder: "EmbeddingGenerator", extractor: "EntityExtractor") -> None:
        self.embedder = embedder
        self.extractor = extractor

    def available(self) -> bool:
        return True

    def score_sentences(self, question: str, sentences: list[EvidenceSentence],
                        atype: AnswerType) -> tuple[np.ndarray, list[list[tuple[str, str]]]]:
        q_terms = set(content_terms(question, drop_question_words=True))
        texts = [s.text for s in sentences]
        sims = self.embedder.encode(texts) @ self.embedder.encode_query(question)
        overlaps = np.array([len(set(content_terms(t)) & q_terms) / max(1, len(q_terms)) for t in texts])
        pre = 0.6 * sims + 0.4 * overlaps
        top = list(np.argsort(-pre)[:12])
        ents_by_sentence: list[list[tuple[str, str]]] = [[] for _ in texts]
        for i, ents in zip(top, self.extractor.entities_in_texts([texts[i] for i in top])):
            ents_by_sentence[i] = ents
        q_lower = question.lower()
        type_match = np.zeros(len(texts))
        for i, ents in enumerate(ents_by_sentence):
            if any(lbl in atype.ner_labels and txt.lower() not in q_lower for txt, lbl in ents):
                type_match[i] = 1.0
        return 0.5 * sims + 0.3 * overlaps + 0.2 * type_match, ents_by_sentence

    def generate(self, question: str, evidence: list[EvidencePassage], chain: Optional[EvidenceChain],
                 atype: AnswerType) -> GenerationResult:
        sentences = evidence_sentences(evidence)
        if not sentences:
            return GenerationResult("", None, [], [], 0.0, self.name, False, abstained=True)
        scores, ents = self.score_sentences(question, sentences, atype)
        best = int(np.argmax(scores))
        sent = sentences[best]
        q_lower = question.lower()
        short = next((txt for txt, lbl in ents[best] if lbl in atype.ner_labels and txt.lower() not in q_lower),
                     None)
        answer = short or sent.text
        warnings = []
        if atype.kind == "yesno":
            warnings.append("Yes/no question: the extractive backend returns the most relevant evidence sentence "
                            "instead of a yes/no verdict.")
        return GenerationResult(
            answer=answer, answer_sentence=sent.text, answer_citation_indices=[sent.evidence_index],
            reasoning_steps=[], support_score=float(np.clip(scores[best], 0, 1)), backend=self.name,
            supported=True, warnings=warnings,
        )


class ReaderBackend(GenerationBackend):
    """Extractive QA model over each passage and over the concatenated chain."""

    name = "hf_reader"

    def __init__(self, reader: ExtractiveReader, null_threshold: float = 0.3) -> None:
        self.reader = reader
        self.null_threshold = null_threshold

    def available(self) -> bool:
        return self.reader.available

    def generate(self, question: str, evidence: list[EvidencePassage], chain: Optional[EvidenceChain],
                 atype: AnswerType) -> GenerationResult:
        contexts = [ev.chunk.text for ev in evidence]
        chain_evs = sorted([ev for ev in evidence if ev.chain_step], key=lambda ev: ev.chain_step or 0)
        offsets: list[tuple[int, int]] = []
        if len(chain_evs) >= 2:
            combined, pos = "", 0
            for ev in chain_evs:
                if combined:
                    combined += "\n"
                start = len(combined)
                combined += ev.chunk.text
                offsets.append((ev.index, start))
            contexts.append(combined)
        spans = self.reader.answer(question, contexts)
        # a span that merely repeats part of the question is never the answer
        q_norm = normalize_answer(question)
        spans = [sp for sp in spans if sp.text and normalize_answer(sp.text) not in q_norm]
        if not spans:
            return GenerationResult("", None, [], [], 0.0, self.name, False, abstained=True)
        best = spans[0]
        if best.context_index < len(evidence):
            ev_index, local_start = evidence[best.context_index].index, best.char_start
        else:  # span found in the concatenated chain: map back to its passage
            ev_index, base = offsets[0]
            for idx, start in offsets:
                if best.char_start >= start:
                    ev_index, base = idx, start
            local_start = best.char_start - base
        ev = next(e for e in evidence if e.index == ev_index)
        sentence = next((ev.chunk.text[s:e] for s, e in split_sentences(ev.chunk.text) if s <= local_start < e),
                        None)
        abstained = best.probability < self.null_threshold
        return GenerationResult(
            answer=best.text, answer_sentence=sentence, answer_citation_indices=[ev_index], reasoning_steps=[],
            support_score=best.probability, backend=self.name, supported=not abstained, abstained=abstained,
            raw_output=json.dumps({"span": best.text, "probability": round(best.probability, 4),
                                   "span_score": round(best.score, 3), "null_score": round(best.null_score, 3)}),
        )


class Seq2SeqBackend(GenerationBackend):
    """Flan-T5 answering from compact evidence snippets."""

    name = "hf_generator"
    UNANSWERABLE = {"unanswerable", "unknown", "no answer", "not enough information", "none", "n/a"}

    def __init__(self, generator: Seq2SeqGenerator, embedder: "EmbeddingGenerator") -> None:
        self.generator = generator
        self.embedder = embedder

    def available(self) -> bool:
        return self.generator.available

    PRONOUN_START = re.compile(r"^(he|she|it|they|his|her|its|their|this|these|the company|the organisation|"
                               r"the organization|the model|the paper|the prize|the award)\b", re.IGNORECASE)

    @staticmethod
    def _is_heading(text: str) -> bool:
        return len(text.split()) < 8 and not text.rstrip().endswith((".", "!", "?", '"'))

    def _snippets(self, question: str, evidence: list[EvidencePassage], budget_words: int = 330) -> str:
        """Compact context for Flan-T5 (which reads at most 512 tokens).

        * If the evidence contains a multi-passage chain, only the chain
          passages are used - extra passages mostly add distractors.
        * Otherwise the three best-ranked passages are used.
        * From each passage the most relevant sentences are kept (similarity to
          the question, question-term overlap, bonus for the bridge entity);
          heading fragments are skipped, and a sentence that starts with a
          pronoun brings its preceding sentence along (coreference context).
        """
        chain_evs = sorted([ev for ev in evidence if ev.chain_step], key=lambda ev: ev.chain_step or 0)
        selected = chain_evs if len(chain_evs) >= 2 else evidence[:3]
        per_passage = 4 if len(chain_evs) >= 2 else 3
        sentences = [s for s in evidence_sentences(selected) if not self._is_heading(s.text)]
        if not sentences:
            return ""
        sims = self.embedder.encode([s.text for s in sentences]) @ self.embedder.encode_query(question)
        q_terms = set(content_terms(question))
        lines, used = [], 0
        for ev in selected:
            idxs = [i for i, s in enumerate(sentences) if s.evidence_index == ev.index]
            bridge = (ev.via_entity_name or "").lower()

            def score(i: int) -> float:
                text = sentences[i].text
                overlap = len(q_terms & set(content_terms(text))) / max(1, len(q_terms))
                bonus = 0.3 if bridge and bridge in text.lower() else 0.0
                return float(sims[i]) + 0.3 * overlap + bonus

            chosen = set(sorted(idxs, key=lambda i: -score(i))[:per_passage])
            for i in list(chosen):
                pos = idxs.index(i)
                if pos > 0 and self.PRONOUN_START.match(sentences[i].text):
                    chosen.add(idxs[pos - 1])
            texts = []
            for i in sorted(chosen):
                n = len(sentences[i].text.split())
                if used + n > budget_words:
                    break
                texts.append(sentences[i].text)
                used += n
            if texts:
                lines.append(f"[{ev.index}] {ev.chunk.document_name}: " + " ".join(texts))
        return "\n".join(lines)

    def generate(self, question: str, evidence: list[EvidencePassage], chain: Optional[EvidenceChain],
                 atype: AnswerType) -> GenerationResult:
        context = self._snippets(question, evidence)
        if atype.kind == "yesno":
            instruction = "Answer the question with yes or no based on the context."
        else:
            instruction = ('Answer the question based on the context. If it cannot be answered, say '
                           '"unanswerable".')
        text, prob = self.generator.generate(f"{instruction}\n\nContext:\n{context}\n\nQuestion: {question}")
        clean = text.strip().strip(".")
        hesitated = False
        if (clean.lower() in self.UNANSWERABLE or not clean) and atype.kind != "yesno":
            # Second pass with a plain reading-comprehension prompt. Its answer is
            # only accepted if it is grounded in the evidence, and with half the
            # support, because the cautious first pass did not commit to it.
            text, prob = self.generator.generate(f"{context}\n\nQ: {question}\nA:")
            clean = text.strip().strip(".")
            hesitated = True
            if clean.lower() in self.UNANSWERABLE or not clean or normalize_answer(clean) in normalize_answer(question):
                return GenerationResult("", None, [], [], prob, self.name, False, abstained=True, raw_output=text)
            prob *= 0.5
        if clean.lower() in self.UNANSWERABLE or not clean:
            return GenerationResult("", None, [], [], prob, self.name, False, abstained=True, raw_output=text)
        warnings = ["The cautious prompt abstained; a second reading-comprehension pass produced this answer, "
                    "which was accepted only after grounding."] if hesitated else []
        if atype.kind == "yesno":
            verdict = "Yes" if clean.lower().startswith("yes") else "No" if clean.lower().startswith("no") else None
            if verdict is None:
                return GenerationResult(clean, None, [], [], prob, self.name, False, raw_output=text,
                                        warnings=["Generator did not return yes/no."])
            return GenerationResult(verdict, None, [], [], prob, self.name, True, raw_output=text,
                                    warnings=["Yes/no answers are an inference over the cited evidence."])
        loc = locate_answer(clean, evidence)
        sentences = evidence_sentences(evidence)
        if loc:
            idx = loc[0]
            sent, _ = best_sentence_for(clean, sentences, {idx})
            return GenerationResult(clean, sent.text if sent else None, [idx], [], prob, self.name, True,
                                    raw_output=text, warnings=warnings)
        sent, overlap = best_sentence_for(clean, sentences)
        if sent is not None and token_f1(clean, sent.text) >= 0.6:
            return GenerationResult(clean, sent.text, [sent.evidence_index], [], prob * overlap, self.name, True,
                                    raw_output=text, warnings=warnings)
        return GenerationResult(clean, None, [], [], prob, self.name, False, raw_output=text,
                                warnings=[f"Generated answer '{clean}' is not supported by the evidence; rejected."])


LLM_SYSTEM_PROMPT = """You answer questions strictly from numbered evidence passages.
Rules:
- Use ONLY the evidence passages. Never use outside knowledge.
- Combine information from several passages when the question needs multiple hops.
- Every reasoning step is either a "fact" (a statement directly supported by the cited passages) or an "inference" (a conclusion you draw by combining facts). Cite passage numbers for every step.
- The answer must be a short phrase (or "yes"/"no" for yes/no questions).
- If the passages do not contain enough information, set insufficient_evidence to true and leave answer empty."""

LLM_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "insufficient_evidence": {"type": "boolean"},
        "answer_citations": {"type": "array", "items": {"type": "integer"}},
        "reasoning_steps": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "kind": {"type": "string", "enum": ["fact", "inference"]},
                    "text": {"type": "string"},
                    "citations": {"type": "array", "items": {"type": "integer"}},
                },
                "required": ["kind", "text", "citations"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["answer", "insufficient_evidence", "answer_citations", "reasoning_steps"],
    "additionalProperties": False,
}


def llm_user_prompt(question: str, evidence: list[EvidencePassage], chain: Optional[EvidenceChain]) -> str:
    lines = ["Evidence passages:"]
    for ev in evidence:
        lines.append(f"[{ev.index}] ({ev.title})\n{ev.chunk.text}")
    if chain and chain.length > 1:
        steps = " -> ".join(
            f"[{next((e.index for e in evidence if e.chunk.id == s.chunk_id), '?')}]"
            + (f" (via {s.via_entity_name})" if s.via_entity_name else "")
            for s in chain.steps
        )
        lines.append(f"\nCandidate evidence chain found by graph traversal: {steps}")
    lines.append(f"\nQuestion: {question}")
    lines.append("Return the JSON object described by the schema.")
    return "\n\n".join(lines)


def parse_llm_json(raw: str, name: str, evidence: list[EvidencePassage]) -> GenerationResult:
    data = json.loads(raw)
    valid = {ev.index for ev in evidence}
    steps = []
    for i, st in enumerate(data.get("reasoning_steps", []), start=1):
        cites = [c for c in st.get("citations", []) if c in valid]
        steps.append(ReasoningStep(i, "inference" if st.get("kind") == "inference" else "fact", st.get("text", ""),
                                   cites))
    answer = (data.get("answer") or "").strip()
    cites = [c for c in data.get("answer_citations", []) if c in valid]
    abstained = bool(data.get("insufficient_evidence")) or not answer
    return GenerationResult(answer, None, cites, steps, 0.0, name, not abstained, abstained=abstained, raw_output=raw)


class OllamaBackend(GenerationBackend):
    """Optional local LLM via Ollama's native REST API (no API key needed)."""

    name = "ollama"

    def __init__(self, url: str, model: str, timeout: float) -> None:
        self.url = url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def available(self) -> bool:
        return bool(self.model)

    def generate(self, question: str, evidence: list[EvidencePassage], chain: Optional[EvidenceChain],
                 atype: AnswerType) -> GenerationResult:
        import httpx

        payload = {
            "model": self.model,
            "stream": False,
            "format": LLM_SCHEMA,
            "options": {"temperature": 0},
            "messages": [
                {"role": "system", "content": LLM_SYSTEM_PROMPT},
                {"role": "user", "content": llm_user_prompt(question, evidence, chain)},
            ],
        }
        resp = httpx.post(f"{self.url}/api/chat", json=payload, timeout=self.timeout)
        resp.raise_for_status()
        raw = resp.json()["message"]["content"]
        return parse_llm_json(raw, self.name, evidence)


class AnthropicBackend(GenerationBackend):
    """Optional Claude API backend using the official ``anthropic`` SDK.

    Structured JSON output is enforced with ``output_config.format``; the
    server-side refusal fallback (beta) is enabled so a declined request is
    re-run on a fallback model. Any API error makes the pipeline fall back to
    the local backends.
    """

    name = "anthropic"

    def __init__(self, model: str, effort: str, timeout: float) -> None:
        self.model = model
        self.effort = effort
        self.timeout = timeout
        self._client: Any = None

    def available(self) -> bool:
        try:
            import anthropic  # noqa: F401
        except ImportError:
            return False
        import os

        return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")
                    or os.environ.get("ANTHROPIC_PROFILE"))

    def generate(self, question: str, evidence: list[EvidencePassage], chain: Optional[EvidenceChain],
                 atype: AnswerType) -> GenerationResult:
        import anthropic

        if self._client is None:
            self._client = anthropic.Anthropic(timeout=self.timeout)
        response = self._client.beta.messages.create(
            model=self.model,
            max_tokens=16000,
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            system=LLM_SYSTEM_PROMPT,
            output_config={"effort": self.effort, "format": {"type": "json_schema", "schema": LLM_SCHEMA}},
            messages=[{"role": "user", "content": llm_user_prompt(question, evidence, chain)}],
        )
        if response.stop_reason == "refusal":
            category = getattr(response.stop_details, "category", None) if response.stop_details else None
            raise RuntimeError(f"Claude declined the request (category: {category})")
        raw = next(b.text for b in response.content if b.type == "text")
        return parse_llm_json(raw, f"{self.name}:{response.model}", evidence)


# ============================================================ orchestrator
class AnswerGenerator:
    """Selects a backend, enforces grounding and builds fact/inference steps."""

    def __init__(self, settings: "Settings", embedder: "EmbeddingGenerator", extractor: "EntityExtractor") -> None:
        self.settings = settings
        self.embedder = embedder
        self.extractor = extractor
        self.reader = ExtractiveReader(settings.hf_reader_model, settings.generation_cache_dir, settings.device)
        self.seq2seq = Seq2SeqGenerator(settings.hf_generator_model, settings.generation_cache_dir, settings.device)
        self.backends: dict[str, GenerationBackend] = {
            "extractive": ExtractiveBackend(embedder, extractor),
            "hf_reader": ReaderBackend(self.reader, settings.reader_null_threshold),
            "hf_generator": Seq2SeqBackend(self.seq2seq, embedder),
            "ollama": OllamaBackend(settings.ollama_url, settings.ollama_model, settings.llm_timeout),
            "anthropic": AnthropicBackend(settings.anthropic_model, settings.anthropic_effort, settings.llm_timeout),
        }

    def backend_status(self) -> dict[str, Any]:
        return {
            "extractive": {"available": True, "description": "Model-free sentence scoring (always available)"},
            "hf_reader": {"available": self.reader.available, "model": self.reader.model_name,
                          "loaded": self.reader._model is not None, "error": self.reader.error},
            "hf_generator": {"available": self.seq2seq.available, "model": self.seq2seq.model_name,
                             "loaded": self.seq2seq._model is not None, "error": self.seq2seq.error},
            "ollama": {"available": self.backends["ollama"].available(), "model": self.settings.ollama_model or None,
                       "url": self.settings.ollama_url},
            "anthropic": {"available": self.backends["anthropic"].available(), "model": self.settings.anthropic_model},
        }

    def resolve_backend(self, requested: str, atype: AnswerType) -> list[str]:
        """Ordered list of backends to try (the extractive backend is always last)."""
        if requested and requested != "auto":
            order = [requested]
        else:
            # Flan-T5 composes facts from several passages (multi-hop); the
            # extractive reader is the second opinion for single-span answers.
            order = ["hf_generator", "hf_reader"]
        order = [b for b in order if b in self.backends and self.backends[b].available()]
        if "extractive" not in order:
            order.append("extractive")
        return order

    def generate(self, question: str, evidence: list[EvidencePassage], chain: Optional[EvidenceChain],
                 requested_backend: str = "auto") -> GenerationResult:
        atype = expected_answer_type(question)
        if not evidence:
            return GenerationResult("", None, [], [], 0.0, "none", False, abstained=True)
        warnings: list[str] = []
        result: Optional[GenerationResult] = None
        model_abstained = False
        for name in self.resolve_backend(requested_backend, atype):
            if name == "extractive" and model_abstained:
                # A model read the evidence and found no answer: that is the
                # verdict. The extractive scorer is only a fallback for missing
                # models, never a way to overrule an abstention.
                break
            try:
                candidate = self.backends[name].generate(question, evidence, chain, atype)
            except Exception as exc:
                log.warning("Backend %s failed: %s", name, exc)
                warnings.append(f"Backend '{name}' failed ({exc.__class__.__name__}: {str(exc)[:160]}); falling back.")
                continue
            warnings.extend(candidate.warnings)
            if candidate.abstained:
                # an abstaining model is a signal, not an error: remember it and try the next backend
                result = result or candidate
                if name != "extractive":
                    model_abstained = True
                    warnings.append(f"Backend '{name}' found no answer in the evidence.")
                continue
            if candidate.supported:
                grounded = self._ground(question, candidate, evidence)
                if grounded.supported:
                    result = grounded
                    break
                warnings.append(f"Backend '{name}' produced an answer that is not supported by the evidence; "
                                "it was rejected.")
            else:
                warnings.append(f"Backend '{name}' produced an answer that is not supported by the evidence; "
                                "it was rejected.")
        if result is None:
            result = GenerationResult("", None, [], [], 0.0, "none", False, abstained=True)
        result.model_abstained = model_abstained
        result.warnings = list(dict.fromkeys(warnings + [w for w in result.warnings if w not in warnings]))
        if not result.reasoning_steps:
            result.reasoning_steps = self.build_reasoning(question, result, evidence, chain)
        return result

    def _ground(self, question: str, result: GenerationResult, evidence: list[EvidencePassage]) -> GenerationResult:
        """Verify that the answer is supported by the cited evidence."""
        sentences = evidence_sentences(evidence)
        if result.answer.lower() in ("yes", "no"):
            if not result.answer_citation_indices:
                result.answer_citation_indices = [ev.index for ev in evidence[:2]]
            return result
        loc = locate_answer(result.answer, evidence)
        if loc is None:
            sent, _ = best_sentence_for(result.answer, sentences)
            if sent is None or token_f1(result.answer, sent.text) < 0.6:
                result.supported = False
                return result
            loc = (sent.evidence_index, sent.start, sent.end)
        cited = set(result.answer_citation_indices)
        if loc[0] not in cited:
            result.answer_citation_indices = [loc[0], *[c for c in result.answer_citation_indices if c != loc[0]]]
        if not result.answer_sentence:
            sent, _ = best_sentence_for(result.answer, sentences, {loc[0]})
            result.answer_sentence = sent.text if sent else None
        result.supported = True
        return result

    def build_reasoning(self, question: str, result: GenerationResult, evidence: list[EvidencePassage],
                        chain: Optional[EvidenceChain]) -> list[ReasoningStep]:
        """Deterministic fact/inference steps from the evidence chain and the answer sentence."""
        if result.abstained or not result.answer:
            return []
        sentences = evidence_sentences(evidence)
        by_chunk = {ev.chunk.id: ev for ev in evidence}
        steps: list[ReasoningStep] = []
        used: set[tuple[int, int]] = set()
        q_vec = self.embedder.encode_query(question)
        if chain is not None and chain.length > 1:
            for st in chain.steps:
                ev = by_chunk.get(st.chunk_id)
                if ev is None:
                    continue
                cands = [s for s in sentences if s.evidence_index == ev.index]
                if not cands:
                    continue
                # prefer the sentence that mentions the bridge entity / answer, else the most similar one
                def key(s: EvidenceSentence) -> float:
                    bonus = 0.0
                    if st.via_entity_name and st.via_entity_name.lower() in s.text.lower():
                        bonus += 0.3
                    if result.answer and normalize_answer(result.answer) in normalize_answer(s.text):
                        bonus += 0.5
                    return bonus
                sims = self.embedder.encode([s.text for s in cands]) @ q_vec
                best = max(range(len(cands)), key=lambda i: sims[i] + key(cands[i]))
                s = cands[best]
                used.add((s.evidence_index, s.start))
                steps.append(ReasoningStep(len(steps) + 1, "fact", s.text, [s.evidence_index]))
        answer_idx = result.answer_citation_indices[0] if result.answer_citation_indices else None
        if result.answer_sentence and answer_idx is not None:
            sent = next((s for s in sentences if s.evidence_index == answer_idx and s.text == result.answer_sentence.strip()),
                        None)
            if sent is None or (sent.evidence_index, sent.start) not in used:
                steps.append(ReasoningStep(len(steps) + 1, "fact", result.answer_sentence.strip(), [answer_idx]))
        fact_cites = sorted({c for s in steps for c in s.citation_indices})
        if len(fact_cites) >= 2:
            joined = " and ".join(f"[{c}]" for c in fact_cites)
            steps.append(ReasoningStep(len(steps) + 1, "inference",
                                       f"Combining {joined}, the answer is \"{result.answer}\".", fact_cites))
        elif fact_cites:
            steps.append(ReasoningStep(len(steps) + 1, "inference",
                                       f"The answer \"{result.answer}\" is stated in [{fact_cites[0]}].", fact_cites))
        return steps
