"""Named-entity and relation extraction with spaCy.

* Entities: spaCy NER (``en_core_web_sm`` by default), filtered and normalized.
* Relations: for two entities in the same sentence whose dependency paths meet
  at a verb, the verb lemma (plus preposition, e.g. "work at") becomes the
  relation label. Only relations that are explicitly expressed in a sentence
  are produced - nothing is inferred.
* Fallback: if the spaCy model is unavailable, a transparent regex extractor
  (capitalized phrases + years) is used and reported as degraded in /api/health.
"""
from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass
from typing import Any, Iterable, Optional

from app.core.domain import EntityMention, Relation, TextChunk
from app.core.logging_setup import get_logger
from app.entities.normalizer import KEEP_LABELS, clean_surface, is_valid_mention, normalize_entity

log = get_logger(__name__)

_CAPITALIZED_PHRASE = re.compile(
    r"\b(?:[A-Z][a-zA-Z'’\-]+)(?:\s+(?:of|the|and|de|da|van|von|for|in)\s+[A-Z][a-zA-Z'’\-]+|\s+[A-Z][a-zA-Z'’\-]+)*"
)
_YEAR = re.compile(r"\b(1[5-9][0-9]{2}|20[0-9]{2})\b")
_QUOTED = re.compile(r"[\"“]([^\"”]{3,80})[\"”]")
_SENTENCE_START_STOP = frozenset(
    "The A An In On At For From By With Which What Who Whom Whose When Where Why How Is Are Was Were Do Does Did "
    "It He She They This That These Those His Her Their Its After Before During Since Although However".split()
)


@dataclass
class QuestionEntity:
    text: str
    normalized: str
    label: str


class EntityExtractor:
    def __init__(self, model_name: str = "en_core_web_sm") -> None:
        self.model_name = model_name
        self._nlp: Any = None
        self._lock = threading.Lock()
        self.backend = "not_loaded"
        self.load_ms: Optional[float] = None
        self.error: Optional[str] = None

    # ----------------------------------------------------------------- loading
    def load(self) -> None:
        if self._nlp is not None or self.backend == "regex_fallback":
            return
        with self._lock:
            if self._nlp is not None:
                return
            start = time.perf_counter()
            try:
                import spacy

                self._nlp = spacy.load(self.model_name)
                self.backend = f"spacy:{self.model_name}"
            except Exception as exc:  # model missing or spaCy not installed
                self.error = str(exc)
                self.backend = "regex_fallback"
                log.warning("spaCy model '%s' unavailable (%s); using regex fallback NER", self.model_name, exc)
            self.load_ms = (time.perf_counter() - start) * 1000
            log.info("Entity extractor ready: %s (%.0f ms)", self.backend, self.load_ms)

    @property
    def uses_spacy(self) -> bool:
        self.load()
        return self._nlp is not None

    # -------------------------------------------------------------- extraction
    def extract_chunks(self, chunks: list[TextChunk]) -> tuple[list[EntityMention], list[Relation]]:
        """Extract entity mentions and verb relations from every chunk."""
        self.load()
        if not chunks:
            return [], []
        if self._nlp is None:
            mentions = [m for c in chunks for m in self._regex_mentions(c)]
            return mentions, []
        mentions: list[EntityMention] = []
        relations: list[Relation] = []
        with self._lock:
            docs = list(self._nlp.pipe((c.text for c in chunks), batch_size=16))
        for chunk, sdoc in zip(chunks, docs):
            m, r = self._process_doc(chunk, sdoc)
            mentions.extend(m)
            relations.extend(r)
        return mentions, relations

    def _process_doc(self, chunk: TextChunk, sdoc: Any) -> tuple[list[EntityMention], list[Relation]]:
        mentions: list[EntityMention] = []
        relations: list[Relation] = []
        for sent_index, sent in enumerate(sdoc.sents):
            sent_ents = []
            for ent in sent.ents:
                if ent.label_ not in KEEP_LABELS or not is_valid_mention(ent.text, ent.label_):
                    continue
                normalized = normalize_entity(ent.text)
                if not normalized:
                    continue
                mention = EntityMention(
                    text=clean_surface(ent.text),
                    label=ent.label_,
                    normalized=normalized,
                    chunk_id=chunk.id,
                    document_id=chunk.document_id,
                    char_start=ent.start_char,
                    char_end=ent.end_char,
                    sentence_index=sent_index,
                )
                mentions.append(mention)
                sent_ents.append((ent, mention))
            # verb-mediated relations between entity pairs of the sentence
            for i in range(len(sent_ents)):
                for j in range(i + 1, len(sent_ents)):
                    (e1, m1), (e2, m2) = sent_ents[i], sent_ents[j]
                    if m1.normalized == m2.normalized:
                        continue
                    label = self._relation_label(e1, e2)
                    if label:
                        relations.append(
                            Relation(
                                source_text=m1.text, target_text=m2.text,
                                source_normalized=m1.normalized, target_normalized=m2.normalized,
                                relation=label, chunk_id=chunk.id, document_id=chunk.document_id,
                                sentence=sent.text.strip(),
                            )
                        )
        return mentions, relations

    @staticmethod
    def _relation_label(e1: Any, e2: Any) -> Optional[str]:
        """Return the verb phrase connecting two entity spans, if any."""
        h1, h2 = e1.root, e2.root
        chain1 = [h1, *h1.ancestors]
        ancestors2 = {t.i: t for t in [h2, *h2.ancestors]}
        lca = next((t for t in chain1 if t.i in ancestors2), None)
        if lca is None or lca.pos_ not in ("VERB", "AUX"):
            return None
        # keep relations local: each entity at most 3 dependency steps from the verb
        depth1 = next(k for k, t in enumerate(chain1) if t.i == lca.i)
        chain2 = [h2, *h2.ancestors]
        depth2 = next(k for k, t in enumerate(chain2) if t.i == lca.i)
        if depth1 > 3 or depth2 > 3:
            return None
        passive = any(child.dep_ in ("auxpass", "nsubjpass") for child in lca.children)
        verb = lca.text.lower() if passive else lca.lemma_.lower()
        # attach the preposition that introduces the second entity ("work at")
        prep = ""
        for tok in chain2[1:depth2]:
            if tok.dep_ == "prep":
                prep = tok.text.lower()
                break
        if verb in {"be", "have", "do"} and not prep:
            return None
        return f"{verb} {prep}".strip()

    def _regex_mentions(self, chunk: TextChunk) -> list[EntityMention]:
        out = []
        for m in _CAPITALIZED_PHRASE.finditer(chunk.text):
            text = m.group(0).strip()
            first = text.split()[0]
            if first in _SENTENCE_START_STOP and len(text.split()) == 1:
                continue
            if not is_valid_mention(text, "MISC"):
                continue
            out.append(
                EntityMention(text=clean_surface(text), label="MISC", normalized=normalize_entity(text),
                              chunk_id=chunk.id, document_id=chunk.document_id, char_start=m.start(),
                              char_end=m.end(), sentence_index=0)
            )
        for m in _YEAR.finditer(chunk.text):
            out.append(
                EntityMention(text=m.group(0), label="DATE", normalized=m.group(0), chunk_id=chunk.id,
                              document_id=chunk.document_id, char_start=m.start(), char_end=m.end(),
                              sentence_index=0)
            )
        return out

    # ---------------------------------------------------------------- questions
    def question_entities(self, question: str) -> list[QuestionEntity]:
        """Entities and proper-noun phrases mentioned in a question."""
        self.load()
        found: dict[str, QuestionEntity] = {}

        def add(text: str, label: str) -> None:
            text = clean_surface(text)
            norm = normalize_entity(text)
            if norm and len(norm) > 1 and norm not in found:
                found[norm] = QuestionEntity(text=text, normalized=norm, label=label)

        for m in _QUOTED.finditer(question):
            add(m.group(1), "WORK_OF_ART")
        if self._nlp is not None:
            with self._lock:
                sdoc = self._nlp(question)
            for ent in sdoc.ents:
                if ent.label_ in KEEP_LABELS and is_valid_mention(ent.text, ent.label_):
                    add(ent.text, ent.label_)
            # proper-noun sequences missed by the NER model
            run: list[Any] = []
            for tok in list(sdoc) + [None]:
                if tok is not None and tok.pos_ == "PROPN":
                    run.append(tok)
                    continue
                if run:
                    span_text = sdoc[run[0].i : run[-1].i + 1].text
                    if is_valid_mention(span_text, "MISC"):
                        add(span_text, "PROPN")
                    run = []
            for m in _YEAR.finditer(question):
                add(m.group(0), "DATE")
        else:
            for m in _CAPITALIZED_PHRASE.finditer(question):
                text = m.group(0)
                if text.split()[0] in _SENTENCE_START_STOP and len(text.split()) == 1:
                    continue
                add(text, "MISC")
            for m in _YEAR.finditer(question):
                add(m.group(0), "DATE")
        return list(found.values())

    def noun_phrases(self, text: str) -> list[str]:
        """Noun chunks (used to build Wikipedia search queries)."""
        self.load()
        if self._nlp is None:
            return [m.group(0) for m in _CAPITALIZED_PHRASE.finditer(text)]
        with self._lock:
            sdoc = self._nlp(text)
        return [nc.text for nc in sdoc.noun_chunks]

    def entities_in_texts(self, texts: Iterable[str]) -> list[list[tuple[str, str]]]:
        """(text, label) entity lists for arbitrary texts (used by answer typing)."""
        self.load()
        texts = list(texts)
        if self._nlp is None:
            return [[(m.group(0), "MISC") for m in _CAPITALIZED_PHRASE.finditer(t)] for t in texts]
        with self._lock:
            docs = list(self._nlp.pipe(texts, batch_size=16))
        return [[(e.text, e.label_) for e in d.ents] for d in docs]
