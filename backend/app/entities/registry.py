"""Entity registry: canonical entities, duplicate resolution and lookup.

Duplicate resolution is deliberately conservative ("merge obvious duplicates
carefully"). A mention is merged into another entity only when:

1. the normalized strings are identical ("The University of Toronto" ==
   "university of toronto");
2. dropping middle initials makes them identical ("Geoffrey E. Hinton" ->
   "Geoffrey Hinton");
3. a single-token PERSON mention equals the surname of exactly one
   multi-token PERSON in the same document (else: exactly one in the corpus),
   e.g. "Hinton" -> "Geoffrey Hinton";
4. an all-capital acronym matches the initials of exactly one multi-word
   entity in the same document (else: in the corpus), e.g. "MIT" ->
   "Massachusetts Institute of Technology".

Every original surface form is kept as an alias and every mention keeps its
source chunk, so the resolution is fully traceable.
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Iterable, Optional

from app.core.domain import Entity, EntityMention
from app.entities.normalizer import category_for, entity_id_for, normalize_entity

_ACRONYM = re.compile(r"^[A-Z]{2,6}$")
_INITIAL_STOP = {"of", "the", "and", "for", "in", "de", "la", "du", "von", "van", "at", "on"}
_LABEL_PRIORITY_IGNORE = {"PROPN", "MISC"}


def _initials(normalized: str) -> str:
    return "".join(tok[0] for tok in normalized.split() if tok not in _INITIAL_STOP and tok)


def _drop_initials(normalized: str) -> str:
    return " ".join(tok for tok in normalized.split() if len(tok) > 1)


class EntityRegistry:
    def __init__(self) -> None:
        self.entities: dict[str, Entity] = {}
        self.alias_to_id: dict[str, str] = {}
        self.total_chunks: int = 0

    # ------------------------------------------------------------------ build
    @classmethod
    def build(cls, mentions: list[EntityMention], total_chunks: Optional[int] = None) -> "EntityRegistry":
        registry = cls()
        registry._resolve(mentions)
        registry.total_chunks = total_chunks if total_chunks is not None else len({m.chunk_id for m in mentions})
        return registry

    def _resolve(self, mentions: list[EntityMention]) -> None:
        by_norm: dict[str, list[EntityMention]] = defaultdict(list)
        for m in mentions:
            by_norm[m.normalized].append(m)
        label_of = {
            norm: Counter(m.label for m in ms).most_common(1)[0][0] for norm, ms in by_norm.items()
        }

        # rule 2: drop middle initials when the shorter form exists
        base_of: dict[str, str] = {}
        for norm in by_norm:
            stripped = _drop_initials(norm)
            base_of[norm] = stripped if stripped != norm and stripped in by_norm else norm

        # indexes for rules 3 and 4 (per document and corpus-wide)
        persons_multi = [n for n in by_norm if label_of[n] == "PERSON" and len(base_of[n].split()) > 1]
        multiword = [n for n in by_norm if len(n.split()) > 1 and label_of[n] not in ("DATE", "PERSON")]
        docs_of: dict[str, set[str]] = {n: {m.document_id for m in ms} for n, ms in by_norm.items()}
        surname_index: dict[str, set[str]] = defaultdict(set)
        for n in persons_multi:
            surname_index[base_of[n].split()[-1]].add(base_of[n])
        acronym_index: dict[str, set[str]] = defaultdict(set)
        for n in multiword:
            if len(n.split()) >= 2:
                acronym_index[_initials(n)].add(base_of[n])

        def pick(candidates: set[str], doc_id: str) -> Optional[str]:
            if not candidates:
                return None
            same_doc = {c for c in candidates if doc_id in docs_of.get(c, set())}
            if len(same_doc) == 1:
                return next(iter(same_doc))
            if not same_doc and len(candidates) == 1:
                return next(iter(candidates))
            return None

        canonical_for_mention: list[str] = []
        for m in mentions:
            norm = base_of[m.normalized]
            target = None
            if label_of[m.normalized] == "PERSON" and len(norm.split()) == 1:
                target = pick(surname_index.get(norm, set()), m.document_id)
            elif _ACRONYM.match(m.text.replace(".", "")) and len(norm.split()) == 1:
                target = pick(acronym_index.get(norm.replace(" ", ""), set()), m.document_id)
            canonical_for_mention.append(target or norm)

        # build canonical entities
        grouped: dict[str, list[EntityMention]] = defaultdict(list)
        for m, canon in zip(mentions, canonical_for_mention):
            grouped[canon].append(m)
        for canon, ms in grouped.items():
            labels = Counter(m.label for m in ms if m.label not in _LABEL_PRIORITY_IGNORE) or Counter(
                m.label for m in ms
            )
            label = labels.most_common(1)[0][0]
            surfaces = Counter(m.text for m in ms if normalize_entity(m.text) == canon) or Counter(m.text for m in ms)
            top = max(surfaces.items(), key=lambda kv: (kv[1], len(kv[0])))[0]
            entity = Entity(
                id=entity_id_for(canon),
                name=top,
                label=label,
                category=category_for(label),
                normalized=canon,
                aliases={m.text for m in ms},
                mention_count=len(ms),
                chunk_ids={m.chunk_id for m in ms},
                document_ids={m.document_id for m in ms},
            )
            if entity.id in self.entities:  # slug collision: merge
                existing = self.entities[entity.id]
                existing.aliases |= entity.aliases
                existing.mention_count += entity.mention_count
                existing.chunk_ids |= entity.chunk_ids
                existing.document_ids |= entity.document_ids
            else:
                self.entities[entity.id] = entity
            for m in ms:
                m.entity_id = entity.id
                self.alias_to_id.setdefault(m.normalized, entity.id)
            self.alias_to_id[canon] = entity.id

    # ----------------------------------------------------------------- queries
    def get(self, entity_id: str) -> Optional[Entity]:
        return self.entities.get(entity_id)

    def document_frequency(self, entity_id: str) -> int:
        e = self.entities.get(entity_id)
        return len(e.chunk_ids) if e else 0

    def specificity(self, entity_id: str) -> float:
        """IDF-style specificity in (0, 1]: 1 = mentioned in a single passage."""
        df = max(1, self.document_frequency(entity_id))
        n = max(1, self.total_chunks)
        if n <= 1:
            return 1.0
        return math.log(1 + n / df) / math.log(1 + n)

    def match(self, text: str, limit: int = 3) -> list[tuple[str, float]]:
        """Match a question mention against registry entities (exact > alias > partial)."""
        norm = normalize_entity(text)
        if not norm:
            return []
        if norm in self.alias_to_id:
            return [(self.alias_to_id[norm], 1.0)]
        stripped = _drop_initials(norm)
        if stripped in self.alias_to_id:
            return [(self.alias_to_id[stripped], 0.95)]
        q_tokens = set(norm.split())
        scored: list[tuple[str, float]] = []
        for e in self.entities.values():
            e_tokens = set(e.normalized.split())
            if not e_tokens:
                continue
            if len(q_tokens) == 1 and e.label == "PERSON" and norm == e.normalized.split()[-1]:
                scored.append((e.id, 0.85))  # surname
                continue
            inter = q_tokens & e_tokens
            if not inter:
                continue
            if f" {norm} " in f" {e.normalized} " or f" {e.normalized} " in f" {norm} ":
                ratio = min(len(norm), len(e.normalized)) / max(len(norm), len(e.normalized))
                scored.append((e.id, 0.6 + 0.3 * ratio))
            else:
                jaccard = len(inter) / len(q_tokens | e_tokens)
                if jaccard >= 0.5:
                    scored.append((e.id, 0.7 * jaccard))
        scored.sort(key=lambda x: (-x[1], x[0]))
        return scored[:limit]

    def to_list(self) -> list[Entity]:
        return sorted(self.entities.values(), key=lambda e: (-e.mention_count, e.name))

    def entities_in_chunk(self, chunk_id: str) -> list[Entity]:
        return [e for e in self.entities.values() if chunk_id in e.chunk_ids]

    @staticmethod
    def stats(entities: Iterable[Entity]) -> dict[str, int]:
        counter = Counter(e.category for e in entities)
        return dict(counter.most_common())
