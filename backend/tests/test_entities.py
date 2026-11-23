"""Entity extraction, normalization and duplicate resolution."""
from __future__ import annotations

from app.core.domain import EntityMention, TextChunk
from app.entities.normalizer import is_valid_mention, normalize_entity
from app.entities.registry import EntityRegistry


def chunk(cid: str, doc: str, text: str) -> TextChunk:
    return TextChunk(cid, doc, doc + ".txt", 0, text, 1, 1, 0, len(text), len(text.split()))


def test_spacy_extracts_entities_and_relations(container):
    c = chunk("d1:c0", "d1", "Geoffrey Hinton won the 2024 Nobel Prize in Physics. Hinton works at the University of Toronto.")
    mentions, relations = container.extractor.extract_chunks([c])
    assert container.extractor.backend.startswith("spacy")
    found = {(m.normalized, m.label) for m in mentions}
    assert ("geoffrey hinton", "PERSON") in found
    assert any(m.normalized == "university of toronto" for m in mentions)
    assert all(c.text[m.char_start:m.char_end] for m in mentions)  # offsets valid
    # relations are only produced from verbs present in the sentence
    for r in relations:
        assert r.relation.split()[0] in r.sentence.lower() or r.relation.split()[0] in ("win", "work", "bear")


def test_normalization():
    assert normalize_entity("The University of Toronto's") == "university of toronto"
    assert normalize_entity("U.S.") == "us"
    assert normalize_entity("AT&T") == "at and t"
    assert not is_valid_mention("last week", "DATE")
    assert is_valid_mention("8 October 2024", "DATE")


def mention(text: str, label: str, cid: str, doc: str) -> EntityMention:
    return EntityMention(text, label, normalize_entity(text), cid, doc, 0, len(text), 0)


def test_registry_merges_obvious_duplicates_only():
    mentions = [
        mention("Geoffrey E. Hinton", "PERSON", "a:c0", "a"),
        mention("Geoffrey Hinton", "PERSON", "b:c0", "b"),
        mention("Hinton", "PERSON", "b:c1", "b"),
        mention("Massachusetts Institute of Technology", "ORG", "c:c0", "c"),
        mention("MIT", "ORG", "c:c1", "c"),
        mention("John Smith", "PERSON", "d:c0", "d"),
        mention("Jane Smith", "PERSON", "d:c1", "d"),
        mention("Smith", "PERSON", "d:c2", "d"),  # ambiguous surname -> must NOT be merged
    ]
    reg = EntityRegistry.build(mentions)
    hinton = reg.entities["ent:geoffrey_hinton"]
    assert hinton.chunk_ids == {"a:c0", "b:c0", "b:c1"}
    assert {"Geoffrey E. Hinton", "Hinton"} <= hinton.aliases
    mit = reg.get(reg.alias_to_id["mit"])
    assert mit.normalized == "massachusetts institute of technology"
    assert reg.alias_to_id["smith"] == "ent:smith"  # stays separate
    assert reg.match("Hinton")[0][0] == "ent:geoffrey_hinton"


def test_specificity_is_idf_like():
    mentions = [mention("Common Org", "ORG", f"x:c{i}", "x") for i in range(5)] + [mention("Rare Person", "PERSON", "x:c0", "x")]
    reg = EntityRegistry.build(mentions, total_chunks=5)
    assert reg.specificity("ent:rare_person") > reg.specificity("ent:common_org")
