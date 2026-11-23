"""Open-domain (Wikipedia) mode with the HTTP layer stubbed.

Only ``WikipediaClient._get`` is replaced: search, article parsing, caching,
two-hop bridge retrieval, graph construction, answering and section-level
citations all run for real. Live access is exercised manually (it needs a
policy-compliant User-Agent, see README section 16).
"""
from __future__ import annotations

import pytest

from app.retrieval.wikipedia import WikipediaClient, WikipediaUnavailableError

ARTICLES = {
    "Inception": (
        "Inception is a 2010 science fiction action film written and directed by Christopher Nolan. "
        "The film stars Leonardo DiCaprio as a professional thief who steals information by infiltrating "
        "the subconscious of his targets.\n\n== Production ==\nPrincipal photography began in Tokyo in June 2009. "
        "The film was produced by Emma Thomas and Christopher Nolan.\n\n== References ==\nNot used."
    ),
    "Christopher Nolan": (
        "Christopher Edward Nolan is a British and American filmmaker. He is known for complex narratives.\n\n"
        "== Early life ==\nNolan was born in Westminster, London. He studied English literature at University "
        "College London, where he also made short films with the college film society.\n\n== Career ==\n"
        "His breakthrough film was Memento, released in 2000."
    ),
}
SEARCH = {
    "inception": ["Inception"],
    "christopher nolan": ["Christopher Nolan"],
}


def fake_get(self, params):
    if params.get("list") == "search":
        q = params["srsearch"].lower()
        titles = next((v for k, v in SEARCH.items() if k in q), [])
        if "director" in q and "inception" in q:
            titles = ["Inception"]
        return {"query": {"search": [{"title": t} for t in titles]}}
    if params.get("prop") == "extracts|info":
        title = params["titles"]
        if title not in ARTICLES:
            return {"query": {"pages": [{"title": title, "missing": True}]}}
        return {"query": {"pages": [{"title": title, "extract": ARTICLES[title],
                                     "fullurl": "https://en.wikipedia.org/wiki/" + title.replace(" ", "_")}]}}
    return {}


def test_section_parsing_skips_reference_sections():
    sections = WikipediaClient.split_sections(ARTICLES["Inception"])
    assert [s["title"] for s in sections] == ["Introduction", "Production"]


def test_open_domain_two_hop_answer(container, monkeypatch):
    monkeypatch.setattr(WikipediaClient, "_get", fake_get)
    out = container.pipeline.ask("Which university did the director of the film Inception study at?",
                                 mode="open_domain", strategy="graph", persist=False)
    info = out["open_domain"]
    titles = [a["title"] for a in info["articles"]]
    assert titles[0] == "Inception" and "Christopher Nolan" in titles
    assert any(b["entity"] == "Christopher Nolan" for b in info["bridge_entities"]), "bridge hop expected"
    assert "University College London" in out["answer"], out
    cites = {c["document_name"]: c for c in out["citations"]}
    assert "Christopher Nolan" in cites
    assert cites["Christopher Nolan"]["section"].startswith("Early life")  # tiny sections are merged
    assert cites["Christopher Nolan"]["url"].startswith("https://en.wikipedia.org/wiki/")
    # articles are cached: a second run works without the network
    monkeypatch.setattr(WikipediaClient, "_get", lambda self, params: (_ for _ in ()).throw(
        WikipediaUnavailableError("offline")))
    again = container.pipeline.ask("Which university did the director of the film Inception study at?",
                                   mode="open_domain", strategy="graph", persist=False)
    assert again["answer"] == out["answer"]


def test_wikipedia_unreachable_is_reported(container, monkeypatch):
    def offline(self, params):
        raise WikipediaUnavailableError("Wikipedia could not be reached (no internet connection?).")

    monkeypatch.setattr(WikipediaClient, "_get", offline)
    with pytest.raises(WikipediaUnavailableError):
        container.pipeline.ask("Who painted a completely uncached painting?", mode="open_domain", persist=False)
