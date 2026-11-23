"""Open-domain retrieval from Wikipedia (MediaWiki Action API, no key needed).

Iterative two-hop retrieval for multi-hop questions:

1. hop 1  - search Wikipedia with the full question and with every question
            entity; fetch the top articles;
2. bridge - run hybrid retrieval over the fetched passages, extract named
            entities from the best passages that are *not* in the question
            (bridge entities, e.g. the person a question describes indirectly);
3. hop 2  - fetch the articles of the best bridge entities.

Articles are split into sections (title kept for citations), cached in SQLite
(so demos keep working offline after the first run) and chunked like
uploaded documents. All requests send a descriptive User-Agent as required by
the Wikimedia API etiquette; it can be customised with GRQA_WIKIPEDIA_USER_AGENT.
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, Optional

import httpx
import numpy as np

from app.core.domain import Document, Section, TextChunk
from app.core.logging_setup import get_logger
from app.core.timing import StageTimer
from app.entities.normalizer import normalize_entity, slugify
from app.ingestion.text_normalizer import normalize_text
from app.retrieval.corpus import Corpus, build_ephemeral_corpus
from app.retrieval.retrievers import HybridRetriever

if TYPE_CHECKING:  # pragma: no cover
    from app.core.container import Container

log = get_logger(__name__)

SKIP_SECTIONS = {
    "see also", "references", "external links", "further reading", "notes", "bibliography", "sources",
    "citations", "footnotes", "works cited", "notes and references", "explanatory notes", "selected publications",
    "gallery", "filmography", "discography",
}
_HEADING = re.compile(r"^(={2,6})\s*(.+?)\s*\1\s*$", re.MULTILINE)
BRIDGE_LABELS = {"PERSON", "ORG", "WORK_OF_ART", "GPE", "EVENT", "FAC", "LOC", "NORP", "PRODUCT", "LAW"}


class WikipediaUnavailableError(RuntimeError):
    """Raised when Wikipedia cannot be reached and nothing is cached."""


class WikipediaClient:
    def __init__(self, api_url: str, user_agent: str, timeout: float, repo: Any) -> None:
        self.api_url = api_url
        self.timeout = timeout
        self.repo = repo
        self.headers = {"User-Agent": user_agent, "Accept": "application/json"}

    def _get(self, params: dict[str, Any]) -> dict[str, Any]:
        params = {**params, "format": "json", "formatversion": 2}
        try:
            resp = httpx.get(self.api_url, params=params, headers=self.headers, timeout=self.timeout)
            resp.raise_for_status()
            return resp.json()
        except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout) as exc:
            raise WikipediaUnavailableError(
                "Wikipedia could not be reached (no internet connection?). Open-domain mode needs internet "
                "access unless the articles were cached by an earlier run."
            ) from exc
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 403:
                raise WikipediaUnavailableError(
                    "Wikipedia refused the request (HTTP 403). Wikimedia's User-Agent policy requires contact "
                    "information: set GRQA_WIKIPEDIA_USER_AGENT in .env, e.g. "
                    "'GraphRAG-QA/1.0 (https://your-site.example; you@example.com)', then restart the backend."
                ) from exc
            if exc.response.status_code == 429:
                raise WikipediaUnavailableError("Wikipedia rate limit reached (HTTP 429); retry in a minute.") from exc
            raise WikipediaUnavailableError(f"Wikipedia API error: {exc}") from exc
        except httpx.HTTPError as exc:
            raise WikipediaUnavailableError(f"Wikipedia API error: {exc}") from exc

    def is_reachable(self) -> bool:
        try:
            self._get({"action": "query", "meta": "siteinfo"})
            return True
        except WikipediaUnavailableError:
            return False

    def search(self, query: str, limit: int = 3) -> list[str]:
        key = f"search::{limit}::{query.strip().lower()}"
        cached = self.repo.get_wiki_article(key)
        if cached is not None:
            return cached["titles"]
        data = self._get({"action": "query", "list": "search", "srsearch": query, "srlimit": limit,
                          "srprop": "", "srwhat": "text"})
        titles = [hit["title"] for hit in data.get("query", {}).get("search", [])]
        self.repo.put_wiki_article(key, "", {"titles": titles})
        return titles

    def fetch_article(self, title: str) -> Optional[dict[str, Any]]:
        cached = self.repo.get_wiki_article(title)
        if cached is not None and "sections" in cached:
            return cached
        data = self._get({"action": "query", "prop": "extracts|info", "inprop": "url", "explaintext": 1,
                          "exsectionformat": "wiki", "redirects": 1, "titles": title})
        pages = data.get("query", {}).get("pages", [])
        if not pages or pages[0].get("missing") or not pages[0].get("extract"):
            return None
        page = pages[0]
        article = {"title": page["title"], "url": page.get("fullurl") or
                   "https://en.wikipedia.org/wiki/" + page["title"].replace(" ", "_"),
                   "sections": self.split_sections(page["extract"])}
        self.repo.put_wiki_article(page["title"], article["url"], article)
        if page["title"] != title:
            self.repo.put_wiki_article(title, article["url"], article)
        return article

    @staticmethod
    def split_sections(extract: str) -> list[dict[str, str]]:
        sections: list[dict[str, str]] = []
        positions = [(m.start(), m.end(), m.group(2).strip()) for m in _HEADING.finditer(extract)]
        bounds = [(0, "Introduction")] + [(end, title) for _, end, title in positions]
        starts = [s for s, _, _ in positions] + [len(extract)]
        for (begin, title), stop in zip(bounds, starts):
            text = extract[begin:stop].strip()
            if text and title.lower() not in SKIP_SECTIONS:
                sections.append({"title": title, "text": text})
        return sections


def article_to_document(article: dict[str, Any]) -> Document:
    """Turn a cached article into a Document with one Section per wiki section."""
    text, sections = "", []
    for sec in article["sections"]:
        body = normalize_text(sec["text"])
        if not body:
            continue
        if text:
            text += "\n\n"
        start = len(text)
        text += body
        sections.append(Section(sec["title"], start, len(text)))
    return Document(id="wiki_" + slugify(article["title"])[:60], name=article["title"], file_type="wikipedia",
                    source_type="wikipedia", url=article["url"], text=text, sections=sections)


class OpenDomainRetriever:
    """Builds a per-question corpus from Wikipedia articles (two-hop)."""

    def __init__(self, container: "Container") -> None:
        self.c = container
        s = container.settings
        self.client = WikipediaClient(s.wikipedia_api, s.wikipedia_user_agent, s.wikipedia_timeout, container.repo)

    def _chunks_for(self, article: dict[str, Any]) -> list[TextChunk]:
        doc = article_to_document(article)
        chunks = self.c.make_chunker().chunk_document(doc)
        return chunks[: self.c.settings.wikipedia_max_chunks_per_article]

    def build_corpus(self, question: str, timer: Optional[StageTimer] = None) -> tuple[Corpus, dict[str, Any]]:
        timer = timer or StageTimer()
        settings = self.c.settings
        info: dict[str, Any] = {"searches": [], "articles": [], "bridge_entities": []}
        with timer.stage("wikipedia_search"):
            q_entities = self.c.extractor.question_entities(question)
            queries = [question] + [e.text for e in q_entities if e.label != "DATE"]
            titles: list[str] = []
            for i, q in enumerate(dict.fromkeys(queries)):
                found = self.client.search(q, limit=3 if i == 0 else 2)
                info["searches"].append({"query": q, "titles": found})
                for t in found:
                    if t not in titles:
                        titles.append(t)
            titles = titles[: settings.wikipedia_max_articles]
        with timer.stage("wikipedia_fetch"):
            articles = [a for a in (self.client.fetch_article(t) for t in titles) if a]
        if not articles:
            raise WikipediaUnavailableError("No Wikipedia articles matched the question.")
        chunks = [c for a in articles for c in self._chunks_for(a)]
        n_first = len(chunks)
        fetched = {normalize_entity(a["title"]) for a in articles}

        # bridge hop: entities in the best first-hop passages that the question does not mention
        with timer.stage("wikipedia_bridge"):
            first = build_ephemeral_corpus("wikipedia-hop1", self.c.embedder, self.c.extractor,
                                           self.c.graph_builder, chunks, precomputed_mentions=([], []))
            top = HybridRetriever(first, self.c.embedder, self.c.runtime().hybrid_alpha).search(question, 6)
            q_lower = question.lower()
            q_norms = {e.normalized for e in q_entities}
            candidates: list[tuple[float, str]] = []
            ents = self.c.extractor.entities_in_texts([rp.chunk.text for rp in top])
            for rp, ents_in in zip(top, ents):
                for text, label in ents_in:
                    norm = normalize_entity(text)
                    if (label in BRIDGE_LABELS and norm and norm not in q_norms and norm not in fetched
                            and text.lower() not in q_lower and len(norm) > 2):
                        candidates.append((rp.hybrid_score, text))
            seen: set[str] = set()
            bridge_titles: list[str] = []
            for _, text in sorted(candidates, key=lambda x: -x[0]):
                norm = normalize_entity(text)
                if norm in seen:
                    continue
                seen.add(norm)
                hits = self.client.search(text, limit=1)
                info["bridge_entities"].append({"entity": text, "titles": hits})
                for t in hits:
                    if normalize_entity(t) not in fetched and t not in bridge_titles:
                        bridge_titles.append(t)
                if len(bridge_titles) >= settings.wikipedia_bridge_articles:
                    break
            for t in bridge_titles:
                art = self.client.fetch_article(t)
                if art and normalize_entity(art["title"]) not in fetched:
                    articles.append(art)
                    fetched.add(normalize_entity(art["title"]))
                    chunks.extend(self._chunks_for(art))
        info["articles"] = [{"title": a["title"], "url": a["url"], "hop": 1 if i < len(titles) else 2}
                            for i, a in enumerate(articles)]
        documents = {}
        for a in articles:
            doc_id = "wiki_" + slugify(a["title"])[:60]
            documents[doc_id] = {"name": a["title"], "file_type": "wikipedia", "source_type": "wikipedia",
                                 "url": a["url"], "num_chunks": sum(1 for c in chunks if c.document_id == doc_id)}
        # reuse the first-hop embeddings; embed only the bridge-article passages
        new_chunks = chunks[n_first:]
        embeddings = first.embeddings
        if new_chunks:
            with timer.stage("embedding"):
                embeddings = np.concatenate(
                    [first.embeddings, self.c.embedder.encode_passages([c.text for c in new_chunks])], axis=0
                )
        corpus = build_ephemeral_corpus("wikipedia", self.c.embedder, self.c.extractor, self.c.graph_builder,
                                        chunks, documents, timer=timer, precomputed_embeddings=embeddings)
        return corpus, info
