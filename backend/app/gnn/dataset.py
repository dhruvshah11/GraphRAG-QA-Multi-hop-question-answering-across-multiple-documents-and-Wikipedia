"""Training/evaluation data for the GNN reranker.

Two sources are supported and always labelled in checkpoint metadata:

``hotpotqa`` (recommended)
    Real multi-hop questions from HotpotQA (distractor setting, CC BY-SA 4.0).
    Each question comes with 10 Wikipedia paragraphs, 2 of which are the gold
    supporting paragraphs. ``scripts/download_datasets.py`` fetches a small
    seeded sample (train/val from the official train split, test from the
    official dev split) - the full 540 MB training file is never downloaded.

``demo`` (fallback, demonstration only)
    Rule-derived examples built from the bundled demo documents: the
    hand-written demo questions (gold passages = chunks containing the
    annotated evidence strings) plus template questions generated from
    extracted RELATED_TO relations. This set is tiny and only demonstrates
    that the training pipeline runs end to end; it says nothing about
    generalization.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional

from torch_geometric.data import Data

from app.core.domain import Document, TextChunk
from app.core.logging_setup import get_logger
from app.gnn.features import FeatureBuilder
from app.ingestion.document_processor import DocumentProcessor, build_document_text
from app.ingestion.extractors import extract_pages
from app.retrieval.corpus import Corpus, build_ephemeral_corpus

if TYPE_CHECKING:  # pragma: no cover
    from app.core.container import Container

log = get_logger(__name__)


# ================================================================== HotpotQA
def hotpotqa_dir(samples_dir: Path) -> Path:
    return samples_dir / "datasets" / "hotpotqa"


def load_hotpotqa(samples_dir: Path, split: str) -> list[dict[str, Any]]:
    path = hotpotqa_dir(samples_dir) / f"{split}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python scripts/download_datasets.py` to fetch the HotpotQA sample."
        )
    return json.loads(path.read_text(encoding="utf-8"))["examples"]


def hotpot_chunks(example: dict[str, Any]) -> tuple[list[TextChunk], set[str]]:
    """One passage per HotpotQA paragraph; returns chunks and gold chunk ids."""
    chunks, gold = [], set()
    supporting = set(example["supporting_titles"])
    for i, para in enumerate(example["paragraphs"]):
        title = para["title"]
        text = f"{title}: {para['text']}"
        cid = f"{example['id']}:p{i}"
        chunks.append(TextChunk(
            id=cid, document_id=f"hp:{example['id']}:{i}", document_name=title, chunk_index=0, text=text,
            page_number=None, page_end=None, char_start=0, char_end=len(text), token_count=len(text.split()),
            section=None, url="https://en.wikipedia.org/wiki/" + title.replace(" ", "_"), source_type="hotpotqa",
        ))
        if title in supporting:
            gold.add(cid)
    return chunks, gold


def hotpot_corpus(container: "Container", example: dict[str, Any]) -> tuple[Corpus, set[str]]:
    chunks, gold = hotpot_chunks(example)
    corpus = build_ephemeral_corpus(f"hotpotqa:{example['id']}", container.embedder, container.extractor,
                                    container.graph_builder, chunks)
    return corpus, gold


# ====================================================================== demo
def load_demo_dataset(samples_dir: Path) -> dict[str, Any]:
    return json.loads((samples_dir / "datasets" / "demo_qa.json").read_text(encoding="utf-8"))


def demo_corpus(container: "Container") -> Corpus:
    """Ephemeral corpus of the bundled sample documents (current chunk settings)."""
    settings = container.settings
    docs_dir = settings.samples_dir / "documents"
    chunker = container.make_chunker()
    chunks: list[TextChunk] = []
    documents: dict[str, dict[str, Any]] = {}
    for path in sorted(docs_dir.iterdir()):
        if path.suffix.lower() not in settings.allowed_extensions:
            continue
        doc_id = "demo_" + path.stem.lower().replace(" ", "_")[:40]
        raw_pages = extract_pages(path)
        text, pages, sections = build_document_text(raw_pages)
        doc = Document(id=doc_id, name=path.name, file_type=path.suffix.lstrip("."), source_type="upload",
                       path=str(path), text=text, pages=pages, sections=sections, num_pages=len(raw_pages))
        doc_chunks = chunker.chunk_document(doc)
        chunks.extend(doc_chunks)
        documents[doc_id] = {"name": path.name, "file_type": doc.file_type, "source_type": "upload", "url": None,
                             "num_chunks": len(doc_chunks), "num_pages": doc.num_pages}
    return build_ephemeral_corpus("demo", container.embedder, container.extractor, container.graph_builder, chunks,
                                  documents)


def gold_chunks_for_evidence(corpus: Corpus, evidence: list[str], gold_documents: Optional[list[str]] = None) -> set[str]:
    """Chunks containing an annotated evidence string (whitespace-insensitive)."""
    def norm(s: str) -> str:
        return " ".join(s.split()).lower()

    gold = set()
    for snippet in evidence:
        target = norm(snippet)
        for c in corpus.chunks:
            if gold_documents and c.document_name not in gold_documents:
                continue
            if target in norm(c.text):
                gold.add(c.id)
    return gold


# ============================================================ graph datasets
def build_graph_examples(
    container: "Container",
    items: list[tuple[str, str, Corpus, set[str]]],
    top_k: int = 4,
    max_hops: int = 2,
    progress: Optional[Callable[[int, int], None]] = None,
) -> list[Data]:
    """Convert (id, question, corpus, gold chunk ids) items into PyG graphs."""
    features = FeatureBuilder(container.embedder)
    out: list[Data] = []
    for i, (qid, question, corpus, gold) in enumerate(items):
        result = container.reasoner.analyze(corpus, question, top_k=top_k, max_hops=max_hops, use_gnn=False,
                                            alpha=container.runtime().hybrid_alpha)
        data, _ = features.build(result.qgraph, corpus, result.scores.query_vec, gold)
        data.qid = qid
        out.append(data)
        if progress:
            progress(i + 1, len(items))
    return out


def hotpotqa_graphs(container: "Container", split: str, limit: Optional[int] = None, top_k: int = 4,
                    max_hops: int = 2, progress: Optional[Callable[[int, int], None]] = None) -> list[Data]:
    examples = load_hotpotqa(container.settings.samples_dir, split)[: limit or None]

    def items():
        for ex in examples:
            corpus, gold = hotpot_corpus(container, ex)
            yield ex["id"], ex["question"], corpus, gold

    features = FeatureBuilder(container.embedder)
    out: list[Data] = []
    for i, (qid, question, corpus, gold) in enumerate(items()):
        result = container.reasoner.analyze(corpus, question, top_k=top_k, max_hops=max_hops, use_gnn=False,
                                            alpha=container.runtime().hybrid_alpha)
        data, _ = features.build(result.qgraph, corpus, result.scores.query_vec, gold)
        data.qid = qid
        out.append(data)
        if progress:
            progress(i + 1, len(examples))
    return out


def demo_graphs(container: "Container", top_k: int = 4, max_hops: int = 3) -> tuple[list[Data], dict[str, Any]]:
    """Rule-derived demonstration set (hand-written demo questions + relation templates)."""
    corpus = demo_corpus(container)
    dataset = load_demo_dataset(container.settings.samples_dir)
    items: list[tuple[str, str, Corpus, set[str]]] = []
    for q in dataset["questions"]:
        gold = gold_chunks_for_evidence(corpus, q.get("gold_evidence", []), q.get("gold_documents"))
        if gold:
            items.append((q["id"], q["question"], corpus, gold))
    n_handwritten = len(items)
    # template questions from explicitly extracted relations ("What did <source> <relation>?")
    seen = set()
    for r in corpus.relations:
        if not r.source_id or not r.target_id:
            continue
        key = (r.source_id, r.relation)
        if key in seen:
            continue
        seen.add(key)
        source = corpus.registry.entities[r.source_id].name
        items.append((f"rel-{len(seen):03d}", f"What did {source} {r.relation}?", corpus, {r.chunk_id}))
    graphs = build_graph_examples(container, items, top_k=top_k, max_hops=max_hops)
    info = {"handwritten_questions": n_handwritten, "relation_template_questions": len(items) - n_handwritten,
            "documents": len(corpus.documents), "chunks": len(corpus.chunks)}
    return graphs, info
