"""Load the bundled demo documents into the application (same code path as uploads).

    python scripts/seed_demo.py            # ingest all sample documents
    python scripts/seed_demo.py --reset    # delete existing uploads first
    python scripts/seed_demo.py --ask      # also answer the demo questions once
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.core.container import get_container  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reset", action="store_true", help="remove previously uploaded documents first")
    parser.add_argument("--ask", action="store_true", help="answer the demo questions after ingestion")
    args = parser.parse_args()

    c = get_container()
    docs_dir = c.settings.samples_dir / "documents"
    if not any(docs_dir.glob("*")):
        print("No sample documents found - run `python scripts/make_sample_documents.py` first.")
        return 1
    if args.reset:
        for d in c.repo.list_documents(source_type="upload"):
            c.ingestion.delete_document(d["id"])
            print("deleted", d["name"])

    for path in sorted(docs_dir.iterdir()):
        if path.suffix.lower() not in c.settings.allowed_extensions:
            continue
        doc, duplicate = c.processor.register_upload(path.name, path.read_bytes())
        if duplicate and c.repo.get_document(doc.id)["status"] == "processed":
            print(f"= {path.name:32s} already processed ({doc.id})")
            continue
        t = time.perf_counter()
        summary = c.ingestion.process_document(doc.id)
        print(f"+ {path.name:32s} {summary['pages']} page(s), {summary['chunks']} chunk(s), "
              f"{summary['distinct_entities']} entities  ({(time.perf_counter() - t) * 1000:.0f} ms)")

    corpus = c.corpus_manager.corpus
    print("\nCorpus:", json.dumps(corpus.stats()["graph"], indent=None))
    if args.ask:
        data = json.loads((c.settings.samples_dir / "datasets" / "demo_qa.json").read_text(encoding="utf-8"))
        for q in data["questions"]:
            out = c.pipeline.ask(q["question"], strategy="gnn")
            print(f"\nQ: {q['question']}\n   -> {out['answer']}  (expected: {q['answer'] or 'insufficient evidence'}; "
                  f"strategy {out['effective_strategy']}, confidence {out['confidence']:.2f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
