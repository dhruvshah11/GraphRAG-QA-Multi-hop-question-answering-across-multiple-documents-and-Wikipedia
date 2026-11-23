"""Download the NLP models used by GraphRAG-QA into ./models (run once, needs internet).

    python scripts/download_models.py                    # embeddings + spaCy + extractive reader
    python scripts/download_models.py --with-generator   # also Flan-T5 (about 1 GB)

Afterwards the application works fully offline (except Wikipedia mode).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.core.config import get_settings  # noqa: E402


def step(title: str) -> float:
    print(f"\n==> {title}", flush=True)
    return time.perf_counter()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--with-generator", action="store_true", help="also download google/flan-t5-base")
    parser.add_argument("--skip-reader", action="store_true", help="skip the extractive QA reader")
    args = parser.parse_args()
    settings = get_settings()
    failures = 0

    t = step(f"Embedding model: {settings.embedding_model}")
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(settings.embedding_model, cache_folder=str(settings.embedding_cache_dir))
        vec = model.encode(["GraphRAG-QA model check"], normalize_embeddings=True)
        print(f"    ok - dimension {vec.shape[1]} ({time.perf_counter() - t:.1f}s)")
    except Exception as exc:
        failures += 1
        print(f"    FAILED: {exc}")

    t = step(f"spaCy model: {settings.spacy_model}")
    try:
        import spacy

        if not spacy.util.is_package(settings.spacy_model):
            from spacy.cli import download

            download(settings.spacy_model)
        nlp = spacy.load(settings.spacy_model)
        print(f"    ok - pipeline {nlp.pipe_names} ({time.perf_counter() - t:.1f}s)")
    except Exception as exc:
        failures += 1
        print(f"    FAILED: {exc}")

    hf_models = []
    if not args.skip_reader:
        hf_models.append(("reader", settings.hf_reader_model, "AutoModelForQuestionAnswering"))
    if args.with_generator:
        hf_models.append(("generator", settings.hf_generator_model, "AutoModelForSeq2SeqLM"))
    for label, name, cls_name in hf_models:
        t = step(f"Answer {label}: {name}")
        try:
            import transformers

            cls = getattr(transformers, cls_name)
            transformers.AutoTokenizer.from_pretrained(name, cache_dir=str(settings.generation_cache_dir))
            cls.from_pretrained(name, cache_dir=str(settings.generation_cache_dir))
            print(f"    ok ({time.perf_counter() - t:.1f}s)")
        except Exception as exc:
            failures += 1
            print(f"    FAILED: {exc}")

    print("\nModels directory:", settings.models_dir)
    print("Done." if failures == 0 else f"Finished with {failures} failure(s).")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
