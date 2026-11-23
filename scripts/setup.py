"""One-shot project setup check (run after installing requirements and models).

    python scripts/setup.py            # check + create folders/DB + seed demo documents
    python scripts/setup.py --no-seed  # only checks

Creates data folders and the SQLite database, generates the sample documents if
missing, verifies every Python dependency and model, and loads the demo
documents through the normal ingestion pipeline. Prints what is missing and how
to fix it; never downloads anything by itself.
"""
from __future__ import annotations

import argparse
import importlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

PACKAGES = ["fastapi", "uvicorn", "pydantic", "numpy", "torch", "torch_geometric", "faiss", "rank_bm25", "networkx",
            "spacy", "sentence_transformers", "transformers", "pymupdf", "docx", "httpx", "psutil", "pytest"]


def check(label: str, ok: bool, hint: str = "") -> bool:
    print(f"  [{'OK' if ok else '!!'}] {label}" + ("" if ok or not hint else f"  ->  {hint}"))
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-seed", action="store_true")
    args = parser.parse_args()
    ok = True
    print("Python", sys.version.split()[0])
    ok &= check("Python >= 3.11", sys.version_info >= (3, 11), "install Python 3.11+")
    print("\nPackages")
    for name in PACKAGES:
        try:
            importlib.import_module(name)
            check(name, True)
        except Exception:
            ok &= check(name, False, "pip install -r backend/requirements.txt")

    from app.core.config import get_settings
    from app.generation.hf_models import hf_model_cached

    s = get_settings()
    s.ensure_dirs()
    print("\nModels")
    ok &= check(f"embedding model {s.embedding_model}", hf_model_cached(s.embedding_model, s.embedding_cache_dir),
                "python scripts/download_models.py")
    try:
        import spacy

        ok &= check(f"spaCy {s.spacy_model}", spacy.util.is_package(s.spacy_model), "python scripts/download_models.py")
    except Exception:
        ok = False
    check(f"reader {s.hf_reader_model} (optional)", hf_model_cached(s.hf_reader_model, s.generation_cache_dir),
          "python scripts/download_models.py")
    check(f"generator {s.hf_generator_model} (optional, recommended)",
          hf_model_cached(s.hf_generator_model, s.generation_cache_dir), "python scripts/download_models.py --with-generator")
    check("trained GNN checkpoint (optional)", s.gnn_checkpoint_path.exists(),
          "python scripts/download_datasets.py && python scripts/train_gnn.py")

    docs = s.samples_dir / "documents"
    if not any(docs.glob("*.*")):
        print("\nGenerating sample documents ...")
        subprocess.run([sys.executable, str(ROOT / "scripts" / "make_sample_documents.py")], check=True)

    from app.core.container import get_container

    c = get_container()
    print("\nDatabase:", c.settings.db_path)
    if not args.no_seed and ok:
        print("\nSeeding demo documents ...")
        subprocess.run([sys.executable, str(ROOT / "scripts" / "seed_demo.py")], check=False)
    print("\nSetup", "complete." if ok else "incomplete - fix the items marked [!!] above.")
    print("Start the backend:  cd backend && python -m uvicorn app.main:app --port 8000")
    print("Start the frontend: cd frontend && npm run dev")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
