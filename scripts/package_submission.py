"""Build a clean submission zip of the project.

    python scripts/package_submission.py                      # -> ../GraphRAG-QA_Submission.zip
    python scripts/package_submission.py --out D:/submit.zip

Left out (all re-created by the install steps in README §15):
  * .venv, frontend/node_modules, frontend/dist, caches (__pycache__, .pytest_cache, .vite)
  * downloaded models: models/embeddings/*, models/generation/*  (scripts/download_models.py)
  * runtime data: data/*.sqlite3*, data/uploads, data/processed, data/indexes, data/graphs
    (scripts/setup.py re-creates the database and seeds the demo documents)
  * tools/ (PlantUML jar, fetched by scripts/render_diagrams.py), .env files, Office lock files

Kept: all source code, tests, scripts, docs (report, slides, diagrams, screenshots),
results/, data/samples/ (documents + datasets) and the trained GNN checkpoint models/gnn/.
"""
from __future__ import annotations

import argparse
import fnmatch
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOP = "graphrag-qa"

EXCLUDED_DIR_NAMES = {".git", ".venv", "node_modules", "__pycache__", ".pytest_cache", ".vite", ".mypy_cache", ".ruff_cache"}
EXCLUDED_PREFIXES = ("frontend/dist/", "tools/", "models/embeddings/", "models/generation/",
                     "data/uploads/", "data/processed/", "data/indexes/", "data/graphs/")
EXCLUDED_FILE_GLOBS = ("*.pyc", ".env", "*.sqlite3", "*.sqlite3-*", "~$*", "*.tsbuildinfo", "*.log")


def included(rel: str) -> bool:
    parts = rel.split("/")
    if any(p in EXCLUDED_DIR_NAMES for p in parts[:-1]):
        return False
    name = parts[-1]
    if name == ".gitkeep":  # keep the empty runtime folders
        return True
    if any(rel.startswith(p) for p in EXCLUDED_PREFIXES):
        return False
    if any(fnmatch.fnmatch(name, g) for g in EXCLUDED_FILE_GLOBS):
        return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=ROOT.parent / "GraphRAG-QA_Submission.zip")
    args = parser.parse_args()
    out = args.out.resolve()
    files = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(ROOT).as_posix()
        if any(p in EXCLUDED_DIR_NAMES for p in rel.split("/")[:-1]):
            continue
        if included(rel) and path.resolve() != out:
            files.append((path, rel))
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path, rel in files:
            zf.write(path, f"{TOP}/{rel}")
    raw = sum(p.stat().st_size for p, _ in files)
    print(f"{len(files)} files, {raw / 1e6:.1f} MB uncompressed -> {out} ({out.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
