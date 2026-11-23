"""Shared pytest fixtures.

Tests run against a temporary data directory (fresh SQLite DB, FAISS index and
graph files) but reuse the real downloaded models (./models) and the bundled
sample documents/datasets (./data/samples). Heavy objects are session-scoped.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parents[1]
ROOT = BACKEND.parent
sys.path.insert(0, str(BACKEND))

from app.core.config import Settings  # noqa: E402
from app.core.container import Container, set_container  # noqa: E402

SAMPLES = ROOT / "data" / "samples"
DOCS = SAMPLES / "documents"


@pytest.fixture(scope="session")
def settings(tmp_path_factory: pytest.TempPathFactory) -> Settings:
    tmp = tmp_path_factory.mktemp("grqa")
    s = Settings(data_dir=tmp / "data", results_dir=tmp / "results", samples_path=SAMPLES,
                 models_dir=ROOT / "models")
    s.ensure_dirs()
    return s


@pytest.fixture(scope="session")
def container(settings: Settings) -> Container:
    c = Container(settings)
    set_container(c)
    return c


@pytest.fixture(scope="session")
def demo_corpus(container: Container):
    from app.gnn.dataset import demo_corpus as build

    return build(container)


@pytest.fixture(scope="session")
def demo_questions() -> dict[str, dict]:
    import json

    data = json.loads((SAMPLES / "datasets" / "demo_qa.json").read_text(encoding="utf-8"))
    return {q["id"]: q for q in data["questions"]}
