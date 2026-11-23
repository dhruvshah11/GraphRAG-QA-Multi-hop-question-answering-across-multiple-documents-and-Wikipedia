"""SQLite connection management and schema.

The standard-library ``sqlite3`` module is used directly (no ORM) so that the
storage layer stays small and easy to explain. Each operation opens a short
lived connection; WAL mode allows concurrent readers while a writer is active.
"""
from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id            TEXT PRIMARY KEY,
    name          TEXT NOT NULL,
    file_type     TEXT NOT NULL,
    source_type   TEXT NOT NULL DEFAULT 'upload',
    path          TEXT,
    url           TEXT,
    size_bytes    INTEGER NOT NULL DEFAULT 0,
    num_pages     INTEGER NOT NULL DEFAULT 0,
    num_chunks    INTEGER NOT NULL DEFAULT 0,
    num_entities  INTEGER NOT NULL DEFAULT 0,
    status        TEXT NOT NULL DEFAULT 'uploaded',
    error         TEXT,
    checksum      TEXT,
    created_at    TEXT NOT NULL,
    processed_at  TEXT,
    processing_ms REAL
);

CREATE TABLE IF NOT EXISTS pages (
    document_id  TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    page_number  INTEGER NOT NULL,
    text         TEXT NOT NULL,
    char_offset  INTEGER NOT NULL,
    PRIMARY KEY (document_id, page_number)
);

CREATE TABLE IF NOT EXISTS sections (
    document_id  TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    idx          INTEGER NOT NULL,
    title        TEXT,
    char_start   INTEGER NOT NULL,
    char_end     INTEGER NOT NULL,
    PRIMARY KEY (document_id, idx)
);

CREATE TABLE IF NOT EXISTS chunks (
    id           TEXT PRIMARY KEY,
    document_id  TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index  INTEGER NOT NULL,
    text         TEXT NOT NULL,
    page_number  INTEGER,
    page_end     INTEGER,
    char_start   INTEGER NOT NULL,
    char_end     INTEGER NOT NULL,
    token_count  INTEGER NOT NULL,
    section      TEXT,
    vector_id    INTEGER UNIQUE
);
CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id);

CREATE TABLE IF NOT EXISTS entity_mentions (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    chunk_id       TEXT NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    document_id    TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    text           TEXT NOT NULL,
    label          TEXT NOT NULL,
    normalized     TEXT NOT NULL,
    char_start     INTEGER NOT NULL,
    char_end       INTEGER NOT NULL,
    sentence_index INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_mentions_chunk ON entity_mentions(chunk_id);
CREATE INDEX IF NOT EXISTS idx_mentions_norm ON entity_mentions(normalized);

CREATE TABLE IF NOT EXISTS relations (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    chunk_id           TEXT NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    document_id        TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    source_text        TEXT NOT NULL,
    target_text        TEXT NOT NULL,
    source_normalized  TEXT NOT NULL,
    target_normalized  TEXT NOT NULL,
    relation           TEXT NOT NULL,
    sentence           TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_relations_chunk ON relations(chunk_id);

-- Canonical entity registry (rebuilt from entity_mentions whenever the graph is rebuilt)
CREATE TABLE IF NOT EXISTS entities (
    id             TEXT PRIMARY KEY,
    name           TEXT NOT NULL,
    label          TEXT NOT NULL,
    category       TEXT NOT NULL,
    normalized     TEXT NOT NULL,
    aliases        TEXT NOT NULL,
    mention_count  INTEGER NOT NULL,
    num_chunks     INTEGER NOT NULL,
    num_documents  INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS queries (
    id          TEXT PRIMARY KEY,
    question    TEXT NOT NULL,
    mode        TEXT NOT NULL,
    strategy    TEXT NOT NULL,
    top_k       INTEGER,
    max_hops    INTEGER,
    created_at  TEXT NOT NULL,
    latency_ms  REAL,
    status      TEXT NOT NULL DEFAULT 'completed',
    error       TEXT
);

CREATE TABLE IF NOT EXISTS answers (
    id                     TEXT PRIMARY KEY,
    query_id               TEXT NOT NULL REFERENCES queries(id) ON DELETE CASCADE,
    answer                 TEXT NOT NULL,
    reasoning_summary      TEXT,
    confidence             REAL,
    insufficient_evidence  INTEGER NOT NULL DEFAULT 0,
    generator_backend      TEXT,
    payload                TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_answers_query ON answers(query_id);

CREATE TABLE IF NOT EXISTS citations (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    answer_id            TEXT NOT NULL REFERENCES answers(id) ON DELETE CASCADE,
    citation_index       INTEGER NOT NULL,
    chunk_id             TEXT NOT NULL,
    document_id          TEXT,
    document_name        TEXT,
    page_number          INTEGER,
    passage_text         TEXT,
    supporting_sentence  TEXT,
    retrieval_score      REAL,
    graph_score          REAL,
    gnn_score            REAL,
    support_score        REAL,
    claim                TEXT
);

CREATE TABLE IF NOT EXISTS evaluation_runs (
    id           TEXT PRIMARY KEY,
    dataset      TEXT NOT NULL,
    status       TEXT NOT NULL,
    created_at   TEXT NOT NULL,
    finished_at  TEXT,
    config       TEXT NOT NULL,
    summary      TEXT,
    error        TEXT,
    progress     REAL NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS evaluation_results (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id              TEXT NOT NULL REFERENCES evaluation_runs(id) ON DELETE CASCADE,
    question_id         TEXT NOT NULL,
    question_type       TEXT,
    question            TEXT NOT NULL,
    strategy            TEXT NOT NULL,
    expected_answer     TEXT,
    predicted_answer    TEXT,
    exact_match         REAL,
    f1                  REAL,
    precision_at_k      REAL,
    recall_at_k         REAL,
    mrr                 REAL,
    evidence_recall     REAL,
    citation_coverage   REAL,
    citation_precision  REAL,
    answer_latency_ms   REAL,
    graph_latency_ms    REAL,
    insufficient        INTEGER,
    details             TEXT
);
CREATE INDEX IF NOT EXISTS idx_eval_results_run ON evaluation_results(run_id);

CREATE TABLE IF NOT EXISTS metrics_log (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at   TEXT NOT NULL,
    kind         TEXT NOT NULL,
    ref_id       TEXT,
    stage        TEXT NOT NULL,
    duration_ms  REAL NOT NULL,
    meta         TEXT
);
CREATE INDEX IF NOT EXISTS idx_metrics_kind ON metrics_log(kind, stage);

CREATE TABLE IF NOT EXISTS settings (
    key    TEXT PRIMARY KEY,
    value  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS wiki_cache (
    title       TEXT PRIMARY KEY,
    url         TEXT,
    fetched_at  TEXT NOT NULL,
    payload     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS jobs (
    id          TEXT PRIMARY KEY,
    kind        TEXT NOT NULL,
    status      TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL,
    progress    REAL NOT NULL DEFAULT 0,
    message     TEXT,
    result      TEXT
);
"""


class Database:
    """Thin wrapper around a SQLite file."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._write_lock = threading.RLock()
        with self.connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript(SCHEMA)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(str(self.path), timeout=30, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Serialized write transaction (SQLite allows one writer at a time)."""
        with self._write_lock:
            with self.connect() as conn:
                yield conn
