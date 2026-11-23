"""Data-access layer: every SQL statement of the application lives here."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Iterable, Optional

from app.core.domain import Document, EntityMention, Entity, PageText, Relation, Section, TextChunk
from app.database.db import Database


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _row_to_chunk(row: Any, doc_name: str = "", url: Optional[str] = None, source_type: str = "upload") -> TextChunk:
    keys = row.keys()
    return TextChunk(
        id=row["id"],
        document_id=row["document_id"],
        document_name=row["document_name"] if "document_name" in keys else doc_name,
        chunk_index=row["chunk_index"],
        text=row["text"],
        page_number=row["page_number"],
        page_end=row["page_end"],
        char_start=row["char_start"],
        char_end=row["char_end"],
        token_count=row["token_count"],
        section=row["section"],
        url=row["url"] if "url" in keys else url,
        source_type=row["source_type"] if "source_type" in keys else source_type,
    )


class Repository:
    def __init__(self, db: Database) -> None:
        self.db = db

    # ================================================================ documents
    def insert_document(self, doc: Document) -> None:
        with self.db.transaction() as conn:
            conn.execute(
                """INSERT INTO documents (id, name, file_type, source_type, path, url, size_bytes,
                       num_pages, num_chunks, num_entities, status, error, checksum, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    doc.id, doc.name, doc.file_type, doc.source_type, doc.path, doc.url, doc.size_bytes,
                    doc.num_pages, doc.num_chunks, doc.num_entities, doc.status, doc.error, doc.checksum,
                    doc.created_at or utc_now(),
                ),
            )

    def update_document(self, doc_id: str, **fields: Any) -> None:
        if not fields:
            return
        cols = ", ".join(f"{k} = ?" for k in fields)
        with self.db.transaction() as conn:
            conn.execute(f"UPDATE documents SET {cols} WHERE id = ?", (*fields.values(), doc_id))

    def get_document(self, doc_id: str) -> Optional[dict[str, Any]]:
        with self.db.connect() as conn:
            row = conn.execute("SELECT * FROM documents WHERE id = ?", (doc_id,)).fetchone()
            return dict(row) if row else None

    def find_document_by_checksum(self, checksum: str) -> Optional[dict[str, Any]]:
        with self.db.connect() as conn:
            row = conn.execute("SELECT * FROM documents WHERE checksum = ?", (checksum,)).fetchone()
            return dict(row) if row else None

    def list_documents(self, source_type: Optional[str] = None) -> list[dict[str, Any]]:
        with self.db.connect() as conn:
            if source_type:
                rows = conn.execute(
                    "SELECT * FROM documents WHERE source_type = ? ORDER BY created_at DESC", (source_type,)
                ).fetchall()
            else:
                rows = conn.execute("SELECT * FROM documents ORDER BY created_at DESC").fetchall()
            return [dict(r) for r in rows]

    def delete_document(self, doc_id: str) -> None:
        with self.db.transaction() as conn:
            conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))

    def save_document_content(self, doc: Document) -> None:
        """Replace stored pages and sections of a document."""
        with self.db.transaction() as conn:
            conn.execute("DELETE FROM pages WHERE document_id = ?", (doc.id,))
            conn.execute("DELETE FROM sections WHERE document_id = ?", (doc.id,))
            conn.executemany(
                "INSERT INTO pages (document_id, page_number, text, char_offset) VALUES (?, ?, ?, ?)",
                [(doc.id, p.page_number, p.text, p.char_offset) for p in doc.pages],
            )
            conn.executemany(
                "INSERT INTO sections (document_id, idx, title, char_start, char_end) VALUES (?, ?, ?, ?, ?)",
                [(doc.id, i, s.title, s.char_start, s.char_end) for i, s in enumerate(doc.sections)],
            )

    def get_pages(self, doc_id: str) -> list[PageText]:
        with self.db.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM pages WHERE document_id = ? ORDER BY page_number", (doc_id,)
            ).fetchall()
            return [PageText(r["page_number"], r["text"], r["char_offset"]) for r in rows]

    def get_sections(self, doc_id: str) -> list[Section]:
        with self.db.connect() as conn:
            rows = conn.execute("SELECT * FROM sections WHERE document_id = ? ORDER BY idx", (doc_id,)).fetchall()
            return [Section(r["title"], r["char_start"], r["char_end"]) for r in rows]

    # =================================================================== chunks
    def replace_chunks(self, doc_id: str, chunks: list[TextChunk], vector_ids: list[int]) -> None:
        with self.db.transaction() as conn:
            conn.execute("DELETE FROM chunks WHERE document_id = ?", (doc_id,))
            conn.executemany(
                """INSERT INTO chunks (id, document_id, chunk_index, text, page_number, page_end,
                       char_start, char_end, token_count, section, vector_id)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    (c.id, c.document_id, c.chunk_index, c.text, c.page_number, c.page_end,
                     c.char_start, c.char_end, c.token_count, c.section, vid)
                    for c, vid in zip(chunks, vector_ids)
                ],
            )

    _CHUNK_SELECT = """SELECT c.*, d.name AS document_name, d.url AS url, d.source_type AS source_type
                       FROM chunks c JOIN documents d ON d.id = c.document_id"""

    def get_chunks(self, doc_id: Optional[str] = None, only_processed: bool = False) -> list[TextChunk]:
        sql = self._CHUNK_SELECT
        params: tuple[Any, ...] = ()
        clauses = []
        if doc_id:
            clauses.append("c.document_id = ?")
            params += (doc_id,)
        if only_processed:
            clauses.append("d.status = 'processed'")
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY c.document_id, c.chunk_index"
        with self.db.connect() as conn:
            return [_row_to_chunk(r) for r in conn.execute(sql, params).fetchall()]

    def get_chunks_with_vector_ids(self, only_processed: bool = True) -> list[tuple[TextChunk, int]]:
        sql = self._CHUNK_SELECT + " WHERE c.vector_id IS NOT NULL"
        if only_processed:
            sql += " AND d.status = 'processed'"
        sql += " ORDER BY c.vector_id"
        with self.db.connect() as conn:
            return [(_row_to_chunk(r), r["vector_id"]) for r in conn.execute(sql).fetchall()]

    def get_chunk(self, chunk_id: str) -> Optional[TextChunk]:
        with self.db.connect() as conn:
            row = conn.execute(self._CHUNK_SELECT + " WHERE c.id = ?", (chunk_id,)).fetchone()
            return _row_to_chunk(row) if row else None

    def get_vector_ids(self, doc_id: str) -> list[int]:
        with self.db.connect() as conn:
            rows = conn.execute(
                "SELECT vector_id FROM chunks WHERE document_id = ? AND vector_id IS NOT NULL", (doc_id,)
            ).fetchall()
            return [r["vector_id"] for r in rows]

    def next_vector_id(self) -> int:
        with self.db.connect() as conn:
            row = conn.execute("SELECT COALESCE(MAX(vector_id), -1) + 1 AS nxt FROM chunks").fetchone()
            return int(row["nxt"])

    # ========================================================= entities/relations
    def replace_mentions(self, doc_id: str, mentions: list[EntityMention], relations: list[Relation]) -> None:
        with self.db.transaction() as conn:
            conn.execute("DELETE FROM entity_mentions WHERE document_id = ?", (doc_id,))
            conn.execute("DELETE FROM relations WHERE document_id = ?", (doc_id,))
            conn.executemany(
                """INSERT INTO entity_mentions (chunk_id, document_id, text, label, normalized,
                       char_start, char_end, sentence_index) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    (m.chunk_id, m.document_id, m.text, m.label, m.normalized, m.char_start, m.char_end,
                     m.sentence_index)
                    for m in mentions
                ],
            )
            conn.executemany(
                """INSERT INTO relations (chunk_id, document_id, source_text, target_text, source_normalized,
                       target_normalized, relation, sentence) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    (r.chunk_id, r.document_id, r.source_text, r.target_text, r.source_normalized,
                     r.target_normalized, r.relation, r.sentence)
                    for r in relations
                ],
            )

    def get_mentions(self, doc_id: Optional[str] = None, only_processed: bool = True) -> list[EntityMention]:
        sql = "SELECT m.* FROM entity_mentions m JOIN documents d ON d.id = m.document_id"
        clauses, params = [], []
        if doc_id:
            clauses.append("m.document_id = ?")
            params.append(doc_id)
        if only_processed:
            clauses.append("d.status = 'processed'")
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY m.id"
        with self.db.connect() as conn:
            return [
                EntityMention(
                    text=r["text"], label=r["label"], normalized=r["normalized"], chunk_id=r["chunk_id"],
                    document_id=r["document_id"], char_start=r["char_start"], char_end=r["char_end"],
                    sentence_index=r["sentence_index"],
                )
                for r in conn.execute(sql, params).fetchall()
            ]

    def get_relations(self, doc_id: Optional[str] = None, only_processed: bool = True) -> list[Relation]:
        sql = "SELECT r.* FROM relations r JOIN documents d ON d.id = r.document_id"
        clauses, params = [], []
        if doc_id:
            clauses.append("r.document_id = ?")
            params.append(doc_id)
        if only_processed:
            clauses.append("d.status = 'processed'")
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        with self.db.connect() as conn:
            return [
                Relation(
                    source_text=r["source_text"], target_text=r["target_text"],
                    source_normalized=r["source_normalized"], target_normalized=r["target_normalized"],
                    relation=r["relation"], chunk_id=r["chunk_id"], document_id=r["document_id"],
                    sentence=r["sentence"],
                )
                for r in conn.execute(sql, params).fetchall()
            ]

    def replace_entities(self, entities: Iterable[Entity]) -> None:
        with self.db.transaction() as conn:
            conn.execute("DELETE FROM entities")
            conn.executemany(
                """INSERT INTO entities (id, name, label, category, normalized, aliases, mention_count,
                       num_chunks, num_documents) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    (e.id, e.name, e.label, e.category, e.normalized, json.dumps(sorted(e.aliases)),
                     e.mention_count, len(e.chunk_ids), len(e.document_ids))
                    for e in entities
                ],
            )

    def list_entities(self, search: Optional[str] = None, limit: int = 200) -> list[dict[str, Any]]:
        sql = "SELECT * FROM entities"
        params: list[Any] = []
        if search:
            sql += " WHERE name LIKE ? OR aliases LIKE ?"
            params += [f"%{search}%", f"%{search}%"]
        sql += " ORDER BY mention_count DESC, name LIMIT ?"
        params.append(limit)
        with self.db.connect() as conn:
            out = []
            for r in conn.execute(sql, params).fetchall():
                d = dict(r)
                d["aliases"] = json.loads(d["aliases"])
                out.append(d)
            return out

    def document_entity_summary(self, doc_id: str) -> list[dict[str, Any]]:
        with self.db.connect() as conn:
            rows = conn.execute(
                """SELECT normalized, MAX(label) AS label, MIN(text) AS text, COUNT(*) AS mentions,
                          COUNT(DISTINCT chunk_id) AS chunks
                   FROM entity_mentions WHERE document_id = ?
                   GROUP BY normalized ORDER BY mentions DESC, text""",
                (doc_id,),
            ).fetchall()
            return [dict(r) for r in rows]

    # =================================================================== counts
    def counts(self) -> dict[str, Any]:
        with self.db.connect() as conn:
            def one(sql: str) -> Any:
                return conn.execute(sql).fetchone()[0]

            return {
                "documents": one("SELECT COUNT(*) FROM documents WHERE source_type = 'upload'"),
                "processed_documents": one(
                    "SELECT COUNT(*) FROM documents WHERE source_type = 'upload' AND status = 'processed'"
                ),
                "wikipedia_articles_cached": one("SELECT COUNT(*) FROM wiki_cache"),
                "pages": one(
                    "SELECT COALESCE(SUM(num_pages), 0) FROM documents WHERE source_type = 'upload' "
                    "AND status = 'processed'"
                ),
                "chunks": one(
                    "SELECT COUNT(*) FROM chunks c JOIN documents d ON d.id = c.document_id "
                    "WHERE d.status = 'processed'"
                ),
                "avg_chunk_tokens": one(
                    "SELECT COALESCE(AVG(token_count), 0) FROM chunks c JOIN documents d ON d.id = c.document_id "
                    "WHERE d.status = 'processed'"
                ),
                "avg_chunk_chars": one(
                    "SELECT COALESCE(AVG(LENGTH(text)), 0) FROM chunks c JOIN documents d ON d.id = c.document_id "
                    "WHERE d.status = 'processed'"
                ),
                "entity_mentions": one("SELECT COUNT(*) FROM entity_mentions"),
                "entities": one("SELECT COUNT(*) FROM entities"),
                "relations": one("SELECT COUNT(*) FROM relations"),
                "questions_answered": one("SELECT COUNT(*) FROM queries WHERE status = 'completed'"),
            }

    # ========================================================== queries/answers
    def save_query_result(
        self,
        query_id: str,
        question: str,
        mode: str,
        strategy: str,
        top_k: int,
        max_hops: int,
        latency_ms: float,
        payload: dict[str, Any],
    ) -> None:
        answer_id = f"ans-{query_id}"
        with self.db.transaction() as conn:
            conn.execute(
                """INSERT INTO queries (id, question, mode, strategy, top_k, max_hops, created_at, latency_ms, status)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'completed')""",
                (query_id, question, mode, strategy, top_k, max_hops, utc_now(), latency_ms),
            )
            conn.execute(
                """INSERT INTO answers (id, query_id, answer, reasoning_summary, confidence,
                       insufficient_evidence, generator_backend, payload) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    answer_id, query_id, payload.get("answer", ""), payload.get("reasoning_summary", ""),
                    payload.get("confidence", 0.0), int(bool(payload.get("insufficient_evidence"))),
                    payload.get("generator_backend"), json.dumps(payload),
                ),
            )
            conn.executemany(
                """INSERT INTO citations (answer_id, citation_index, chunk_id, document_id, document_name,
                       page_number, passage_text, supporting_sentence, retrieval_score, graph_score, gnn_score,
                       support_score, claim) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    (
                        answer_id, c["index"], c["chunk_id"], c.get("document_id"), c.get("document_name"),
                        c.get("page_number"), c.get("passage_text"), c.get("supporting_sentence"),
                        c.get("retrieval_score"), c.get("graph_score"), c.get("gnn_score"),
                        c.get("support_score"), c.get("claim"),
                    )
                    for c in payload.get("citations", [])
                ],
            )

    def save_failed_query(self, query_id: str, question: str, mode: str, strategy: str, error: str) -> None:
        with self.db.transaction() as conn:
            conn.execute(
                """INSERT INTO queries (id, question, mode, strategy, created_at, status, error)
                   VALUES (?, ?, ?, ?, ?, 'failed', ?)""",
                (query_id, question, mode, strategy, utc_now(), error),
            )

    def list_queries(self, limit: int = 20) -> list[dict[str, Any]]:
        with self.db.connect() as conn:
            rows = conn.execute(
                """SELECT q.*, a.answer, a.confidence, a.insufficient_evidence, a.generator_backend
                   FROM queries q LEFT JOIN answers a ON a.query_id = q.id
                   ORDER BY q.created_at DESC LIMIT ?""",
                (limit,),
            ).fetchall()
            out = []
            for r in rows:
                d = dict(r)
                if d.get("insufficient_evidence") is not None:
                    d["insufficient_evidence"] = bool(d["insufficient_evidence"])
                out.append(d)
            return out

    def get_query_payload(self, query_id: str) -> Optional[dict[str, Any]]:
        with self.db.connect() as conn:
            row = conn.execute("SELECT payload FROM answers WHERE query_id = ?", (query_id,)).fetchone()
            return json.loads(row["payload"]) if row else None

    def latest_query_id(self) -> Optional[str]:
        with self.db.connect() as conn:
            row = conn.execute(
                "SELECT id FROM queries WHERE status = 'completed' ORDER BY created_at DESC LIMIT 1"
            ).fetchone()
            return row["id"] if row else None

    # =============================================================== evaluation
    def create_eval_run(self, run_id: str, dataset: str, config: dict[str, Any]) -> None:
        with self.db.transaction() as conn:
            conn.execute(
                "INSERT INTO evaluation_runs (id, dataset, status, created_at, config) VALUES (?, ?, 'running', ?, ?)",
                (run_id, dataset, utc_now(), json.dumps(config)),
            )

    def update_eval_run(self, run_id: str, **fields: Any) -> None:
        if "summary" in fields and not isinstance(fields["summary"], str):
            fields["summary"] = json.dumps(fields["summary"])
        cols = ", ".join(f"{k} = ?" for k in fields)
        with self.db.transaction() as conn:
            conn.execute(f"UPDATE evaluation_runs SET {cols} WHERE id = ?", (*fields.values(), run_id))

    def add_eval_results(self, run_id: str, rows: list[dict[str, Any]]) -> None:
        cols = [
            "question_id", "question_type", "question", "strategy", "expected_answer", "predicted_answer",
            "exact_match", "f1",
            "precision_at_k", "recall_at_k", "mrr", "evidence_recall", "citation_coverage", "citation_precision",
            "answer_latency_ms", "graph_latency_ms", "insufficient", "details",
        ]
        with self.db.transaction() as conn:
            conn.executemany(
                f"INSERT INTO evaluation_results (run_id, {', '.join(cols)}) VALUES (?, {', '.join('?' * len(cols))})",
                [
                    (run_id, *[json.dumps(r[c]) if c == "details" else r.get(c) for c in cols])
                    for r in rows
                ],
            )

    def get_eval_run(self, run_id: str) -> Optional[dict[str, Any]]:
        with self.db.connect() as conn:
            row = conn.execute("SELECT * FROM evaluation_runs WHERE id = ?", (run_id,)).fetchone()
            if not row:
                return None
            d = dict(row)
            d["config"] = json.loads(d["config"]) if d["config"] else {}
            d["summary"] = json.loads(d["summary"]) if d["summary"] else None
            return d

    def list_eval_runs(self, limit: int = 20) -> list[dict[str, Any]]:
        with self.db.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM evaluation_runs ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
            out = []
            for r in rows:
                d = dict(r)
                d["config"] = json.loads(d["config"]) if d["config"] else {}
                d["summary"] = json.loads(d["summary"]) if d["summary"] else None
                out.append(d)
            return out

    def get_eval_results(self, run_id: str) -> list[dict[str, Any]]:
        with self.db.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM evaluation_results WHERE run_id = ? ORDER BY id", (run_id,)
            ).fetchall()
            out = []
            for r in rows:
                d = dict(r)
                d["details"] = json.loads(d["details"]) if d["details"] else {}
                out.append(d)
            return out

    # ================================================================== metrics
    def log_metrics(self, kind: str, ref_id: Optional[str], durations: dict[str, float],
                    meta: Optional[dict[str, Any]] = None) -> None:
        ts = utc_now()
        with self.db.transaction() as conn:
            conn.executemany(
                "INSERT INTO metrics_log (created_at, kind, ref_id, stage, duration_ms, meta) VALUES (?, ?, ?, ?, ?, ?)",
                [(ts, kind, ref_id, stage, float(ms), json.dumps(meta or {})) for stage, ms in durations.items()],
            )

    def metrics_rows(self, kind: Optional[str] = None, limit: int = 5000) -> list[dict[str, Any]]:
        sql = "SELECT * FROM metrics_log"
        params: list[Any] = []
        if kind:
            sql += " WHERE kind = ?"
            params.append(kind)
        sql += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        with self.db.connect() as conn:
            return [dict(r) for r in conn.execute(sql, params).fetchall()]

    # ================================================================= settings
    def get_setting_overrides(self) -> dict[str, Any]:
        with self.db.connect() as conn:
            return {r["key"]: json.loads(r["value"]) for r in conn.execute("SELECT * FROM settings").fetchall()}

    def set_settings(self, values: dict[str, Any]) -> None:
        with self.db.transaction() as conn:
            conn.executemany(
                "INSERT INTO settings (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                [(k, json.dumps(v)) for k, v in values.items()],
            )

    # =============================================================== wiki cache
    def get_wiki_article(self, title: str) -> Optional[dict[str, Any]]:
        with self.db.connect() as conn:
            row = conn.execute("SELECT payload FROM wiki_cache WHERE title = ?", (title,)).fetchone()
            return json.loads(row["payload"]) if row else None

    def put_wiki_article(self, title: str, url: str, payload: dict[str, Any]) -> None:
        with self.db.transaction() as conn:
            conn.execute(
                """INSERT INTO wiki_cache (title, url, fetched_at, payload) VALUES (?, ?, ?, ?)
                   ON CONFLICT(title) DO UPDATE SET url = excluded.url, fetched_at = excluded.fetched_at,
                   payload = excluded.payload""",
                (title, url, utc_now(), json.dumps(payload)),
            )

    # ===================================================================== jobs
    def create_job(self, job_id: str, kind: str, message: str = "") -> None:
        ts = utc_now()
        with self.db.transaction() as conn:
            conn.execute(
                "INSERT INTO jobs (id, kind, status, created_at, updated_at, progress, message) VALUES (?, ?, 'queued', ?, ?, 0, ?)",
                (job_id, kind, ts, ts, message),
            )

    def update_job(self, job_id: str, **fields: Any) -> None:
        if "result" in fields and not isinstance(fields["result"], str):
            fields["result"] = json.dumps(fields["result"])
        fields["updated_at"] = utc_now()
        cols = ", ".join(f"{k} = ?" for k in fields)
        with self.db.transaction() as conn:
            conn.execute(f"UPDATE jobs SET {cols} WHERE id = ?", (*fields.values(), job_id))

    def get_job(self, job_id: str) -> Optional[dict[str, Any]]:
        with self.db.connect() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
            if not row:
                return None
            d = dict(row)
            d["result"] = json.loads(d["result"]) if d["result"] else None
            return d

    def latest_job(self, kind: str) -> Optional[dict[str, Any]]:
        with self.db.connect() as conn:
            row = conn.execute(
                "SELECT * FROM jobs WHERE kind = ? ORDER BY created_at DESC LIMIT 1", (kind,)
            ).fetchone()
            if not row:
                return None
            d = dict(row)
            d["result"] = json.loads(d["result"]) if d["result"] else None
            return d
