"""Background jobs (document processing, evaluation runs, GNN training).

Two single-worker thread pools keep index mutations serialized (ingestion)
and avoid running two CPU-heavy tasks at once (evaluation/training). Job state
is stored in SQLite so the UI can poll it.
"""
from __future__ import annotations

import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

from app.core.logging_setup import get_logger

log = get_logger(__name__)

ProgressFn = Callable[[float, str], None]


class JobManager:
    def __init__(self, repo: Any) -> None:
        self.repo = repo
        self.pools = {
            "ingest": ThreadPoolExecutor(max_workers=1, thread_name_prefix="ingest"),
            "task": ThreadPoolExecutor(max_workers=1, thread_name_prefix="task"),
        }

    def submit(self, kind: str, fn: Callable[..., Any], *args: Any, pool: str = "task", job_id: str | None = None,
               message: str = "", **kwargs: Any) -> str:
        job_id = job_id or f"{kind}-{uuid.uuid4().hex[:10]}"
        self.repo.create_job(job_id, kind, message)

        def progress(fraction: float, msg: str = "") -> None:
            self.repo.update_job(job_id, progress=round(max(0.0, min(1.0, fraction)), 4), message=msg)

        def run() -> None:
            self.repo.update_job(job_id, status="running")
            try:
                result = fn(*args, progress=progress, **kwargs)
                self.repo.update_job(job_id, status="completed", progress=1.0, result=result or {})
            except Exception as exc:  # recorded for the UI
                log.error("Job %s failed: %s\n%s", job_id, exc, traceback.format_exc())
                self.repo.update_job(job_id, status="failed", message=f"{exc.__class__.__name__}: {exc}")

        self.pools[pool].submit(run)
        return job_id

    def shutdown(self) -> None:
        for p in self.pools.values():
            p.shutdown(wait=False, cancel_futures=True)
