"""FastAPI application entry point.

Run from the ``backend`` directory:

    uvicorn app.main:app --host 127.0.0.1 --port 8000

Interactive API documentation: http://127.0.0.1:8000/docs
"""
from __future__ import annotations

import threading
import time
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api import (routes_analytics, routes_documents, routes_evaluation, routes_gnn, routes_graph, routes_qa,
                     routes_system)
from app.core.container import get_container
from app.core.logging_setup import get_logger
from app.ingestion.document_processor import DocumentValidationError
from app.ingestion.extractors import ExtractionError
from app.pipeline.qa_pipeline import NoDocumentsError
from app.retrieval.wikipedia import WikipediaUnavailableError

log = get_logger("app")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    container = get_container()
    if not getattr(app.state, "skip_warmup", False):
        # load models in the background so the API answers /api/health immediately
        threading.Thread(target=container.warm_up, name="warmup", daemon=True).start()
    log.info("GraphRAG-QA API started (data dir: %s)", container.settings.data_dir)
    yield
    container.jobs.shutdown()


def create_app(skip_warmup: bool = False) -> FastAPI:
    app = FastAPI(
        title="GraphRAG-QA API",
        version="1.0.0",
        description="Multi-hop question answering across documents and Wikipedia with evidence graphs and a "
                    "GAT reranker. Every answer is grounded in cited evidence passages.",
        lifespan=lifespan,
    )
    app.state.skip_warmup = skip_warmup
    settings = get_container().settings
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def timing_header(request: Request, call_next):  # type: ignore[no-untyped-def]
        start = time.perf_counter()
        response = await call_next(request)
        elapsed = (time.perf_counter() - start) * 1000
        response.headers["X-Process-Time-Ms"] = f"{elapsed:.1f}"
        if request.url.path.startswith("/api") and request.method != "GET":
            log.info("%s %s -> %s (%.0f ms)", request.method, request.url.path, response.status_code, elapsed)
        return response

    def error(status: int, exc: Exception) -> JSONResponse:
        return JSONResponse(status_code=status, content={"detail": str(exc).strip("'\"")})

    @app.exception_handler(KeyError)
    async def not_found(_: Request, exc: KeyError) -> JSONResponse:
        return error(404, exc)

    @app.exception_handler(DocumentValidationError)
    async def invalid_document(_: Request, exc: DocumentValidationError) -> JSONResponse:
        return error(400, exc)

    @app.exception_handler(ExtractionError)
    async def extraction_failed(_: Request, exc: ExtractionError) -> JSONResponse:
        return error(422, exc)

    @app.exception_handler(NoDocumentsError)
    async def no_documents(_: Request, exc: NoDocumentsError) -> JSONResponse:
        return error(409, exc)

    @app.exception_handler(WikipediaUnavailableError)
    async def wiki_unavailable(_: Request, exc: WikipediaUnavailableError) -> JSONResponse:
        return error(503, exc)

    @app.exception_handler(ValueError)
    async def bad_value(_: Request, exc: ValueError) -> JSONResponse:
        return error(400, exc)

    @app.exception_handler(RequestValidationError)
    async def validation(_: Request, exc: RequestValidationError) -> JSONResponse:
        msgs = ["{}: {}".format(".".join(str(p) for p in e["loc"][1:]), e["msg"]) for e in exc.errors()]
        return JSONResponse(status_code=422, content={"detail": "; ".join(msgs)})

    for module in (routes_system, routes_documents, routes_qa, routes_graph, routes_analytics, routes_evaluation,
                   routes_gnn):
        app.include_router(module.router)

    @app.get("/", include_in_schema=False)
    def root() -> dict[str, str]:
        return {"name": "GraphRAG-QA API", "docs": "/docs", "health": "/api/health"}

    return app


app = create_app()
