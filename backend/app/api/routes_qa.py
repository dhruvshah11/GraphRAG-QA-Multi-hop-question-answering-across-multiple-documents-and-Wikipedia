"""Question answering endpoints."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends
from fastapi.concurrency import run_in_threadpool

from app.api.controller import APIController
from app.api.deps import get_controller
from app.schemas.api import AskRequest, OpenDomainRequest, QAResponse

router = APIRouter(prefix="/api/qa", tags=["question answering"])


@router.post("/ask", response_model=QAResponse, summary="Answer a question (multi-document or open-domain)")
async def ask(req: AskRequest, ctl: APIController = Depends(get_controller)) -> dict[str, Any]:
    return await run_in_threadpool(ctl.ask, req.question, req.mode, req.strategy, req.top_k, req.max_hops,
                                   req.backend)


@router.post("/open-domain", response_model=QAResponse, summary="Open-domain multi-hop QA over Wikipedia")
async def open_domain(req: OpenDomainRequest, ctl: APIController = Depends(get_controller)) -> dict[str, Any]:
    return await run_in_threadpool(ctl.ask, req.question, "open_domain", req.strategy, req.top_k, req.max_hops,
                                   req.backend)


@router.get("/history", summary="Recently answered questions")
def history(limit: int = 20, ctl: APIController = Depends(get_controller)) -> dict[str, Any]:
    return {"queries": ctl.c.repo.list_queries(limit=limit)}


@router.get("/{query_id}", summary="Stored answer payload of a previous question")
def get_query(query_id: str, ctl: APIController = Depends(get_controller)) -> dict[str, Any]:
    return ctl.query(query_id)
