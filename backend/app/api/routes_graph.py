"""Knowledge-graph explorer endpoints."""
from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, Depends, Query
from fastapi.concurrency import run_in_threadpool

from app.api.controller import APIController
from app.api.deps import get_controller
from app.schemas.api import GraphQueryRequest

router = APIRouter(prefix="/api/graph", tags=["graph"])


@router.get("", summary="Corpus evidence graph (limited to the most connected nodes)")
def full_graph(limit: int = Query(300, ge=10, le=3000), search: Optional[str] = None,
               node_types: Optional[str] = Query(None, description="comma separated: document,passage,entity"),
               ctl: APIController = Depends(get_controller)) -> dict[str, Any]:
    types = [t.strip() for t in node_types.split(",")] if node_types else None
    return ctl.full_graph(limit=limit, search=search, node_types=types)


@router.post("/query", summary="Question-specific subgraph or evidence-chain graph")
async def graph_query(req: GraphQueryRequest, ctl: APIController = Depends(get_controller)) -> dict[str, Any]:
    return await run_in_threadpool(ctl.graph_query, req.query_id, req.question, req.view, req.top_k, req.max_hops)


@router.get("/node/{node_id:path}/neighbors", summary="Neighbourhood of a node (for expansion)")
def neighbors(node_id: str, ctl: APIController = Depends(get_controller)) -> dict[str, Any]:
    return ctl.neighbors(node_id)
