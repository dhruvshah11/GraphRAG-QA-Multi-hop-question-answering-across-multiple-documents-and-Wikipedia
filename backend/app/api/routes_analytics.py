"""Dashboard and analytics endpoints (all values are measured, never mocked)."""
from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, Depends

from app.api.controller import APIController
from app.api.deps import get_controller

router = APIRouter(prefix="/api/analytics", tags=["analytics"])


@router.get("/overview", summary="Dashboard counts, pipeline status, recent queries")
def overview(ctl: APIController = Depends(get_controller)) -> dict[str, Any]:
    return ctl.overview()


@router.get("/retrieval", summary="Per-passage scores of every strategy for a query + evaluation summaries")
def retrieval(query_id: Optional[str] = None, ctl: APIController = Depends(get_controller)) -> dict[str, Any]:
    return ctl.retrieval_analytics(query_id)


@router.get("/performance", summary="Measured stage latencies and benchmark results")
def performance(ctl: APIController = Depends(get_controller)) -> dict[str, Any]:
    return ctl.performance()
