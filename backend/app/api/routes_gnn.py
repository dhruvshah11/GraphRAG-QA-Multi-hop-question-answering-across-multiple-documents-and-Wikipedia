"""GNN training and status endpoints."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends

from app.api.controller import APIController
from app.api.deps import get_controller
from app.schemas.api import TrainRequest

router = APIRouter(prefix="/api/gnn", tags=["gnn"])


@router.post("/train", summary="Train the GAT reranker (background job)")
def train(req: TrainRequest, ctl: APIController = Depends(get_controller)) -> dict[str, Any]:
    return ctl.start_training(**req.model_dump())


@router.get("/status", summary="Checkpoint status, metadata and training history")
def status(ctl: APIController = Depends(get_controller)) -> dict[str, Any]:
    return ctl.gnn_status()
