"""Evaluation lab endpoints."""
from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, Depends
from fastapi.responses import Response

from app.api.controller import APIController
from app.api.deps import get_controller
from app.schemas.api import EvaluationRequest

router = APIRouter(prefix="/api/evaluation", tags=["evaluation"])


@router.post("/run", summary="Start an evaluation run (background job)")
def run(req: EvaluationRequest, ctl: APIController = Depends(get_controller)) -> dict[str, Any]:
    return ctl.start_evaluation(**req.model_dump())


@router.get("/results", summary="List runs, or one run with per-question rows (?run_id=)")
def results(run_id: Optional[str] = None, ctl: APIController = Depends(get_controller)) -> dict[str, Any]:
    return ctl.evaluation_results(run_id)


@router.get("/results/{run_id}/csv", summary="Download a run as CSV")
def results_csv(run_id: str, ctl: APIController = Depends(get_controller)) -> Response:
    headers = {"Content-Disposition": "attachment; filename=" + run_id + ".csv"}
    return Response(ctl.evaluation_csv(run_id), media_type="text/csv", headers=headers)


@router.get("/datasets", summary="Available evaluation datasets")
def datasets(ctl: APIController = Depends(get_controller)) -> dict[str, Any]:
    return {"datasets": ctl.datasets()}
