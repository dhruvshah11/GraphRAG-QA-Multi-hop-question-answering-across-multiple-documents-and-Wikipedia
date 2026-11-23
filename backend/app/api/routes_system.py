"""Health, settings and job endpoints."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from app.api.controller import APIController
from app.api.deps import get_controller
from app.schemas.api import JobOut, SettingsUpdate

router = APIRouter(prefix="/api", tags=["system"])


@router.get("/health", summary="Component health (models, index, GNN, generator)")
def health(check_wikipedia: bool = False, ctl: APIController = Depends(get_controller)) -> dict[str, Any]:
    return ctl.health(check_wikipedia=check_wikipedia)


@router.get("/settings", summary="Runtime settings, model status and storage information")
def get_settings(ctl: APIController = Depends(get_controller)) -> dict[str, Any]:
    return ctl.get_settings()


@router.put("/settings", summary="Update runtime settings")
def update_settings(req: SettingsUpdate, ctl: APIController = Depends(get_controller)) -> dict[str, Any]:
    return ctl.update_settings(req.model_dump(exclude_none=True))


@router.post("/settings/reprocess-all", summary="Reprocess every document (after chunk-size changes)")
def reprocess_all(ctl: APIController = Depends(get_controller)) -> dict[str, Any]:
    return ctl.reprocess_all()


@router.get("/jobs/{job_id}", response_model=JobOut, summary="Background job status")
def job(job_id: str, ctl: APIController = Depends(get_controller)) -> dict[str, Any]:
    j = ctl.c.repo.get_job(job_id)
    if j is None:
        raise HTTPException(404, f"Job {job_id} not found")
    return j
