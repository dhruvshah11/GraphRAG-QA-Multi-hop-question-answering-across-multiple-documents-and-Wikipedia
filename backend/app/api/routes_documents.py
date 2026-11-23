"""Document workspace endpoints."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile

from app.api.controller import APIController
from app.api.deps import get_controller

router = APIRouter(prefix="/api/documents", tags=["documents"])


@router.post("/upload", summary="Upload PDF/TXT/DOCX files (processed in the background)")
async def upload_documents(files: list[UploadFile] = File(...), auto_process: bool = Query(True),
                           ctl: APIController = Depends(get_controller)) -> dict[str, Any]:
    max_bytes = ctl.c.settings.max_upload_mb * 1024 * 1024
    payload = []
    for f in files:
        data = await f.read(max_bytes + 1)
        if len(data) > max_bytes:
            raise HTTPException(413, f"{f.filename} exceeds the {ctl.c.settings.max_upload_mb} MB limit")
        payload.append((f.filename or "document", data))
    return {"results": ctl.upload(payload, auto_process=auto_process)}


@router.get("", summary="List uploaded documents with ingestion statistics")
def list_documents(ctl: APIController = Depends(get_controller)) -> dict[str, Any]:
    return ctl.list_documents()


@router.get("/{document_id}", summary="Document metadata, extracted text, chunks and entities")
def get_document(document_id: str, ctl: APIController = Depends(get_controller)) -> dict[str, Any]:
    return ctl.get_document(document_id)


@router.delete("/{document_id}", summary="Delete a document and remove it from all indexes")
def delete_document(document_id: str, ctl: APIController = Depends(get_controller)) -> dict[str, Any]:
    return ctl.delete_document(document_id)


@router.post("/{document_id}/process", summary="(Re)process a document")
def process_document(document_id: str, ctl: APIController = Depends(get_controller)) -> dict[str, Any]:
    return ctl.process(document_id)
