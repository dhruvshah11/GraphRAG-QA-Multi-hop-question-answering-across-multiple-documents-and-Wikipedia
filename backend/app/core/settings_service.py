"""Runtime-tunable settings (persisted in SQLite, editable from the UI)."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, model_validator

from app.core.config import Settings

KNOWN_EMBEDDING_MODELS = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-MiniLM-L12-v2",
    "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
    "sentence-transformers/all-mpnet-base-v2",
]
GENERATOR_BACKENDS = ["auto", "extractive", "hf_reader", "hf_generator", "ollama", "anthropic"]
RUNTIME_KEYS = ("chunk_size", "chunk_overlap", "top_k", "max_hops", "hybrid_alpha", "generator_backend",
                "max_evidence_passages", "embedding_model")


class RuntimeSettings(BaseModel):
    chunk_size: int = Field(ge=50, le=2000, description="Chunk size in word-level tokens")
    chunk_overlap: int = Field(ge=0, le=500, description="Overlap between consecutive chunks (tokens)")
    top_k: int = Field(ge=1, le=50, description="Passages retrieved by first-stage retrieval")
    max_hops: int = Field(ge=1, le=4, description="Maximum passage hops in evidence chains")
    hybrid_alpha: float = Field(ge=0.0, le=1.0, description="Weight of semantic score in hybrid retrieval")
    generator_backend: str
    max_evidence_passages: int = Field(ge=1, le=10)
    embedding_model: str

    @model_validator(mode="after")
    def _check(self) -> "RuntimeSettings":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        if self.generator_backend not in GENERATOR_BACKENDS:
            raise ValueError(f"generator_backend must be one of {GENERATOR_BACKENDS}")
        if self.embedding_model not in KNOWN_EMBEDDING_MODELS:
            raise ValueError(f"embedding_model must be one of {KNOWN_EMBEDDING_MODELS}")
        return self


class SettingsService:
    def __init__(self, settings: Settings, repo: Any) -> None:
        self.settings = settings
        self.repo = repo

    def defaults(self) -> dict[str, Any]:
        return {k: getattr(self.settings, k) for k in RUNTIME_KEYS}

    def get(self) -> RuntimeSettings:
        merged = {**self.defaults(), **{k: v for k, v in self.repo.get_setting_overrides().items() if k in RUNTIME_KEYS}}
        return RuntimeSettings(**merged)

    def update(self, changes: dict[str, Any]) -> tuple[RuntimeSettings, dict[str, Any]]:
        current = self.get()
        unknown = set(changes) - set(RUNTIME_KEYS)
        if unknown:
            raise ValueError(f"Unknown settings: {sorted(unknown)}")
        updated = RuntimeSettings(**{**current.model_dump(), **changes})  # validates
        diff = {k: v for k, v in updated.model_dump().items() if getattr(current, k) != v}
        if diff:
            self.repo.set_settings(diff)
        effects = {
            "changed": sorted(diff),
            "requires_reprocessing": bool({"chunk_size", "chunk_overlap"} & set(diff)),
            "requires_reindex": "embedding_model" in diff,
        }
        return updated, effects
