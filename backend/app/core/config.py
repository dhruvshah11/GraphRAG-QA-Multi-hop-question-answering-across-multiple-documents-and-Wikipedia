"""Central configuration for GraphRAG-QA.

Static configuration (paths, model names, API keys) comes from environment
variables prefixed with ``GRQA_`` or from a ``.env`` file in the project root.

Runtime-tunable parameters (chunk size, top-k, hybrid alpha, ...) have their
defaults here and can be changed from the Settings page; the changed values are
persisted in SQLite by :class:`app.core.settings_service.SettingsService`.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict

# graphrag-qa/backend/app/core/config.py -> parents[3] == graphrag-qa/
PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    """Application settings. Every field can be overridden by ``GRQA_<NAME>``."""

    model_config = SettingsConfigDict(
        env_prefix="GRQA_",
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ------------------------------------------------------------------ paths
    project_root: Path = PROJECT_ROOT
    data_dir: Path = PROJECT_ROOT / "data"
    models_dir: Path = PROJECT_ROOT / "models"
    results_dir: Path = PROJECT_ROOT / "results"
    db_filename: str = "graphrag_qa.sqlite3"
    samples_path: Optional[Path] = None  # defaults to <data_dir>/samples (tests point it at the bundled samples)

    # ------------------------------------------------------------------ models
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_batch_size: int = 32
    # all-MiniLM-L6-v2 truncates at 256 word pieces; longer chunks are embedded
    # as the mean of sentence windows of at most this many words.
    embedding_window_words: int = 180
    spacy_model: str = "en_core_web_sm"
    hf_reader_model: str = "deepset/tinyroberta-squad2"
    hf_generator_model: str = "google/flan-t5-base"
    device: str = "cpu"

    # ------------------------------------------------- runtime-tunable defaults
    chunk_size: int = 400  # tokens (word-level tokens, see preprocessing/tokenize.py)
    chunk_overlap: int = 80
    top_k: int = 8
    max_hops: int = 3
    hybrid_alpha: float = 0.7
    generator_backend: str = "auto"  # auto|extractive|hf_reader|hf_generator|ollama|anthropic
    max_evidence_passages: int = 5

    # ------------------------------------------------------------- ingestion
    max_upload_mb: int = 25
    allowed_extensions: tuple[str, ...] = (".pdf", ".txt", ".docx")

    # --------------------------------------------------------------- graph
    # entities mentioned in more than this fraction of passages are treated as
    # "hub" entities and are not used as bridges between passages.
    max_bridge_entity_df_ratio: float = 0.25
    max_bridge_entity_df_abs: int = 40
    graph_expansion_per_hop: int = 12
    max_chains: int = 8

    # ----------------------------------------------------- answer thresholds
    min_semantic_score: float = 0.22
    min_term_coverage: float = 0.30
    min_confidence: float = 0.30

    # ------------------------------------------------------------- Wikipedia
    wikipedia_api: str = "https://en.wikipedia.org/w/api.php"
    # Wikimedia requires contact details (URL or e-mail) in the User-Agent; set your own via
    # GRQA_WIKIPEDIA_USER_AGENT, e.g. "GraphRAG-QA/1.0 (https://your-site.example; you@example.com)".
    wikipedia_user_agent: str = "GraphRAG-QA/1.0 (academic NLP project; contact not configured)"
    wikipedia_timeout: float = 15.0
    wikipedia_max_articles: int = 6
    wikipedia_bridge_articles: int = 3
    wikipedia_max_chunks_per_article: int = 12

    # ------------------------------------------------------- LLM API backends
    # Optional local LLM served by Ollama (native /api/chat endpoint).
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = ""  # e.g. "llama3.1:8b"; empty = backend disabled
    # Optional Claude API backend (official `anthropic` SDK). Credentials are
    # resolved by the SDK (ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN or `ant auth login`).
    anthropic_model: str = "claude-opus-5-5"
    anthropic_effort: str = "medium"  # low | medium | high | xhigh | max
    llm_timeout: float = 120.0
    reader_null_threshold: float = 0.3

    # ------------------------------------------------------------------- GNN
    gnn_hidden_dim: int = 64
    gnn_heads: int = 4
    gnn_dropout: float = 0.2

    # ------------------------------------------------------------------ misc
    seed: int = 42
    cors_origins: tuple[str, ...] = (
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:4173",
        "http://127.0.0.1:4173",
    )
    log_level: str = "INFO"

    # ------------------------------------------------------- derived paths
    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def indexes_dir(self) -> Path:
        return self.data_dir / "indexes"

    @property
    def graphs_dir(self) -> Path:
        return self.data_dir / "graphs"

    @property
    def query_graphs_dir(self) -> Path:
        return self.data_dir / "graphs" / "queries"

    @property
    def samples_dir(self) -> Path:
        return self.samples_path or self.data_dir / "samples"

    @property
    def db_path(self) -> Path:
        return self.data_dir / self.db_filename

    @property
    def embedding_cache_dir(self) -> Path:
        return self.models_dir / "embeddings"

    @property
    def generation_cache_dir(self) -> Path:
        return self.models_dir / "generation"

    @property
    def gnn_dir(self) -> Path:
        return self.models_dir / "gnn"

    @property
    def gnn_checkpoint_path(self) -> Path:
        return self.gnn_dir / "gat_reranker.pt"

    def ensure_dirs(self) -> None:
        """Create every directory the application writes to."""
        for path in (
            self.data_dir,
            self.uploads_dir,
            self.processed_dir,
            self.indexes_dir,
            self.graphs_dir,
            self.query_graphs_dir,
            self.samples_dir,
            self.models_dir,
            self.embedding_cache_dir,
            self.generation_cache_dir,
            self.gnn_dir,
            self.results_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_dirs()
    return settings
