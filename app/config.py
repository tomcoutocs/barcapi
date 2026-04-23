from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# `app/config.py` -> service root is `vet-rag-api/`, repo root is parent (e.g. `barc/`)
_SERVICE_ROOT = Path(__file__).resolve().parent.parent
_REPO_ROOT = _SERVICE_ROOT.parent


def _resolved_env_files() -> tuple[str, ...]:
    """
    Dotenv load order (later files override earlier for the same key):
    vet-rag-api/.env -> repo/.env -> vet-rag-api/.env.local -> repo/.env.local -> VET_RAG_ENV_FILE
    Real process environment variables still win over all dotenv files.
    """
    candidates: list[Path] = [
        _SERVICE_ROOT / ".env",
        _REPO_ROOT / ".env",
        _SERVICE_ROOT / ".env.local",
        _REPO_ROOT / ".env.local",
    ]
    override = os.environ.get("VET_RAG_ENV_FILE", "").strip()
    if override:
        candidates.append(Path(override).expanduser())

    seen: set[Path] = set()
    ordered: list[Path] = []
    for p in candidates:
        try:
            resolved = p.resolve()
        except OSError:
            continue
        if resolved.is_file() and resolved not in seen:
            seen.add(resolved)
            ordered.append(resolved)

    if not ordered:
        return (str(_SERVICE_ROOT / ".env"),)
    return tuple(str(p) for p in ordered)


_ENV_FILES = _resolved_env_files()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_ENV_FILES,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openai_api_key: str = Field(default="", validation_alias="OPENAI_API_KEY")
    openai_embedding_model: str = Field(
        default="text-embedding-3-large",
        validation_alias="OPENAI_EMBEDDING_MODEL",
    )
    openai_embedding_dimensions: int | None = Field(
        default=None,
        validation_alias="OPENAI_EMBEDDING_DIMENSIONS",
    )
    openai_chat_model: str = Field(
        default="gpt-4o-mini",
        validation_alias="OPENAI_CHAT_MODEL",
    )

    agent_eval_log_path: str = Field(
        default="logs/agent_interactions.jsonl",
        validation_alias="AGENT_EVAL_LOG_PATH",
    )

    vector_backend: str = Field(default="pinecone", validation_alias="VECTOR_BACKEND")

    pinecone_api_key: str = Field(default="", validation_alias="PINECONE_API_KEY")
    pinecone_index_name: str = Field(default="vet-rag", validation_alias="PINECONE_INDEX_NAME")
    pinecone_host: str | None = Field(default=None, validation_alias="PINECONE_HOST")

    weaviate_url: str = Field(default="http://localhost:8080", validation_alias="WEAVIATE_URL")
    weaviate_api_key: str = Field(default="", validation_alias="WEAVIATE_API_KEY")
    weaviate_class_name: str = Field(default="VetChunk", validation_alias="WEAVIATE_CLASS_NAME")

    retrieval_top_k: int = Field(default=8, ge=1, le=50, validation_alias="RETRIEVAL_TOP_K")
    source_quality_enabled: bool = Field(default=True, validation_alias="SOURCE_QUALITY_ENABLED")

    query_cache_enabled: bool = Field(default=True, validation_alias="QUERY_CACHE_ENABLED")
    query_cache_max_entries: int = Field(
        default=256,
        ge=0,
        validation_alias="QUERY_CACHE_MAX_ENTRIES",
    )

    rerank_enabled: bool = Field(default=False, validation_alias="RERANK_ENABLED")
    rerank_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        validation_alias="RERANK_MODEL",
    )

    api_host: str = Field(default="0.0.0.0", validation_alias="API_HOST")
    api_port: int = Field(default=8000, validation_alias="API_PORT")

    merck_crawl_max_articles: int = Field(
        default=2500,
        ge=1,
        le=10_000,
        validation_alias="MERCK_CRAWL_MAX_ARTICLES",
    )
    merck_crawl_max_visits: int = Field(
        default=35_000,
        ge=50,
        le=50_000,
        validation_alias="MERCK_CRAWL_MAX_VISITS",
    )
    merck_crawl_delay_s: float = Field(
        default=1.0,
        ge=0.25,
        validation_alias="MERCK_CRAWL_DELAY_S",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
