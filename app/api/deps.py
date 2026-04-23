from __future__ import annotations

from functools import lru_cache

from app.config import Settings, get_settings
from app.embeddings.base import EmbeddingProvider
from app.embeddings.openai_provider import OpenAIEmbeddingProvider
from app.retrieval.service import RetrievalService
from app.vectorstore.base import VectorStore
from app.vectorstore.factory import get_vector_store


@lru_cache
def settings_dep() -> Settings:
    return get_settings()


@lru_cache
def vector_store_dep() -> VectorStore:
    return get_vector_store(settings_dep())


@lru_cache
def embedder_dep() -> EmbeddingProvider:
    return OpenAIEmbeddingProvider(settings_dep())


@lru_cache
def retrieval_dep() -> RetrievalService:
    return RetrievalService(
        settings=settings_dep(),
        vector_store=vector_store_dep(),
        embedder=embedder_dep(),
    )
