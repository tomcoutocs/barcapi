from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, TypedDict

from app.models import ChunkWithVector


class VectorQueryHit(TypedDict, total=False):
    id: str
    score: float
    text: str
    source: str
    metadata: dict[str, Any]


class VectorStore(ABC):
    @abstractmethod
    def upsert_chunks(self, chunks_with_embeddings: list[ChunkWithVector]) -> int:
        """Persist vectors; returns count upserted."""

    @abstractmethod
    def query_similar(
        self,
        vector: list[float],
        *,
        top_k: int = 5,
        filters: dict[str, str | None] | None = None,
    ) -> list[VectorQueryHit]:
        """Return ranked hits with scores and payload for LLM context."""
