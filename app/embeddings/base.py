from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol

from app.models import Chunk, ChunkWithVector


class EmbeddingProvider(ABC):
    """Pluggable embedding backend (OpenAI today; swap for local models later)."""

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError

    def embed_chunks(self, chunks: list[Chunk]) -> list[ChunkWithVector]:
        if not chunks:
            return []
        texts = [c.text for c in chunks]
        vectors = self.embed_texts(texts)
        if len(vectors) != len(chunks):
            raise RuntimeError("Embedding provider returned unexpected vector count")
        return [ChunkWithVector(chunk=c, vector=v) for c, v in zip(chunks, vectors, strict=True)]


class SupportsEmbed(Protocol):
    def embed_texts(self, texts: list[str]) -> list[list[float]]: ...
