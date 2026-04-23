from __future__ import annotations

from app.config import Settings, get_settings
from app.vectorstore.base import VectorStore


def get_vector_store(settings: Settings | None = None) -> VectorStore:
    s = settings or get_settings()
    backend = (s.vector_backend or "pinecone").lower().strip()
    if backend == "pinecone":
        from app.vectorstore.pinecone_store import PineconeVectorStore

        return PineconeVectorStore(s)
    if backend in {"weaviate", "weaviate_cloud"}:
        from app.vectorstore.weaviate_store import WeaviateVectorStore

        return WeaviateVectorStore(s)
    raise ValueError(f"Unsupported VECTOR_BACKEND: {s.vector_backend}")
