from __future__ import annotations

from typing import Any

from pinecone import Pinecone

from app.config import Settings
from app.models import ChunkWithVector
from app.vectorstore.base import VectorQueryHit, VectorStore


def _pinecone_metadata_filter(filters: dict[str, str | None] | None) -> dict | None:
    if not filters:
        return None
    parts: list[dict] = []
    for key, val in filters.items():
        if val is None or val == "":
            continue
        parts.append({key: {"$eq": val}})
    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]
    return {"$and": parts}


def _matches_from_response(res: Any) -> list[Any]:
    matches = getattr(res, "matches", None)
    if matches is None and isinstance(res, dict):
        matches = res.get("matches")
    return list(matches or [])


def _match_fields(match: Any) -> tuple[str, float, dict]:
    if isinstance(match, dict):
        mid = str(match.get("id", ""))
        score = float(match.get("score", 0.0))
        md = dict(match.get("metadata") or {})
        return mid, score, md
    mid = str(getattr(match, "id", "") or "")
    score = float(getattr(match, "score", 0.0) or 0.0)
    md_obj = getattr(match, "metadata", None) or {}
    md = dict(md_obj) if isinstance(md_obj, dict) else {}
    return mid, score, md


class PineconeVectorStore(VectorStore):
    def __init__(self, settings: Settings):
        if not settings.pinecone_api_key:
            raise ValueError("PINECONE_API_KEY is required for Pinecone backend")
        self._pc = Pinecone(api_key=settings.pinecone_api_key)
        if settings.pinecone_host:
            self._index = self._pc.Index(host=settings.pinecone_host)
        else:
            self._index = self._pc.Index(name=settings.pinecone_index_name)

    def upsert_chunks(self, chunks_with_embeddings: list[ChunkWithVector]) -> int:
        if not chunks_with_embeddings:
            return 0
        vectors: list[dict] = []
        for item in chunks_with_embeddings:
            c = item.chunk
            meta: dict[str, Any] = {
                "text": c.text[:40000],
                "document_id": c.source,
                "chunk_id": c.chunk_id,
                "type": str(c.metadata.get("type", "manual")),
                **({"species": c.metadata["species"]} if c.metadata.get("species") else {}),
                **({"topic": c.metadata["topic"]} if c.metadata.get("topic") else {}),
                **(
                    {"source_label": c.metadata["source_label"]}
                    if c.metadata.get("source_label")
                    else {}
                ),
                **({"title": c.metadata["title"]} if c.metadata.get("title") else {}),
                **({"urgency": c.metadata["urgency"]} if c.metadata.get("urgency") else {}),
                **(
                    {"authority_weight": float(c.metadata["authority_weight"])}
                    if c.metadata.get("authority_weight") is not None
                    else {}
                ),
            }
            vectors.append({"id": c.chunk_id, "values": item.vector, "metadata": meta})
        batch = 100
        for i in range(0, len(vectors), batch):
            self._index.upsert(vectors=vectors[i : i + batch])
        return len(vectors)

    def query_similar(
        self,
        vector: list[float],
        *,
        top_k: int = 5,
        filters: dict[str, str | None] | None = None,
    ) -> list[VectorQueryHit]:
        fl = _pinecone_metadata_filter(filters)
        res = self._index.query(
            vector=vector,
            top_k=top_k,
            include_metadata=True,
            filter=fl,
        )
        hits: list[VectorQueryHit] = []
        for match in _matches_from_response(res):
            mid, score, md = _match_fields(match)
            hits.append(
                {
                    "id": mid,
                    "score": score,
                    "text": str(md.get("text", "")),
                    "source": str(md.get("document_id", "")),
                    "metadata": {
                        "type": md.get("type"),
                        "species": md.get("species"),
                        "topic": md.get("topic"),
                        "source_label": md.get("source_label"),
                        "chunk_id": md.get("chunk_id"),
                        "title": md.get("title"),
                        "urgency": md.get("urgency"),
                        "authority_weight": md.get("authority_weight"),
                    },
                }
            )
        return hits
