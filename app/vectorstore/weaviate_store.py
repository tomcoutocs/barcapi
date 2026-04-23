from __future__ import annotations

from typing import Any
from urllib.parse import urlparse

import weaviate
from weaviate.classes.config import Configure, DataType, Property
from weaviate.classes.query import Filter, MetadataQuery
from app.config import Settings
from app.models import ChunkWithVector
from app.vectorstore.base import VectorQueryHit, VectorStore


class WeaviateVectorStore(VectorStore):
    def __init__(self, settings: Settings):
        self._class_name = settings.weaviate_class_name
        self._client = self._connect(settings)
        self._ensure_schema()

    def _connect(self, settings: Settings) -> weaviate.WeaviateClient:
        parsed = urlparse(settings.weaviate_url)
        host = parsed.hostname or "localhost"
        scheme = parsed.scheme or "http"
        port = parsed.port or (443 if scheme == "https" else 8080)
        is_cloud = bool(settings.weaviate_api_key) and (
            "weaviate.cloud" in (host or "")
            or "weaviate.network" in (host or "")
            or scheme == "https"
        )
        if is_cloud:
            from weaviate.auth import Auth

            return weaviate.connect_to_weaviate_cloud(
                cluster_url=settings.weaviate_url,
                auth_credentials=Auth.api_key(settings.weaviate_api_key),
            )
        grpc_port = 50051
        return weaviate.connect_to_custom(
            http_host=host or "localhost",
            http_port=port,
            http_secure=scheme == "https",
            grpc_host=host or "localhost",
            grpc_port=grpc_port,
            grpc_secure=scheme == "https",
        )

    def _ensure_schema(self) -> None:
        if self._client.collections.exists(self._class_name):
            return
        self._client.collections.create(
            name=self._class_name,
            vectorizer_config=Configure.Vectorizer.none(),
            properties=[
                Property(name="text", data_type=DataType.TEXT),
                Property(name="document_id", data_type=DataType.TEXT),
                Property(name="chunk_id", data_type=DataType.TEXT),
                Property(name="type", data_type=DataType.TEXT),
                Property(name="species", data_type=DataType.TEXT),
                Property(name="topic", data_type=DataType.TEXT),
                Property(name="source_label", data_type=DataType.TEXT),
                Property(name="title", data_type=DataType.TEXT),
                Property(name="urgency", data_type=DataType.TEXT),
                Property(name="authority_weight", data_type=DataType.NUMBER),
            ],
        )

    def upsert_chunks(self, chunks_with_embeddings: list[ChunkWithVector]) -> int:
        collection = self._client.collections.get(self._class_name)
        count = 0
        for item in chunks_with_embeddings:
            c = item.chunk
            aw = c.metadata.get("authority_weight")
            props: dict[str, Any] = {
                "text": c.text,
                "document_id": c.source,
                "chunk_id": c.chunk_id,
                "type": str(c.metadata.get("type", "manual")),
                "species": c.metadata.get("species") or "",
                "topic": c.metadata.get("topic") or "",
                "source_label": c.metadata.get("source_label") or "",
                "title": c.metadata.get("title") or "",
                "urgency": c.metadata.get("urgency") or "",
                "authority_weight": float(aw) if aw is not None else 0.0,
            }
            try:
                collection.data.replace(
                    uuid=c.chunk_id,
                    properties=props,
                    vector=item.vector,
                )
            except Exception:
                collection.data.insert(
                    properties=props,
                    vector=item.vector,
                    uuid=c.chunk_id,
                )
            count += 1
        return count

    def query_similar(
        self,
        vector: list[float],
        *,
        top_k: int = 5,
        filters: dict[str, str | None] | None = None,
    ) -> list[VectorQueryHit]:
        collection = self._client.collections.get(self._class_name)
        where: Filter | None = None
        if filters:
            for key, val in filters.items():
                if val is None or val == "":
                    continue
                clause = Filter.by_property(key).equal(val)
                where = clause if where is None else where & clause
        res = collection.query.near_vector(
            near_vector=vector,
            limit=top_k,
            filters=where,
            return_metadata=MetadataQuery(distance=True),
        )
        hits: list[VectorQueryHit] = []
        for obj in res.objects:
            md = obj.metadata
            distance = getattr(md, "distance", None) if md else None
            score = 1.0 - float(distance) if distance is not None else 0.0
            props = obj.properties or {}
            hits.append(
                {
                    "id": str(props.get("chunk_id", obj.uuid)),
                    "score": score,
                    "text": str(props.get("text", "")),
                    "source": str(props.get("document_id", "")),
                    "metadata": {
                        "type": props.get("type"),
                        "species": props.get("species") or None,
                        "topic": props.get("topic") or None,
                        "source_label": props.get("source_label") or None,
                        "chunk_id": props.get("chunk_id"),
                        "title": props.get("title") or None,
                        "urgency": props.get("urgency") or None,
                        "authority_weight": props.get("authority_weight"),
                    },
                }
            )
        return hits

    def close(self) -> None:
        self._client.close()
