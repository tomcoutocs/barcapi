from __future__ import annotations

from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

DocumentType = Literal["manual", "journal", "guideline", "drug", "sop"]


class DocumentMetadata(BaseModel):
    model_config = ConfigDict(extra="ignore")

    type: DocumentType = "manual"
    species: str | None = None
    topic: str | None = None
    title: str | None = None
    urgency: str | None = None
    authority_weight: float | None = None


class NormalizedDocument(BaseModel):
    text: str
    source: str
    document_id: str
    metadata: DocumentMetadata = Field(default_factory=DocumentMetadata)


class Chunk(BaseModel):
    chunk_id: str
    text: str
    source: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChunkWithVector(BaseModel):
    chunk: Chunk
    vector: list[float]


class RetrievalHit(BaseModel):
    text: str
    source: str
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievalResponse(BaseModel):
    query: str
    results: list[RetrievalHit]


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=8000)
    species: str | None = None
    document_type: DocumentType | None = None
    topic: str | None = None
    top_k: int | None = Field(default=None, ge=1, le=50)


class QueryAPIResponse(BaseModel):
    query: str
    context: list[RetrievalHit]


class IngestMetadataForm(BaseModel):
    document_id: str | None = None
    doc_type: DocumentType = "manual"
    species: str | None = None
    topic: str | None = None
    source_label: str | None = None


class IngestResponse(BaseModel):
    document_id: str
    chunks_indexed: int
    message: str = "ok"


def is_uuid(value: str) -> bool:
    try:
        UUID(value)
        return True
    except ValueError:
        return False
