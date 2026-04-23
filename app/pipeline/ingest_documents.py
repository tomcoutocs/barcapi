from __future__ import annotations

import logging
from typing import Any, Literal
from uuid import NAMESPACE_URL, uuid5

from app.chunking.splitter import chunk_document
from app.config import get_settings
from app.embeddings.openai_provider import OpenAIEmbeddingProvider
from app.ingestion.loaders import normalize_document
from app.models import DocumentType
from app.vectorstore.factory import get_vector_store

logger = logging.getLogger(__name__)

IngestMetaType = Literal["manual", "guideline", "org"]


def _to_document_type(t: IngestMetaType) -> DocumentType:
    if t == "org":
        return "journal"
    if t in ("manual", "guideline"):
        return t
    return "manual"


def ingest_documents(documents: list[dict[str, Any]]) -> None:
    """
    Push structured documents through the existing pipeline:
    normalize → chunk → embed → vector upsert.

    Each document:
      { "text": str, "metadata": { source, title, type, species, topic, urgency, authority_weight } }
    """
    if not documents:
        return
    settings = get_settings()
    embedder = OpenAIEmbeddingProvider(settings)
    store = get_vector_store(settings)

    for doc in documents:
        text = (doc.get("text") or "").strip()
        meta = doc.get("metadata") or {}
        if not text:
            logger.warning("ingest_documents: skip empty text for source=%s", meta.get("source"))
            continue

        source_url = str(meta.get("source", "")).strip()
        if not source_url:
            logger.warning("ingest_documents: skip missing metadata.source")
            continue

        doc_id = str(uuid5(NAMESPACE_URL, source_url))
        title = str(meta.get("title", "")).strip()
        species = str(meta.get("species") or "dog").strip()
        topic = str(meta.get("topic") or "").strip() or None
        urgency = str(meta.get("urgency") or "").strip() or None
        raw_type = meta.get("type", "manual")
        if raw_type not in ("manual", "guideline", "org"):
            raw_type = "manual"
        doc_type = _to_document_type(raw_type)  # type: ignore[arg-type]

        try:
            aw = float(meta.get("authority_weight"))
        except (TypeError, ValueError):
            aw = None

        normalized = normalize_document(
            text,
            source=source_url,
            document_id=doc_id,
            doc_type=doc_type,
            species=species,
            topic=topic,
            extra_metadata={
                "title": title or None,
                "urgency": urgency,
                "authority_weight": aw,
            },
        )
        chunks = chunk_document(normalized)
        if not chunks:
            logger.warning("ingest_documents: no chunks for source=%s", source_url)
            continue
        batch = embedder.embed_chunks(chunks)
        n = store.upsert_chunks(batch)
        logger.info("ingest_documents: upserted %s chunks for %s", n, source_url)
