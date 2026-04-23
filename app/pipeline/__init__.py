"""Scrape → clean → dedupe → enrich → existing RAG ingestion."""

from app.pipeline.ingest_documents import ingest_documents
from app.pipeline.process_and_ingest import process_and_ingest

__all__ = ["ingest_documents", "process_and_ingest"]
