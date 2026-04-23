from __future__ import annotations

import logging

from app.pipeline.ingest_documents import ingest_documents
from app.utils.cleaner import clean_text
from app.utils.deduper import ContentDeduper
from app.utils.metadata import enrich_metadata

logger = logging.getLogger(__name__)


def process_and_ingest(scraped_docs: list[dict], *, min_words: int = 300) -> int:
    """
    For each scraped payload: clean → dedupe (hash + min word count) → enrich metadata → ingest.

    Returns count of documents successfully passed to ingest_documents.
    """
    deduper = ContentDeduper(min_words=min_words)
    ingested = 0
    for raw in scraped_docs:
        url = raw.get("url") or ""
        raw_text = raw.get("raw_text") or ""
        title = raw.get("title") or ""
        source_type = raw.get("source_type") or "merck"
        text = clean_text(raw_text)
        if not deduper.accept(text):
            logger.info("process_and_ingest: skip (short or duplicate) url=%s", url)
            continue
        meta = enrich_metadata(text, source=url, title=title, source_type=str(source_type))
        ingest_documents([{"text": text, "metadata": meta}])
        ingested += 1
    return ingested
