"""
Example: ingest a local file into the vector store (same pipeline as POST /ingest).

Usage (from repo root `vet-rag-api/`, with `.env` configured):

  python -m scripts.ingest_example path/to/file.pdf --doc-type guideline --species dog --topic toxicology
"""

from __future__ import annotations

import argparse
import sys
import uuid
from pathlib import Path

# Allow running as `python scripts/ingest_example.py` by adding parent to path
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app.chunking.splitter import chunk_document
from app.config import get_settings
from app.embeddings.openai_provider import OpenAIEmbeddingProvider
from app.ingestion.loaders import load_docx, load_html, load_pdf, load_txt, normalize_document
from app.vectorstore.factory import get_vector_store


def _load(path: Path) -> str:
    suf = path.suffix.lower()
    if suf == ".pdf":
        return load_pdf(path)
    if suf == ".docx":
        return load_docx(path)
    if suf == ".txt":
        return load_txt(path)
    if suf in {".html", ".htm"}:
        return load_html(path)
    raise SystemExit(f"Unsupported extension: {suf}")


def main() -> None:
    p = argparse.ArgumentParser(description="Ingest a veterinary document into the vector index.")
    p.add_argument("file", type=Path, help="Path to PDF, DOCX, TXT, or HTML")
    p.add_argument("--doc-type", default="manual", help="manual|journal|guideline|drug|sop")
    p.add_argument("--species", default=None)
    p.add_argument("--topic", default=None)
    p.add_argument("--source-label", default=None)
    p.add_argument("--document-id", default=None, help="Stable id; default UUID")
    args = p.parse_args()

    path = args.file.expanduser().resolve()
    if not path.is_file():
        raise SystemExit(f"Not a file: {path}")

    settings = get_settings()
    text = _load(path)
    doc_id = args.document_id or str(uuid.uuid4())
    src = args.source_label or str(path)
    doc = normalize_document(
        text,
        source=src,
        document_id=doc_id,
        doc_type=args.doc_type,  # type: ignore[arg-type]
        species=args.species,
        topic=args.topic,
    )
    chunks = chunk_document(doc)
    if not chunks:
        raise SystemExit("No chunks produced (empty text?)")

    embedder = OpenAIEmbeddingProvider(settings)
    store = get_vector_store(settings)
    with_vectors = embedder.embed_chunks(chunks)
    n = store.upsert_chunks(with_vectors)
    print(f"Indexed document_id={doc_id} chunks={n}")


if __name__ == "__main__":
    main()
