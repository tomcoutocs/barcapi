from __future__ import annotations

import tempfile
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from app.api.deps import embedder_dep, retrieval_dep, vector_store_dep
from app.chunking.splitter import chunk_document
from app.embeddings.base import EmbeddingProvider
from app.ingestion.loaders import load_docx, load_html, load_pdf, load_txt, normalize_document
from app.models import IngestResponse, QueryAPIResponse, QueryRequest
from app.retrieval.service import RetrievalService
from app.vectorstore.base import VectorStore

router = APIRouter(tags=["rag"])


def _load_by_suffix(path: Path, suffix: str) -> str:
    s = suffix.lower()
    if s == ".pdf":
        return load_pdf(path)
    if s == ".docx":
        return load_docx(path)
    if s == ".txt":
        return load_txt(path)
    if s in {".html", ".htm"}:
        return load_html(path)
    raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix}")


@router.post("/ingest", response_model=IngestResponse)
async def ingest(
    file: UploadFile = File(...),
    document_id: str | None = Form(None),
    doc_type: str = Form("manual"),
    species: str | None = Form(None),
    topic: str | None = Form(None),
    source_label: str | None = Form(None),
    embedder: EmbeddingProvider = Depends(embedder_dep),
    store: VectorStore = Depends(vector_store_dep),
) -> IngestResponse:
    allowed: set[str] = {"manual", "journal", "guideline", "drug", "sop"}
    if doc_type not in allowed:
        raise HTTPException(status_code=400, detail="Invalid doc_type")
    doc_uuid = document_id or str(uuid.uuid4())
    suffix = Path(file.filename or "upload").suffix
    raw = await file.read()
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(raw)
        tmp_path = Path(tmp.name)
    try:
        text = _load_by_suffix(tmp_path, suffix)
    finally:
        tmp_path.unlink(missing_ok=True)

    src = source_label or file.filename or doc_uuid
    doc = normalize_document(
        text,
        source=src,
        document_id=doc_uuid,
        doc_type=doc_type,  # type: ignore[arg-type]
        species=species,
        topic=topic,
    )
    chunks = chunk_document(doc)
    if not chunks:
        raise HTTPException(status_code=400, detail="No extractable text in document")
    with_vectors = embedder.embed_chunks(chunks)
    n = store.upsert_chunks(with_vectors)
    return IngestResponse(document_id=doc_uuid, chunks_indexed=n)


@router.post("/query", response_model=QueryAPIResponse)
def query_endpoint(
    req: QueryRequest,
    svc: RetrievalService = Depends(retrieval_dep),
) -> QueryAPIResponse:
    resp = svc.retrieve_context(
        req.query,
        species=req.species,
        document_type=req.document_type,
        topic=req.topic,
        top_k=req.top_k,
    )
    return QueryAPIResponse(query=req.query, context=resp.results)


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
