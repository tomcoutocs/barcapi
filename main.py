from __future__ import annotations

from fastapi import FastAPI

from app.api.chat import router as chat_router
from app.api.routes import router as rag_router

app = FastAPI(
    title="Veterinary RAG API",
    description="Ingest trusted veterinary sources, embed, and retrieve grounded context for LLMs.",
    version="0.1.0",
)

app.include_router(rag_router)
app.include_router(chat_router)


@app.get("/")
def root() -> dict[str, str]:
    return {"service": "vet-rag-api", "docs": "/docs"}
