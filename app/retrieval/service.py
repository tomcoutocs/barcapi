from __future__ import annotations

import hashlib
import re
from typing import Any

from cachetools import LRUCache

from app.config import Settings, get_settings
from app.embeddings.base import EmbeddingProvider
from app.models import RetrievalHit, RetrievalResponse
from app.retrieval.rerank import rerank_with_cross_encoder
from app.vectorstore.base import VectorQueryHit, VectorStore

_SOURCE_QUALITY_WEIGHT: dict[str, float] = {
    "manual": 1.0,
    "guideline": 0.98,
    "journal": 0.92,
    "drug": 0.88,
    "sop": 0.82,
}

_WORD_RE = re.compile(r"[a-z0-9]+", re.I)


def _tokens(text: str) -> set[str]:
    return {m.group(0).lower() for m in _WORD_RE.finditer(text)}


def _keyword_overlap_score(query: str, chunk_text: str) -> float:
    q = _tokens(query)
    c = _tokens(chunk_text)
    if not q or not c:
        return 0.0
    inter = len(q & c)
    return inter / max(len(q), 1)


def _quality_weight(meta: dict[str, Any] | None) -> float:
    if not meta:
        return 0.85
    t = meta.get("type")
    if isinstance(t, str):
        return _SOURCE_QUALITY_WEIGHT.get(t.lower(), 0.85)
    return 0.85


def _length_confidence(text: str) -> float:
    n = len(text.strip())
    if n < 120:
        return 0.82
    if n < 240:
        return 0.92
    return 1.0


def _fingerprint(text: str) -> str:
    normalized = " ".join(text.split()).lower()[:480]
    return hashlib.sha256(normalized.encode("utf-8", errors="ignore")).hexdigest()


def _dedupe_hits(hits: list[RetrievalHit]) -> list[RetrievalHit]:
    by_fp: dict[str, RetrievalHit] = {}
    for h in sorted(hits, key=lambda x: x.score, reverse=True):
        fp = _fingerprint(h.text)
        if fp in by_fp:
            continue
        by_fp[fp] = h
    return list(by_fp.values())


class RetrievalService:
    def __init__(
        self,
        *,
        settings: Settings | None = None,
        vector_store: VectorStore | None = None,
        embedder: EmbeddingProvider | None = None,
    ):
        self._settings = settings or get_settings()
        self._store = vector_store
        self._embedder = embedder
        self._cache: LRUCache[str, RetrievalResponse] | None = None
        if self._settings.query_cache_enabled and self._settings.query_cache_max_entries > 0:
            self._cache = LRUCache(maxsize=self._settings.query_cache_max_entries)

    def _store_lazy(self) -> VectorStore:
        if self._store is None:
            from app.vectorstore.factory import get_vector_store

            self._store = get_vector_store(self._settings)
        return self._store

    def _embedder_lazy(self) -> EmbeddingProvider:
        if self._embedder is None:
            from app.embeddings.openai_provider import OpenAIEmbeddingProvider

            self._embedder = OpenAIEmbeddingProvider(self._settings)
        return self._embedder

    def _cache_key(
        self,
        query: str,
        species: str | None,
        doc_type: str | None,
        topic: str | None,
        top_k: int,
    ) -> str:
        parts = [query.strip().lower(), species or "", doc_type or "", topic or "", str(top_k)]
        return "|".join(parts)

    def retrieve_context(
        self,
        query: str,
        *,
        species: str | None = None,
        document_type: str | None = None,
        topic: str | None = None,
        top_k: int | None = None,
        hybrid_alpha: float = 0.82,
    ) -> RetrievalResponse:
        k = top_k or self._settings.retrieval_top_k
        k = max(5, min(k, 50))

        if self._cache is not None:
            ck = self._cache_key(query, species, document_type, topic, k)
            cached = self._cache.get(ck)
            if cached is not None:
                return cached

        vec = self._embedder_lazy().embed_texts([query])[0]
        fetch_k = min(50, max(k * 3, k + 8))

        fl: dict[str, str | None] = {}
        if species:
            fl["species"] = species
        if document_type:
            fl["type"] = document_type
        if topic:
            fl["topic"] = topic

        raw: list[VectorQueryHit] = self._store_lazy().query_similar(
            vec,
            top_k=fetch_k,
            filters=fl or None,
        )

        hits: list[RetrievalHit] = []
        for row in raw:
            base = float(row.get("score", 0.0))
            md = dict(row.get("metadata") or {})
            wq = _quality_weight(md) if self._settings.source_quality_enabled else 1.0
            wc = _length_confidence(row.get("text", ""))
            kw = _keyword_overlap_score(query, row.get("text", ""))
            combined = (hybrid_alpha * base + (1.0 - hybrid_alpha) * kw) * wq * wc
            hits.append(
                RetrievalHit(
                    text=row.get("text", ""),
                    source=row.get("source", ""),
                    score=combined,
                    metadata={**md, "vector_score": base, "keyword_overlap": kw},
                )
            )

        hits.sort(key=lambda h: h.score, reverse=True)

        if self._settings.rerank_enabled and hits:
            head_n = min(24, len(hits))
            head = hits[:head_n]
            tail = hits[head_n:]
            head = rerank_with_cross_encoder(
                query,
                head,
                model_name=self._settings.rerank_model,
                top_k=head_n,
            )
            hits = head + tail

        hits = _dedupe_hits(hits)
        hits = hits[:k]

        resp = RetrievalResponse(query=query, results=hits)
        if self._cache is not None:
            self._cache[self._cache_key(query, species, document_type, topic, k)] = resp
        return resp
