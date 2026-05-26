from __future__ import annotations

import hashlib
from typing import Any

from app.api.deps import retrieval_dep
from app.config import get_settings


def _text_fingerprint(text: str) -> str:
    normalized = " ".join(text.split()).lower()[:480]
    return hashlib.sha256(normalized.encode("utf-8", errors="ignore")).hexdigest()


def retrieve_context(
    query: str,
    *,
    species: str | None = "dog",
    top_k: int | None = None,
) -> list[dict[str, Any]]:
    """
    Thin adapter over the existing RetrievalService (does not reimplement RAG).

    Returns a list of dicts suitable for prompting: text, source, score, metadata.
    """
    svc = retrieval_dep()
    resp = svc.retrieve_context(query.strip(), species=species, top_k=top_k)
    return [h.model_dump() for h in resp.results]


def _retrieval_queries(interpreted_query: dict[str, Any], *, species: str) -> list[str]:
    """Build focused search queries so retrieval surfaces differential / mechanism chunks."""
    pet = "cat" if species == "cat" else "dog"
    queries: list[str] = []

    primary = (interpreted_query.get("normalized_query") or "").strip()
    if primary:
        queries.append(primary)

    symptoms: list[str] = list(interpreted_query.get("symptoms") or [])
    for sym in symptoms[:3]:
        label = sym.replace("_", " ")
        queries.append(f"{pet} {label} causes differential diagnosis veterinary")

    if len(symptoms) >= 2:
        combo = " ".join(s.replace("_", " ") for s in symptoms[:3])
        queries.append(f"{pet} {combo} common causes pathophysiology")

    toxins: list[str] = list(interpreted_query.get("suspected_toxins") or [])
    for tox in toxins[:2]:
        label = tox.replace("_", " ")
        queries.append(f"{pet} {label} toxicity clinical signs emergency")

    duration = interpreted_query.get("duration")
    if duration and symptoms:
        sym = symptoms[0].replace("_", " ")
        queries.append(f"{pet} {sym} duration {duration} prognosis monitoring")

    # Dedupe while preserving order
    seen: set[str] = set()
    out: list[str] = []
    for q in queries:
        key = q.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(q)
    return out[:5]


def retrieve_context_deep(
    interpreted_query: dict[str, Any],
    *,
    species: str | None = "dog",
    top_k: int | None = None,
) -> list[dict[str, Any]]:
    """
    Multi-query retrieval: primary message plus symptom/toxin-focused searches
    to pull training-doc chunks that explain mechanisms and differentials.
    """
    settings = get_settings()
    k = top_k or settings.retrieval_top_k
    k = max(5, min(k, 50))
    sp = (species or "dog").strip().lower()
    if sp not in ("dog", "cat"):
        sp = "dog"

    queries = _retrieval_queries(interpreted_query, species=sp)
    if not queries:
        return []

    per_query_k = max(4, min(12, (k // len(queries)) + 3))
    merged: list[dict[str, Any]] = []
    seen_fp: set[str] = set()

    for q in queries:
        for hit in retrieve_context(q, species=sp, top_k=per_query_k):
            fp = _text_fingerprint(hit.get("text", ""))
            if fp in seen_fp:
                continue
            seen_fp.add(fp)
            merged.append(hit)

    merged.sort(key=lambda h: float(h.get("score") or 0.0), reverse=True)
    return merged[:k]
