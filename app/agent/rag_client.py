from __future__ import annotations

from typing import Any

from app.api.deps import retrieval_dep


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
