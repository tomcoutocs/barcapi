from __future__ import annotations

from app.models import RetrievalHit


def rerank_with_cross_encoder(
    query: str,
    hits: list[RetrievalHit],
    *,
    model_name: str,
    top_k: int,
) -> list[RetrievalHit]:
    try:
        from sentence_transformers import CrossEncoder
    except ImportError:
        return hits[:top_k]

    if not hits:
        return []
    model = CrossEncoder(model_name)
    pairs = [(query, h.text) for h in hits]
    scores = model.predict(pairs)
    reranked: list[RetrievalHit] = []
    for h, s in zip(hits, scores, strict=True):
        md = {**h.metadata, "cross_encoder_score": float(s)}
        reranked.append(
            RetrievalHit(
                text=h.text,
                source=h.source,
                score=float(s),
                metadata=md,
            )
        )
    reranked.sort(key=lambda x: x.score, reverse=True)
    return reranked[:top_k]
