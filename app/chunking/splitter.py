from __future__ import annotations

import re
import uuid
from typing import Any

import tiktoken

from app.models import Chunk, NormalizedDocument

_DEFAULT_ENCODING = "cl100k_base"
_MIN_TOKENS = 500
_MAX_TOKENS = 800
_OVERLAP_MIN = 100
_OVERLAP_MAX = 150


def _encoder():
    return tiktoken.get_encoding(_DEFAULT_ENCODING)


def count_tokens(text: str) -> int:
    return len(_encoder().encode(text))


_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n{2,}")


def _split_sentences(text: str) -> list[str]:
    parts = [p.strip() for p in _SENT_SPLIT.split(text) if p and p.strip()]
    if not parts:
        return [text.strip()] if text.strip() else []
    return parts


def _split_oversized_sentence(sentence: str, max_tokens: int) -> list[str]:
    enc = _encoder()
    ids = enc.encode(sentence)
    if len(ids) <= max_tokens:
        return [sentence.strip()] if sentence.strip() else []
    out: list[str] = []
    start = 0
    while start < len(ids):
        end = min(start + max_tokens, len(ids))
        piece = enc.decode(ids[start:end]).strip()
        if piece:
            out.append(piece)
        start = end
    return out


def _sentences_for_document(text: str, max_tokens: int) -> list[str]:
    result: list[str] = []
    for sent in _split_sentences(text):
        if count_tokens(sent) > max_tokens:
            result.extend(_split_oversized_sentence(sent, max_tokens))
        else:
            result.append(sent)
    return result


def _next_overlap_start(
    sentences: list[str],
    chunk_start: int,
    chunk_end: int,
    overlap_target: int,
) -> int:
    """Index where the next chunk should begin to include ~overlap_target tokens from the prior chunk."""
    overlap_target = max(_OVERLAP_MIN, min(overlap_target, _OVERLAP_MAX))
    if chunk_end <= chunk_start + 1:
        return chunk_end
    acc = 0
    k = chunk_end - 1
    while k >= chunk_start:
        acc += count_tokens(sentences[k])
        if k < chunk_end - 1:
            acc += 1
        if acc >= overlap_target:
            return k
        k -= 1
    return max(chunk_start, chunk_end - 1)


def chunk_document(
    document: NormalizedDocument,
    *,
    min_tokens: int = _MIN_TOKENS,
    max_tokens: int = _MAX_TOKENS,
    overlap_tokens: int = 125,
) -> list[Chunk]:
    overlap_tokens = max(_OVERLAP_MIN, min(overlap_tokens, _OVERLAP_MAX))
    sentences = _sentences_for_document(document.text, max_tokens)
    if not sentences:
        return []

    meta: dict[str, Any] = {
        **document.metadata.model_dump(exclude_none=True),
    }

    chunks: list[Chunk] = []
    n = len(sentences)
    i = 0
    while i < n:
        j = i
        acc = 0
        while j < n:
            t = count_tokens(sentences[j]) + (1 if j > i else 0)
            if acc + t > max_tokens and j > i:
                break
            acc += t
            j += 1
        if j == i:
            j = i + 1
        text = " ".join(sentences[i:j]).strip()
        if text:
            chunks.append(
                Chunk(
                    chunk_id=str(uuid.uuid4()),
                    text=text,
                    source=document.document_id,
                    metadata={**meta, "source_label": document.source},
                )
            )
        if j >= n:
            break
        i = _next_overlap_start(sentences, i, j, overlap_tokens)

    return chunks
