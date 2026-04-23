from __future__ import annotations

import hashlib
import re


def word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def content_fingerprint(text: str) -> str:
    normalized = " ".join(text.lower().split())
    return hashlib.sha256(normalized.encode("utf-8", errors="ignore")).hexdigest()


class ContentDeduper:
    """Skip duplicate bodies and very short documents (word count)."""

    def __init__(self, *, min_words: int = 300) -> None:
        self._min_words = min_words
        self._seen: set[str] = set()

    def accept(self, text: str) -> bool:
        if word_count(text) < self._min_words:
            return False
        fp = content_fingerprint(text)
        if fp in self._seen:
            return False
        self._seen.add(fp)
        return True
