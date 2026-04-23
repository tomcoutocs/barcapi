from __future__ import annotations

import logging
import time
from typing import Any

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

DEFAULT_HEADERS = {
    "User-Agent": (
        "VetRAGEducationalBot/1.0 (+https://example.invalid; respect robots.txt; contact: admin@localhost)"
    ),
    "Accept": "text/html,application/xhtml+xml,application/pdf;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

_SESSION: requests.Session | None = None


def _session() -> requests.Session:
    global _SESSION
    if _SESSION is None:
        _SESSION = requests.Session()
        _SESSION.headers.update(DEFAULT_HEADERS)
    return _SESSION


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
def fetch_url(url: str, *, timeout: float = 45.0) -> tuple[bytes, str | None]:
    """Return (body, content_type). Raises on hard failure after retries."""
    resp = _session().get(url, timeout=timeout)
    resp.raise_for_status()
    ctype = resp.headers.get("Content-Type", "").split(";")[0].strip().lower() or None
    return resp.content, ctype


def fetch_url_safe(url: str, *, delay_s: float = 0.0) -> tuple[bytes, str | None] | None:
    if delay_s > 0:
        time.sleep(delay_s)
    try:
        return fetch_url(url)
    except Exception:
        logger.exception("fetch failed url=%s", url)
        return None
