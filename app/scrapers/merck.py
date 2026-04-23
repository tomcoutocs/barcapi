from __future__ import annotations

import logging

from app.scrapers.html_extract import extract_main_text
from app.scrapers.http_client import fetch_url_safe

logger = logging.getLogger(__name__)


def scrape_merck(url: str, *, delay_s: float = 1.0) -> dict:
    """
    Scrape a Merck Veterinary Manual HTML page (prefer dog-owner or canine clinical URLs).
    """
    got = fetch_url_safe(url, delay_s=delay_s)
    if not got:
        raise RuntimeError(f"Merck fetch failed: {url}")
    body, ctype = got
    if ctype and "pdf" in ctype:
        raise ValueError(f"Expected HTML from Merck, got {ctype}: {url}")
    html = body.decode("utf-8", errors="replace")
    title, text = extract_main_text(html, url=url)
    if "/dog-owners/" not in url.lower() and "dog" not in url.lower():
        logger.warning("URL may not be dog-focused: %s", url)
    return {
        "url": url,
        "title": title or "Merck Veterinary Manual",
        "raw_text": text,
        "source_type": "merck",
    }
