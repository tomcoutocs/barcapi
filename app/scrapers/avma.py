from __future__ import annotations

from app.scrapers.html_extract import extract_main_text
from app.scrapers.http_client import fetch_url_safe


def scrape_html_resource(url: str, source_type: str, *, delay_s: float = 1.0) -> dict:
    got = fetch_url_safe(url, delay_s=delay_s)
    if not got:
        raise RuntimeError(f"HTML fetch failed ({source_type}): {url}")
    body, ctype = got
    if ctype and "pdf" in ctype:
        raise ValueError(f"URL returned PDF; use scrape_guideline_pdf instead: {url}")
    html = body.decode("utf-8", errors="replace")
    title, text = extract_main_text(html, url=url)
    return {
        "url": url,
        "title": title or f"{source_type.upper()} resource",
        "raw_text": text,
        "source_type": source_type,
    }


def scrape_avma(url: str, *, delay_s: float = 1.0) -> dict:
    return scrape_html_resource(url, "avma", delay_s=delay_s)
