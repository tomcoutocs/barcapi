from __future__ import annotations

import logging
from typing import Literal

from app.ingestion.loaders import load_pdf_bytes
from app.scrapers.http_client import fetch_url_safe

logger = logging.getLogger(__name__)

Org = Literal["aaha", "wsava", "avma"]


def scrape_guideline_pdf(url: str, organization: Org, *, delay_s: float = 1.0) -> dict:
    got = fetch_url_safe(url, delay_s=delay_s)
    if not got:
        raise RuntimeError(f"Guideline PDF fetch failed: {url}")
    body, ctype = got
    if ctype and "pdf" not in ctype and not url.lower().endswith(".pdf"):
        logger.warning("URL may not be a PDF (Content-Type=%s): %s", ctype, url)
    text = load_pdf_bytes(body)
    title = url.rsplit("/", 1)[-1].replace(".pdf", "").replace("-", " ").replace("_", " ")
    label = organization.upper()
    return {
        "url": url,
        "title": title or f"{label} guideline",
        "raw_text": text,
        "source_type": organization,
    }
