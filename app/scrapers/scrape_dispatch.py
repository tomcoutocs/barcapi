from __future__ import annotations

import logging
from urllib.parse import urlparse

from app.scrapers.avma import scrape_avma, scrape_html_resource
from app.scrapers.guidelines import scrape_guideline_pdf
from app.scrapers.merck import scrape_merck

logger = logging.getLogger(__name__)


def scrape_one(url: str, *, delay_s: float = 1.0) -> dict:
    parsed = urlparse(url)
    host = (parsed.netloc or "").lower()
    path = (parsed.path or "").lower()
    is_pdf = path.endswith(".pdf") or url.lower().split("?")[0].endswith(".pdf")

    if "merckvetmanual.com" in host:
        return scrape_merck(url, delay_s=delay_s)
    if "avma.org" in host:
        if is_pdf:
            return scrape_guideline_pdf(url, "avma", delay_s=delay_s)
        return scrape_avma(url, delay_s=delay_s)
    if "aaha.org" in host:
        if is_pdf:
            return scrape_guideline_pdf(url, "aaha", delay_s=delay_s)
        return scrape_html_resource(url, "aaha", delay_s=delay_s)
    if "wsava.org" in host:
        if is_pdf:
            return scrape_guideline_pdf(url, "wsava", delay_s=delay_s)
        return scrape_html_resource(url, "wsava", delay_s=delay_s)

    raise ValueError(f"No scraper registered for URL host={host!r} url={url!r}")


def run_all_scrapers(urls: list[str], *, delay_s: float = 1.0) -> list[dict]:
    out: list[dict] = []
    for u in urls:
        u = u.strip()
        if not u:
            continue
        try:
            out.append(scrape_one(u, delay_s=delay_s))
        except Exception:
            logger.exception("scrape_one failed url=%s", u)
    return out
