from __future__ import annotations

import logging
from collections import deque
from urllib.parse import urljoin, urlparse, urlunparse

from bs4 import BeautifulSoup

from app.scrapers.html_extract import extract_main_text
from app.scrapers.http_client import fetch_url_safe

logger = logging.getLogger(__name__)

HOST_MARKER = "merckvetmanual.com"


def _canonical_dog_owner_url(href: str, base: str) -> str | None:
    full = urljoin(base, href)
    p = urlparse(full)
    host = (p.netloc or "").lower()
    if HOST_MARKER not in host:
        return None
    path = p.path or "/"
    if not path.lower().startswith("/dog-owners"):
        return None
    path_norm = "/" + "/".join(x for x in path.split("/") if x)
    if not path_norm.lower().startswith("/dog-owners"):
        return None
    return urlunparse(("https", "www.merckvetmanual.com", path_norm, "", p.query, ""))


def _path_is_article(path: str) -> bool:
    parts = [x for x in (path or "").strip("/").split("/") if x]
    return len(parts) >= 3 and parts[0].lower() == "dog-owners"


def crawl_merck_dog_owner_articles(
    seeds: list[str] | None = None,
    *,
    max_articles: int = 200,
    max_visits: int = 4000,
    delay_s: float = 1.0,
) -> list[dict]:
    """
    Breadth-first crawl of merckvetmanual.com/dog-owners/*, extracting one HTML fetch per visit.
    Collects up to ``max_articles`` article pages (URL depth: /dog-owners/<section>/<page>/...).
    """
    if seeds is None:
        seeds = ["https://www.merckvetmanual.com/dog-owners"]

    seeds_n: list[str] = []
    for s in seeds:
        c = _canonical_dog_owner_url(s, s)
        if c:
            seeds_n.append(c)
        elif "/dog-owners" in s:
            seeds_n.append(s.split("#")[0].rstrip("/") or "https://www.merckvetmanual.com/dog-owners")
    if not seeds_n:
        seeds_n = ["https://www.merckvetmanual.com/dog-owners"]

    articles: list[dict] = []
    visited: set[str] = set()
    enqueued: set[str] = set()
    q: deque[str] = deque()
    for s in seeds_n:
        if s not in enqueued:
            enqueued.add(s)
            q.append(s)

    visits = 0
    while q and len(articles) < max_articles and visits < max_visits:
        url = q.popleft()
        if url in visited:
            continue
        visited.add(url)
        visits += 1
        if visits % 25 == 0:
            logger.info(
                "merck crawl: visits=%s articles=%s queue=%s",
                visits,
                len(articles),
                len(q),
            )

        got = fetch_url_safe(url, delay_s=delay_s)
        if not got:
            continue
        body, ctype = got
        if ctype and "pdf" in ctype:
            continue
        html = body.decode("utf-8", errors="replace")
        path = urlparse(url).path
        if _path_is_article(path) and len(articles) < max_articles:
            title, text = extract_main_text(html, url=url)
            if text.strip():
                articles.append(
                    {
                        "url": url,
                        "title": title or url,
                        "raw_text": text,
                        "source_type": "merck",
                    }
                )

        soup = BeautifulSoup(html, "lxml")
        for a in soup.find_all("a", href=True):
            child = _canonical_dog_owner_url(a["href"], url)
            if not child or child in visited:
                continue
            if child not in enqueued:
                enqueued.add(child)
                q.append(child)

    logger.info("merck crawl: done visits=%s unique_enqueued=%s articles=%s", visits, len(enqueued), len(articles))
    return articles
