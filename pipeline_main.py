"""
CLI runner: crawl/scrape trusted URLs → process_and_ingest (existing chunk/embed/vector path).

Usage (from vet-rag-api/, venv active):
  python pipeline_main.py

Bulk Merck dog-owner pages:
  MERCK_CRAWL_MAX_ARTICLES=500 python pipeline_main.py

Respect each site's Terms of Use and robots.txt; obtain licenses where required.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app.config import get_settings
from app.pipeline.process_and_ingest import process_and_ingest
from app.scrapers.merck_crawl import crawl_merck_dog_owner_articles
from app.scrapers.scrape_dispatch import run_all_scrapers

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

# Optional: extra non-Merck URLs (HTML or PDF) handled by scrape_dispatch.
ADDITIONAL_URLS: list[str] = [
    # "https://www.avma.org/...",
]

# BFS entry points for Merck dog-owner section (expand if needed).
MERCK_CRAWL_SEEDS = [
    "https://www.merckvetmanual.com/dog-owners",
]


def main() -> None:
    settings = get_settings()
    max_art = settings.merck_crawl_max_articles
    max_vis = settings.merck_crawl_max_visits
    delay = settings.merck_crawl_delay_s

    log = logging.getLogger(__name__)
    log.info(
        "Merck crawl limits: max_articles=%s max_visits=%s delay_s=%s",
        max_art,
        max_vis,
        delay,
    )

    merck_docs = crawl_merck_dog_owner_articles(
        MERCK_CRAWL_SEEDS,
        max_articles=max_art,
        max_visits=max_vis,
        delay_s=delay,
    )
    log.info("Collected %s Merck article(s) from crawl", len(merck_docs))

    extra_docs = run_all_scrapers(ADDITIONAL_URLS, delay_s=delay) if ADDITIONAL_URLS else []
    all_docs = merck_docs + extra_docs

    n = process_and_ingest(all_docs, min_words=300)
    log.info("Pipeline finished; ingested %s document(s) (after dedupe/min-words)", n)


if __name__ == "__main__":
    main()
