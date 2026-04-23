"""Trusted-source scrapers for the veterinary ingestion pipeline."""

from app.scrapers.scrape_dispatch import run_all_scrapers

__all__ = ["run_all_scrapers"]
