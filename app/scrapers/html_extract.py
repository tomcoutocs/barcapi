from __future__ import annotations

import re
from typing import Any

from bs4 import BeautifulSoup

_REMOVE_SELECTORS = (
    "nav",
    "footer",
    "header",
    "aside",
    "script",
    "style",
    "noscript",
    "form",
    ".advertisement",
    ".ad",
    "[role='navigation']",
    "[aria-label*='cookie' i]",
)


def extract_main_text(html: str, *, url: str = "") -> tuple[str, str]:
    """
    Strip chrome and return (title, plain_text with headings preserved as lines).
    """
    soup = BeautifulSoup(html, "lxml")
    title_el = soup.find("title")
    title = title_el.get_text(strip=True) if title_el else ""

    for sel in _REMOVE_SELECTORS:
        for node in soup.select(sel):
            node.decompose()

    main: Any = None
    for sel in (
        "article",
        "main",
        "[role='main']",
        "#main",
        ".main-content",
        ".content",
        "#content",
    ):
        main = soup.select_one(sel)
        if main:
            break
    root = main if main else soup.body or soup

    parts: list[str] = []
    for el in root.find_all(["h1", "h2", "h3", "h4", "p", "li"]):
        t = el.get_text(" ", strip=True)
        if not t or len(t) < 2:
            continue
        if el.name in {"h1", "h2", "h3", "h4"}:
            parts.append(f"\n{t}\n")
        else:
            parts.append(t)

    text = "\n".join(parts)
    text = re.sub(r"\n{3,}", "\n\n", text)
    if not text.strip() and root:
        text = root.get_text("\n", strip=True)
    if not title and url:
        title = url
    return title, text.strip()
