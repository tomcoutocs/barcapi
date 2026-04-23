from __future__ import annotations

import re

_DISCLAIMER_PATTERNS = [
    re.compile(
        r"(?is)\bthe content of this site is subject to copyright\b.*",
    ),
    re.compile(r"(?is)\ball rights reserved\b.*"),
    re.compile(r"(?is)\bprivacy policy\b.*?\n{2,}", re.MULTILINE),
    re.compile(r"(?is)\bcookie policy\b.*?\n{2,}", re.MULTILINE),
    re.compile(r"(?is)\bskip to (main )?content\b"),
    re.compile(r"(?is)\baccept (all )?cookies\b"),
    re.compile(r"(?is)\bsubscribe to our newsletter\b.*"),
    re.compile(r"(?is)\bfollow us on\b.*"),
]

_NAV_BOILERPLATE = re.compile(
    r"(?im)^(home|search|menu|sign in|log in|contact us|about us)\s*$",
    re.MULTILINE,
)


def clean_text(text: str) -> str:
    if not text:
        return ""
    t = text.replace("\x00", " ")
    for pat in _DISCLAIMER_PATTERNS:
        t = pat.sub("\n", t)
    lines = []
    for line in t.splitlines():
        s = line.strip()
        if not s:
            lines.append("")
            continue
        if _NAV_BOILERPLATE.match(s) and len(s) < 40:
            continue
        lines.append(s)
    t = "\n".join(lines)
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()
