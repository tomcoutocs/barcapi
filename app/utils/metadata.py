from __future__ import annotations

import re
from typing import Literal

SourceIngestType = Literal["manual", "guideline", "org"]

_AUTHORITY_BY_SOURCE = {
    "merck": 0.95,
    "aaha": 1.0,
    "wsava": 1.0,
    "avma": 0.9,
}

_TYPE_BY_SOURCE: dict[str, SourceIngestType] = {
    "merck": "manual",
    "aaha": "guideline",
    "wsava": "guideline",
    "avma": "org",
}

_TOPIC_RULES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\b(vomit|vomiting|diarr|gastro|intestinal|nausea)\b", re.I), "gastroenterology"),
    (re.compile(r"\b(itch|prurit|dermat|skin rash|alopecia|mite)\b", re.I), "dermatology"),
    (re.compile(r"\b(seizure|epilep|neuro|ataxi|tremor)\b", re.I), "neurology"),
    (re.compile(r"\b(toxin|poison|chocolate|xylitol|rodenticid|ethylene glycol)\b", re.I), "toxicology"),
    (re.compile(r"\b(bloat|gdv|gastric dilat)\b", re.I), "emergency"),
    (re.compile(r"\b(cardi|heart murmur|arrhythm)\b", re.I), "cardiology"),
    (re.compile(r"\b(urinar|renal|kidney|uti)\b", re.I), "nephrology_urology"),
]

_URGENCY_EMERGENCY = re.compile(
    r"\b(seizure|collapse|cyanos|bloat|gdv|poison|toxin|"
    r"anaphyl|respiratory distress|unable to stand|"
    r"acute abdomen|hemorrhage)\b",
    re.I,
)
_URGENCY_HIGH = re.compile(
    r"\b(persistent|protracted|severe dehydration|blood in vomit|melena|"
    r"not eating for|letharg|weakness)\b",
    re.I,
)


def infer_topic(text: str) -> str:
    low = text.lower()
    for pat, topic in _TOPIC_RULES:
        if pat.search(low):
            return topic
    return "general"


def infer_urgency(text: str) -> str:
    low = text.lower()
    if _URGENCY_EMERGENCY.search(low):
        return "emergency"
    if _URGENCY_HIGH.search(low):
        return "high"
    if re.search(r"\b(chronic|long-?standing|over weeks|monitor closely)\b", low):
        return "medium"
    return "low"


def enrich_metadata(
    text: str,
    source: str,
    title: str,
    *,
    source_type: str,
) -> dict:
    """
    Build the ingestion metadata object (species fixed to dog per pipeline spec).
    """
    st = source_type.lower().strip()
    topic = infer_topic(text)
    urgency = infer_urgency(text)
    authority = _AUTHORITY_BY_SOURCE.get(st, 0.85)
    doc_type = _TYPE_BY_SOURCE.get(st, "manual")

    return {
        "source": source,
        "title": title.strip() or source,
        "type": doc_type,
        "species": "dog",
        "topic": topic,
        "urgency": urgency,
        "authority_weight": authority,
    }
