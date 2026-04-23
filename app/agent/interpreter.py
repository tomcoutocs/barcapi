from __future__ import annotations

import re
from typing import Any

_SYMPTOM_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bvomit(?:ing|ed)?\b", re.I), "vomiting"),
    (re.compile(r"\bthrow(?:ing)?\s*up\b", re.I), "vomiting"),
    (re.compile(r"\bletharg(?:y|ic)\b", re.I), "lethargy"),
    (re.compile(r"\bweak(?:ness)?\b", re.I), "weakness"),
    (re.compile(r"\bitch(?:y|ing)\b", re.I), "itching"),
    (re.compile(r"\bscratch(?:ing)?\b", re.I), "scratching"),
    (re.compile(r"\bdiarrh(?:oea|ea)\b", re.I), "diarrhea"),
    (re.compile(r"\bloose\s*stool\b", re.I), "diarrhea"),
    (re.compile(r"\bseizure\b", re.I), "seizure"),
    (re.compile(r"\bcollaps(?:e|ed|ing)\b", re.I), "collapse"),
    (re.compile(r"\bcan'?t\s+stand\b", re.I), "unable_to_stand"),
    (re.compile(r"\bwon'?t\s+(?:eat|drink)\b", re.I), "anorexia"),
    (re.compile(r"\bnot\s+eating\b", re.I), "anorexia"),
    (re.compile(r"\b(?:difficulty|trouble)\s+breath", re.I), "difficulty_breathing"),
    (re.compile(r"\b.labou?red\s+breath", re.I), "difficulty_breathing"),
    (re.compile(r"\bgas\s*p(?:ing)?\b", re.I), "difficulty_breathing"),
    (re.compile(r"\bbloat(?:ed)?\b", re.I), "bloat"),
    (re.compile(r"\bgdv\b", re.I), "bloat"),
    (re.compile(r"\bpant(?:ing)?\s*(?:a\s*lot|excessively)?\b", re.I), "panting"),
]

_TOXIN_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bchocolate\b", re.I), "chocolate"),
    (re.compile(r"\bxylitol\b", re.I), "xylitol"),
    (re.compile(r"\bgrape[s]?\b|\braisin[s]?\b", re.I), "grapes_raisins"),
    (re.compile(r"\bonion[s]?\b|\bgarlic\b", re.I), "onion_garlic"),
    (re.compile(r"\bantifreeze\b|\bethylene\s+glycol\b", re.I), "antifreeze"),
    (re.compile(r"\brat\s*poison\b|\bbrodifacoum\b", re.I), "rodenticide"),
    (re.compile(r"\bmarijuana\b|\bcannabis\b|\bthc\b", re.I), "cannabis"),
]

_SEVERITY_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bcollaps(?:e|ed|ing)\b", re.I), "collapsed"),
    (re.compile(r"\bunresponsive\b", re.I), "unresponsive"),
    (re.compile(r"\bwon'?t\s+move\b", re.I), "won't_move"),
    (re.compile(r"\bnon[- ]?responsive\b", re.I), "nonresponsive"),
    (re.compile(r"\bblue\s+gums\b|\bcyanot", re.I), "cyanosis"),
    (re.compile(r"\bcan'?t\s+breathe\b", re.I), "severe_respiratory_distress"),
]

_DURATION_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\b(\d+)\s*(?:day|days)\b", re.I), r"\1 days"),
    (re.compile(r"\b(\d+)\s*(?:hour|hours|hr|hrs)\b", re.I), r"\1 hours"),
    (re.compile(r"\b(\d+)\s*(?:week|weeks)\b", re.I), r"\1 weeks"),
    (re.compile(r"\bjust\s+now\b|\btoday\b|\bright\s+now\b|\bthis\s+morning\b", re.I), "acute_same_day"),
    (re.compile(r"\bsince\s+yesterday\b", re.I), "~1 day"),
]


def interpret_query(user_input: str) -> dict[str, Any]:
    text = user_input.strip()
    low = text.lower()

    symptoms: list[str] = []
    for pat, label in _SYMPTOM_PATTERNS:
        if pat.search(text) and label not in symptoms:
            symptoms.append(label)

    suspected_toxins: list[str] = []
    for pat, label in _TOXIN_PATTERNS:
        if pat.search(text) and label not in suspected_toxins:
            suspected_toxins.append(label)

    severity_flags: list[str] = []
    for pat, label in _SEVERITY_PATTERNS:
        if pat.search(text) and label not in severity_flags:
            severity_flags.append(label)

    duration: str | None = None
    for pat, fmt in _DURATION_PATTERNS:
        m = pat.search(text)
        if m:
            if "(" in fmt and m.groups():
                duration = fmt.replace(r"\1", m.group(1))
            else:
                duration = fmt if not m.groups() else m.group(0)
            break

    normalized = text
    if symptoms:
        normalized = f"Dog: {', '.join(symptoms)}. " + text
    if not symptoms and not suspected_toxins and len(text) < 80:
        normalized = f"General dog owner question: {text}"

    return {
        "normalized_query": normalized,
        "symptoms": symptoms,
        "duration": duration,
        "severity_flags": severity_flags,
        "suspected_toxins": suspected_toxins,
    }
