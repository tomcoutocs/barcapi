from __future__ import annotations

import re
from typing import Any

TriageLevel = str  # "emergency" | "high" | "moderate" | "low"

_EMERGENCY = re.compile(
    r"\b(seizure|collapse|collapsed|bloat|gdv|chocolate|xylitol|antifreeze|ethylene\s+glycol|"
    r"poison(?:ed)?|toxin|can'?t\s+breathe|gasping|cyanosis|blue\s+gums|"
    r"distended\s+abdomen|acute\s+abdomen)\b",
    re.I,
)

_GENERAL_INFO = re.compile(
    r"\b(how\s+often|what\s+is|when\s+should|best\s+food|vaccine|training|groom|normal\s+for)\b",
    re.I,
)


def _hours_from_duration(duration: str | None) -> float | None:
    if not duration:
        return None
    d = duration.lower()
    m = re.search(r"(\d+)\s*hours?", d)
    if m:
        return float(m.group(1))
    m = re.search(r"(\d+)\s*days?", d)
    if m:
        return float(m.group(1)) * 24.0
    m = re.search(r"(\d+)\s*weeks?", d)
    if m:
        return float(m.group(1)) * 24.0 * 7
    if "same_day" in d or "~1 day" in d:
        return 12.0
    return None


def classify_triage(interpreted_query: dict[str, Any]) -> TriageLevel:
    text = interpreted_query.get("normalized_query", "") or ""
    symptoms: list[str] = list(interpreted_query.get("symptoms") or [])
    toxins: list[str] = list(interpreted_query.get("suspected_toxins") or [])
    sev: list[str] = list(interpreted_query.get("severity_flags") or [])
    duration = interpreted_query.get("duration")

    if toxins or _EMERGENCY.search(text):
        return "emergency"
    if "seizure" in symptoms or "collapse" in symptoms or "bloat" in symptoms:
        return "emergency"
    if "difficulty_breathing" in symptoms:
        return "emergency"
    if any(s in sev for s in ("collapsed", "cyanosis", "severe_respiratory_distress", "unresponsive")):
        return "emergency"

    hrs = _hours_from_duration(duration)
    if hrs is not None and hrs >= 48 and symptoms:
        return "high"
    if len(symptoms) >= 2:
        return "high"
    if "anorexia" in symptoms and ("vomiting" in symptoms or hrs is not None and hrs >= 24):
        return "high"
    if "vomiting" in symptoms and hrs is not None and hrs >= 48:
        return "high"

    if symptoms and _GENERAL_INFO.search(text):
        return "moderate"

    if symptoms:
        return "moderate"

    if _GENERAL_INFO.search(text) or not symptoms:
        return "low"

    return "moderate"


def should_clarify_before_detail(interpreted_query: dict[str, Any], triage_level: str) -> bool:
    """
    True when the user's message is too thin to tailor guidance safely — prioritize
    targeted questions over a long generic essay (emergency/high still get urgent guidance).
    """
    if triage_level in ("emergency", "high"):
        return False

    user_text = (interpreted_query.get("user_text") or interpreted_query.get("normalized_query") or "").strip()
    if not user_text:
        return False
    user_lower = user_text.lower()

    if _GENERAL_INFO.search(user_lower):
        return False

    symptoms: list[str] = list(interpreted_query.get("symptoms") or [])
    toxins: list[str] = list(interpreted_query.get("suspected_toxins") or [])
    if toxins:
        return False

    if not symptoms:
        return True

    word_count = len(user_text.split())
    if interpreted_query.get("duration") is None and word_count < 28:
        return True

    return False


def generate_followup_questions(interpreted_query: dict[str, Any]) -> list[str]:
    """Ask only for gaps that change triage or guidance."""
    pet = "cat" if interpreted_query.get("species") == "cat" else "dog"
    young = "kitten" if pet == "cat" else "puppy"
    qs: list[str] = []
    if interpreted_query.get("duration") is None and interpreted_query.get("symptoms"):
        qs.append("How long have these signs been going on (hours or days)?")
    if interpreted_query.get("symptoms") and "weight" not in " ".join(
        interpreted_query.get("normalized_query", "").lower(),
    ):
        qs.append(
            f"What is your {pet}'s approximate weight and age ({young}, adult, senior)?",
        )
    if not interpreted_query.get("suspected_toxins") and interpreted_query.get("symptoms"):
        qs.append(
            f"Could your {pet} have eaten anything unusual (food, plants, medications, chemicals)?",
        )
    return qs[:4]
