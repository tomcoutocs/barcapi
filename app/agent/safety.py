from __future__ import annotations

from typing import Any

BASE_RULES = """You are a veterinary assistant for dog owners.
You provide educational guidance only — not a diagnosis, prescription, or substitute for a licensed veterinarian.

Follow these rules strictly:
- Do not diagnose conditions or label disease with certainty.
- Do not provide medication names with dosages, frequencies, or schedules.
- Prioritize safety: when in doubt, recommend contacting a veterinarian.
- Base explanations only on the RETRIEVED CONTEXT provided; if context is thin, say so and stay general.
- Use clear, calm language appropriate for worried pet owners.
- Respond with a single JSON object matching the requested schema exactly (no markdown fences)."""


def urgency_message_for_triage(triage_level: str) -> str:
    t = triage_level.lower()
    if t == "emergency":
        return (
            "Seek immediate veterinary care or an emergency clinic now. "
            "If you cannot travel safely, call your nearest emergency vet for guidance."
        )
    if t == "high":
        return "Strongly recommend contacting a veterinarian today for an in-person evaluation."
    if t == "moderate":
        return "Monitor closely and contact your veterinarian if signs worsen or new symptoms appear."
    return "This is general educational information; consult your veterinarian for concerns specific to your dog."


def triage_addon_instructions(triage_level: str) -> str:
    t = triage_level.lower()
    if t == "emergency":
        return (
            "TRIAGE: EMERGENCY. Open with the urgency_message. "
            "Do not minimize risk. Do not suggest waiting overnight. "
            "Do not give home remedies for toxin ingestion or collapse."
        )
    if t == "high":
        return (
            "TRIAGE: HIGH. Emphasize same-day veterinary contact. "
            "Provide monitoring tips only as adjuncts, not replacements for exam."
        )
    if t == "moderate":
        return "TRIAGE: MODERATE. Balance education with clear red-flag signs that mean urgent care."
    return "TRIAGE: LOW. Keep answers concise and educational; still mention when to call a vet."


def build_system_prompt(triage_level: str, interpreted_query: dict[str, Any] | None = None) -> str:
    parts = [BASE_RULES, "", triage_addon_instructions(triage_level)]
    if interpreted_query:
        sym = interpreted_query.get("symptoms") or []
        if sym:
            parts.append(f"Detected symptom themes (keywords only): {', '.join(sym)}.")
        tox = interpreted_query.get("suspected_toxins") or []
        if tox:
            parts.append(f"Possible exposure themes noted in user message: {', '.join(tox)}.")
    return "\n".join(parts)
