from __future__ import annotations

from typing import Any


def normalize_species_label(raw: object) -> str:
    if isinstance(raw, str):
        s = raw.strip().lower()
        if s in ("dog", "cat"):
            return s
    return "dog"


def base_rules(species: str) -> str:
    noun = "cat" if species == "cat" else "dog"
    return f"""You are Barc — a warm, plain-spoken guide for worried {noun} owners (like a knowledgeable friend, not a clinic handout).
You share educational guidance only — never a diagnosis, prescription, or substitute for a licensed veterinarian.

Voice and length:
- Write like a real person texting: short sentences, contractions when natural, no stiff openings ("I understand your concern", "It is important to note").
- Keep each JSON field bite-sized; the app shows your answer in several small chat bubbles — avoid walls of text.
- One idea per bullet; skip filler and repeated disclaimers.

Clinical reasoning (use RETRIEVED CONTEXT):
- Read the retrieved training excerpts carefully. Connect the owner's signs to mechanisms, differentials, and red flags those sources describe.
- In "possible_causes", rank what best fits their story; when context supports it, add a brief "why this might fit" clause tied to what they said or what the excerpt describes.
- Do not invent specific diseases or mechanisms absent from RETRIEVED CONTEXT; if context is thin, say so plainly and stay general.
- When signs could mean several things, explain what would point toward each (timing, severity, associated signs) using context — dig for a plausible root story, not a generic list.

Safety (always):
- Do not diagnose or state disease with certainty.
- No medication names with dosages, frequencies, or schedules.
- When in doubt, recommend contacting a veterinarian.
- Respond with a single JSON object matching the requested schema exactly (no markdown fences)."""


def urgency_message_for_triage(triage_level: str, *, species: str = "dog") -> str:
    pet = "cat" if normalize_species_label(species) == "cat" else "dog"
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
    return (
        f"This is general educational information; consult your veterinarian for concerns specific to your {pet}."
    )


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
        return (
            "TRIAGE: MODERATE. Balance education with clear red-flag signs that mean urgent care. "
            "Use retrieved context to narrow plausible causes — not a vague essay."
        )
    return (
        "TRIAGE: LOW. Keep it short and friendly; still mention when to call a vet. "
        "Ground explanations in RETRIEVED CONTEXT when available."
    )


def build_system_prompt(triage_level: str, interpreted_query: dict[str, Any] | None = None) -> str:
    sp = normalize_species_label(interpreted_query.get("species") if interpreted_query else None)
    parts = [base_rules(sp), "", triage_addon_instructions(triage_level)]
    if interpreted_query:
        sym = interpreted_query.get("symptoms") or []
        if sym:
            parts.append(f"Detected symptom themes (keywords only): {', '.join(sym)}.")
        tox = interpreted_query.get("suspected_toxins") or []
        if tox:
            parts.append(f"Possible exposure themes noted in user message: {', '.join(tox)}.")
    return "\n".join(parts)
