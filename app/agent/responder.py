from __future__ import annotations

import json
from typing import Any

from openai import OpenAI

from app.agent.rag_client import retrieve_context_deep
from app.agent.safety import (
    build_system_prompt,
    normalize_species_label,
    urgency_message_for_triage,
)
from app.agent.triage import should_clarify_before_detail
from app.config import get_settings

_JSON_INSTRUCTION_BASE = """
Return a single JSON object with exactly these keys:
- "triage_level": one of "emergency", "high", "moderate", "low" (must match the triage we provide)
- "summary": 1-2 short sentences max, conversational; lead with empathy then the key takeaway
- "possible_causes": array of short bullets (educational possibilities, not diagnoses). When RETRIEVED CONTEXT supports it, each bullet should pair a plausible cause with why it might fit this case (sign pattern, timing, species). Order most likely first; omit guesses the context does not support.
- "what_to_monitor": array of short bullets (specific signs that should prompt vet contact soon)
- "recommended_action": array of short bullets (safe, practical steps; vet visit when appropriate)
- "urgency_message": one short string; must align with triage (use the URGENCY_HINT we provide as a guide)

Use short bullets (one line each). No markdown. No medication dosages. Prefer 2-4 items per list unless emergency detail is essential.
"""

_JSON_HINT_SUFFIX = """
If PREFERENCE_HINTS is non-empty, it may list two sections: thumbs-down (patterns to avoid) and thumbs-up (reinforcement of good tone/structure).
Avoid repeating mistakes from the downvote list; prefer qualities from the upvote list when they align with RETRIEVED_CONTEXT and safety. Safety and sources always override stylistic preference.
"""


def _json_instruction(*, include_feedback_hints: bool) -> str:
    return _JSON_INSTRUCTION_BASE + (_JSON_HINT_SUFFIX if include_feedback_hints else "")


_CLARIFICATION_FIRST_SUFFIX = """

CLARIFICATION_FIRST is active: the clinical picture is thin or key details are missing.
- Conversation may include CONVERSATION (prior turns): read it. Do NOT repeat questions the owner already answered; acknowledge what they shared in one clause, then move on.
- Lead the "summary" with one warm, brief line (casual tone), then ask EXACTLY ONE precise, answerable follow-up question in the SAME short paragraph. Wait for their reply—you do NOT have their next replies yet.
- Do NOT bundle multiple questions, numbered lists of questions, or "first / second / third" question drills. Maximum one question mark that seeks new information total in summary + bullets.
- Keep "possible_causes" empty or one very broad phrase only if harmless; omit detailed differentials until you have specifics.
- Keep "what_to_monitor" to 0-3 universal red-flag signs for this species (optional).
- Keep "recommended_action" to SAFE general steps ONLY (hydration, rest, observe, when to contact a vet)—no question bullets. The single clarification question MUST live only in "summary".
- Still honor triage: if something could be urgent, say so plainly in the summary and urgency_message.
"""


def _context_block(chunks: list[dict[str, Any]], max_chars: int = 14000) -> str:
    lines: list[str] = []
    n = 0
    for i, c in enumerate(chunks, 1):
        meta = c.get("metadata") or {}
        src = meta.get("source_label") or c.get("source", "")
        piece = f"[{i}] (source: {src})\n{c.get('text', '')}\n"
        if n + len(piece) > max_chars:
            break
        lines.append(piece)
        n += len(piece)
    return "\n".join(lines) if lines else "(No retrieved context; answer conservatively and recommend vet for clinical concerns.)"


def generate_response(
    user_input: str,
    triage_level: str,
    interpreted_query: dict[str, Any],
    *,
    preference_hints: str | None = None,
    context: list[dict[str, Any]] | None = None,
    conversation_plain: str | None = None,
) -> str:
    """
    Call OpenAI with RAG context and safety-first system prompt.
    If ``context`` is None, retrieves via multi-query ``retrieve_context_deep(interpreted_query)``.
    Returns raw JSON string from the model.
    """
    settings = get_settings()
    if not settings.openai_api_key:
        raise ValueError("OPENAI_API_KEY is required for chat responses")

    sp = normalize_species_label(interpreted_query.get("species"))

    if context is None:
        context = retrieve_context_deep(interpreted_query, species=sp)

    client = OpenAI(api_key=settings.openai_api_key)
    system = build_system_prompt(triage_level, interpreted_query)
    urgency_hint = urgency_message_for_triage(triage_level, species=sp)
    hints = (preference_hints or "").strip()
    clarify_first = should_clarify_before_detail(interpreted_query, triage_level)
    output_instructions = _json_instruction(include_feedback_hints=bool(hints))
    if clarify_first:
        output_instructions += _CLARIFICATION_FIRST_SUFFIX

    conv = (conversation_plain or "").strip()
    user_payload = {
        "user_message": user_input,
        "interpreted": interpreted_query,
        "triage_level": triage_level,
        "CLARIFICATION_FIRST": clarify_first,
        "URGENCY_HINT": urgency_hint,
        "RETRIEVED_CONTEXT": _context_block(context),
        "CONVERSATION": conv if conv else None,
        "PREFERENCE_HINTS": hints,
        "output_instructions": output_instructions,
    }

    resp = client.chat.completions.create(
        model=settings.openai_chat_model,
        temperature=0.42,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ],
    )
    choice = resp.choices[0].message.content
    if not choice:
        return json.dumps(
            {
                "triage_level": triage_level,
                "summary": "I could not generate a full response. Please contact your veterinarian.",
                "possible_causes": [],
                "what_to_monitor": [],
                "recommended_action": ["Contact your veterinarian with this message."],
                "urgency_message": urgency_message_for_triage(triage_level, species=sp),
            }
        )
    return choice
