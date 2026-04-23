from __future__ import annotations

import json
from typing import Any

from openai import OpenAI

from app.agent.rag_client import retrieve_context
from app.agent.safety import build_system_prompt, urgency_message_for_triage
from app.config import get_settings

_JSON_INSTRUCTION = """
Return a single JSON object with exactly these keys:
- "triage_level": one of "emergency", "high", "moderate", "low" (must match the triage we provide)
- "summary": 2-4 short sentences, plain language
- "possible_causes": array of short bullet strings (educational possibilities, not diagnoses)
- "what_to_monitor": array of short bullet strings (signs that should prompt vet contact)
- "recommended_action": array of short bullet strings (safe, practical steps; vet visit when appropriate)
- "urgency_message": one string; must align with triage (use the URGENCY_HINT we provide as a guide)

Use short bullets. No markdown. No medication dosages.
"""


def _context_block(chunks: list[dict[str, Any]], max_chars: int = 12000) -> str:
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
    context: list[dict[str, Any]] | None = None,
) -> str:
    """
    Call OpenAI with RAG context and safety-first system prompt.
    If ``context`` is None, retrieves via ``retrieve_context(interpreted_query['normalized_query'])``.
    Returns raw JSON string from the model.
    """
    settings = get_settings()
    if not settings.openai_api_key:
        raise ValueError("OPENAI_API_KEY is required for chat responses")

    if context is None:
        context = retrieve_context(interpreted_query.get("normalized_query") or user_input, species="dog")

    client = OpenAI(api_key=settings.openai_api_key)
    system = build_system_prompt(triage_level, interpreted_query)
    urgency_hint = urgency_message_for_triage(triage_level)
    user_payload = {
        "user_message": user_input,
        "interpreted": interpreted_query,
        "triage_level": triage_level,
        "URGENCY_HINT": urgency_hint,
        "RETRIEVED_CONTEXT": _context_block(context),
        "output_instructions": _JSON_INSTRUCTION,
    }

    resp = client.chat.completions.create(
        model=settings.openai_chat_model,
        temperature=0.35,
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
                "urgency_message": urgency_hint,
            }
        )
    return choice
