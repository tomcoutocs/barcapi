from __future__ import annotations

import logging
from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.agent.evaluator import evaluate_flags, log_interaction
from app.agent.formatter import FormattedResponse, format_response
from app.agent.interpreter import interpret_query
from app.agent.responder import generate_response
from app.agent.safety import normalize_species_label, urgency_message_for_triage
from app.agent.triage import (
    classify_triage,
    generate_followup_questions,
    should_clarify_before_detail,
)

router = APIRouter(tags=["agent"])
logger = logging.getLogger(__name__)

_MAX_CONVERSATION_TURNS = 32


class ConversationTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field("", max_length=8000)


def _combined_user_turns(
    latest: str,
    *,
    history: list[ConversationTurn] | None,
) -> str:
    parts: list[str] = []
    if history:
        for turn in history:
            if turn.role == "user" and turn.content.strip():
                parts.append(turn.content.strip())
    msg = latest.strip()
    if msg:
        parts.append(msg)
    return "\n".join(parts)


def _conversation_transcript(history: list[ConversationTurn] | None) -> str | None:
    if not history:
        return None
    lines: list[str] = []
    for turn in history:
        blob = turn.content.strip()
        if not blob:
            continue
        label = "Owner" if turn.role == "user" else "Assistant"
        lines.append(f"{label}: {blob}")
    return "\n".join(lines) if lines else None


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=8000)
    species: str | None = Field(default=None, description='Retrieval filter: "dog" or "cat"; defaults to dog.')
    preference_hints: str | None = Field(
        default=None,
        max_length=6000,
        description="Optional user-specific steering from thumbs-down history (plain text).",
    )
    conversation_history: list[ConversationTurn] | None = Field(
        default=None,
        description="Prior turns in this chat (excluding the latest user message in `message`).",
    )


@router.post("/chat", response_model=FormattedResponse)
def chat(req: ChatRequest) -> FormattedResponse:
    species = normalize_species_label(req.species)
    hist = req.conversation_history
    if hist is not None and len(hist) > _MAX_CONVERSATION_TURNS:
        hist = hist[-_MAX_CONVERSATION_TURNS :]

    interpret_source = _combined_user_turns(req.message, history=hist)
    interpreted = interpret_query(interpret_source, species=species)
    triage = classify_triage(interpreted)
    followups = generate_followup_questions(interpreted)
    clarification_first = should_clarify_before_detail(interpreted, triage)
    pet_word = "cat" if species == "cat" else "dog"

    try:
        raw = generate_response(
            req.message,
            triage,
            interpreted,
            preference_hints=req.preference_hints,
            conversation_plain=_conversation_transcript(hist),
        )
        out = format_response(
            raw,
            triage,
            follow_up_questions=None if clarification_first else followups,
            species=species,
            clarification_first=clarification_first,
        )
    except Exception:
        logger.exception("chat generation failed")

        out = FormattedResponse(
            triage_level=triage,  # type: ignore[arg-type]
            summary=(
                "Something went wrong generating a detailed answer. Please contact your veterinarian, "
                f"especially if your {pet_word} seems unwell."
            ),
            possible_causes=[],
            what_to_monitor=[],
            recommended_action=followups
            + ["Contact your veterinarian or an emergency clinic if you are worried."],
            urgency_message=urgency_message_for_triage(triage, species=species),
        )

    flags = evaluate_flags(req.message, triage, interpreted)
    try:
        log_interaction(user_input=req.message, triage_level=triage, response=out, flags=flags)
    except OSError:
        logger.warning("could not write eval log")

    return out
