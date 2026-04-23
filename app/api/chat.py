from __future__ import annotations

import logging

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.agent.evaluator import evaluate_flags, log_interaction
from app.agent.formatter import FormattedResponse, format_response
from app.agent.interpreter import interpret_query
from app.agent.responder import generate_response
from app.agent.triage import classify_triage, generate_followup_questions

router = APIRouter(tags=["agent"])
logger = logging.getLogger(__name__)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=8000)


@router.post("/chat", response_model=FormattedResponse)
def chat(req: ChatRequest) -> FormattedResponse:
    interpreted = interpret_query(req.message)
    triage = classify_triage(interpreted)
    followups = generate_followup_questions(interpreted)

    try:
        raw = generate_response(req.message, triage, interpreted)
        out = format_response(raw, triage, follow_up_questions=followups)
    except Exception:
        logger.exception("chat generation failed")
        from app.agent.safety import urgency_message_for_triage

        out = FormattedResponse(
            triage_level=triage,  # type: ignore[arg-type]
            summary="Something went wrong generating a detailed answer. Please contact your veterinarian, especially if your dog seems unwell.",
            possible_causes=[],
            what_to_monitor=[],
            recommended_action=followups
            + ["Contact your veterinarian or an emergency clinic if you are worried."],
            urgency_message=urgency_message_for_triage(triage),
        )

    flags = evaluate_flags(req.message, triage, interpreted)
    try:
        log_interaction(user_input=req.message, triage_level=triage, response=out, flags=flags)
    except OSError:
        logger.warning("could not write eval log")

    return out
