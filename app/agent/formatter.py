from __future__ import annotations

import json
import re
from typing import Any, Literal

from pydantic import BaseModel, Field

TriageLiteral = Literal["emergency", "high", "moderate", "low"]


class FormattedResponse(BaseModel):
    triage_level: TriageLiteral
    summary: str
    possible_causes: list[str] = Field(default_factory=list)
    what_to_monitor: list[str] = Field(default_factory=list)
    recommended_action: list[str] = Field(default_factory=list)
    urgency_message: str


def _strip_fences(raw: str) -> str:
    t = raw.strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*", "", t)
        t = re.sub(r"\s*```$", "", t)
    return t.strip()


def format_response(
    llm_raw: str | dict[str, Any],
    triage_level: str,
    *,
    follow_up_questions: list[str] | None = None,
) -> FormattedResponse:
    """Validate / coerce model output into the public API shape."""
    um = _urgency_fallback(triage_level)
    if isinstance(llm_raw, dict):
        data = dict(llm_raw)
    else:
        try:
            data = json.loads(_strip_fences(llm_raw))
        except json.JSONDecodeError:
            data = {
                "summary": llm_raw[:2000] if llm_raw else "Unable to parse structured response.",
                "possible_causes": [],
                "what_to_monitor": [],
                "recommended_action": [],
            }

    # Authoritative triage from rules engine (do not let the model downgrade escalation).
    tl = triage_level.lower()
    if tl not in ("emergency", "high", "moderate", "low"):
        tl = "moderate"

    rec = [str(x) for x in (data.get("recommended_action") or []) if str(x).strip()]
    if follow_up_questions:
        rec = [f"(Your vet may ask) {q}" for q in follow_up_questions] + rec

    return FormattedResponse(
        triage_level=tl,  # type: ignore[arg-type]
        summary=str(data.get("summary") or "").strip() or "See recommended actions.",
        possible_causes=[str(x) for x in (data.get("possible_causes") or []) if str(x).strip()][:8],
        what_to_monitor=[str(x) for x in (data.get("what_to_monitor") or []) if str(x).strip()][:8],
        recommended_action=rec[:12],
        urgency_message=str(data.get("urgency_message") or "").strip() or um,
    )


def _urgency_fallback(triage_level: str) -> str:
    from app.agent.safety import urgency_message_for_triage

    return urgency_message_for_triage(triage_level)
