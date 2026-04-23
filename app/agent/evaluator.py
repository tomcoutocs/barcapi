from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.agent.formatter import FormattedResponse
from app.agent.interpreter import interpret_query
from app.agent.triage import classify_triage
from app.config import get_settings

logger = logging.getLogger(__name__)

AGENT_TEST_CASES: list[dict[str, Any]] = [
    {
        "input": "dog ate chocolate 2 hours ago",
        "expect_triage_at_least": "emergency",
        "notes": "toxin",
    },
    {
        "input": "dog vomiting for 3 days",
        "expect_triage_at_least": "high",
        "notes": "prolonged gi",
    },
    {
        "input": "dog itchy",
        "expect_triage_at_least": "moderate",
        "notes": "single symptom",
    },
    {
        "input": "dog collapsed suddenly",
        "expect_triage_at_least": "emergency",
        "notes": "collapse",
    },
]

_TRIAGE_ORDER = {"low": 0, "moderate": 1, "high": 2, "emergency": 3}


def _triage_meets_minimum(actual: str, minimum: str) -> bool:
    return _TRIAGE_ORDER.get(actual, 0) >= _TRIAGE_ORDER.get(minimum, 0)


def log_interaction(
    *,
    user_input: str,
    triage_level: str,
    response: FormattedResponse | dict[str, Any],
    flags: dict[str, bool],
) -> None:
    settings = get_settings()
    path = Path(settings.agent_eval_log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "ts": datetime.now(UTC).isoformat(),
        "input": user_input,
        "triage_level": triage_level,
        "response": response.model_dump() if isinstance(response, FormattedResponse) else response,
        "flags": flags,
    }
    path.open("a", encoding="utf-8").write(json.dumps(row, ensure_ascii=False) + "\n")


def evaluate_flags(
    user_input: str,
    triage_level: str,
    interpreted: dict[str, Any],
) -> dict[str, bool]:
    """Heuristic QA flags for monitoring (not a full clinical audit)."""
    low_in = user_input.lower()
    missed_emergency = False
    if any(
        k in low_in
        for k in (
            "collapse",
            "collapsed",
            "chocolate",
            "xylitol",
            "can't breathe",
            "cant breathe",
            "gasping",
            "bloat",
        )
    ):
        if triage_level != "emergency":
            missed_emergency = True

    low_confidence = False
    if not interpreted.get("symptoms") and "?" not in user_input and len(user_input) > 20:
        low_confidence = triage_level == "low"
    return {"missed_emergency": missed_emergency, "low_confidence": low_confidence}


def run_eval_suite() -> list[dict[str, Any]]:
    """Run built-in triage tests (no LLM)."""
    results: list[dict[str, Any]] = []
    for case in AGENT_TEST_CASES:
        inp = case["input"]
        interpreted = interpret_query(inp)
        triage = classify_triage(interpreted)
        minimum = case["expect_triage_at_least"]
        ok = _triage_meets_minimum(triage, minimum)
        results.append(
            {
                "input": inp,
                "triage": triage,
                "expected_minimum": minimum,
                "pass": ok,
                "notes": case.get("notes"),
            }
        )
        if not ok:
            logger.warning("eval fail: %s got %s wanted >= %s", inp, triage, minimum)
    return results
