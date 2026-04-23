"""Run triage-only eval suite (no LLM). From vet-rag-api: python -m scripts.run_agent_eval"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app.agent.evaluator import AGENT_TEST_CASES, run_eval_suite  # noqa: E402


def main() -> None:
    results = run_eval_suite()
    print(json.dumps({"cases": AGENT_TEST_CASES, "results": results}, indent=2))
    failed = [r for r in results if not r["pass"]]
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
