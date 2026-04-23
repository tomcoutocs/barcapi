"""
Example: run the scrape pipeline from the vet-rag-api directory:

  python -m scripts.scrape_example

Equivalent to: python pipeline_main.py
(Prefer editing SEED_URLS in pipeline_main.py.)
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pipeline_main import main  # noqa: E402

if __name__ == "__main__":
    main()
