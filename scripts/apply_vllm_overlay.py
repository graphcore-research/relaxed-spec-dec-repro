#!/usr/bin/env python3
"""Apply the bundled relaxed-spec-dec vLLM overlay."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vllm_overlay.apply_overlay import main


if __name__ == "__main__":
    raise SystemExit(main())
