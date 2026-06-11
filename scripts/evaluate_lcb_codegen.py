#!/usr/bin/env python3
"""Evaluate a saved LiveCodeBench code-generation run locally."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ens_spec_dec.evaluation.lcb_codegen import CodeEvalConfig, evaluate_lcb_codegen_run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="Completed run directory.")
    parser.add_argument(
        "--num-process-evaluate",
        type=int,
        default=4,
        help="Parallel worker count for CPU code evaluation.",
    )
    parser.add_argument("--timeout", type=int, default=6, help="Per-test timeout.")
    parser.add_argument("--debug", action="store_true", help="Run sequentially.")
    args = parser.parse_args()

    payload = evaluate_lcb_codegen_run(
        args.run_dir,
        CodeEvalConfig(
            num_process_evaluate=args.num_process_evaluate,
            timeout=args.timeout,
            debug=args.debug,
        ),
    )
    print(json.dumps({key: payload[key] for key in ("n_samples", "n_correct", "pass_at_1")}))


if __name__ == "__main__":
    main()
