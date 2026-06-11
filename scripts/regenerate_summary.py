"""Regenerate stable summaries from completed local run directories."""

from __future__ import annotations

import argparse
import secrets
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ens_spec_dec.evaluation import regenerate_run_summary, write_multi_run_summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regenerate summary.json from saved offline run artifacts."
    )
    parser.add_argument("run_dirs", nargs="+", help="Completed run directories.")
    parser.add_argument(
        "--output",
        default=None,
        help="Output path for multi-run analysis summary.",
    )
    parser.add_argument(
        "--analysis-id",
        default=None,
        help="Analysis id used for multi-run summaries.",
    )
    parser.add_argument(
        "--runs-root",
        default="runs",
        help="Runs root used when --output is omitted for multi-run summaries.",
    )
    args = parser.parse_args()

    if len(args.run_dirs) == 1 and args.output is None:
        summary = regenerate_run_summary(args.run_dirs[0])
        print(Path(args.run_dirs[0]) / "summary.json")
        print(f"run_id={summary.run_id} samples={summary.n_samples}")
        return

    analysis_id = args.analysis_id or _make_analysis_id()
    output_path = (
        Path(args.output)
        if args.output is not None
        else Path(args.runs_root) / "analysis" / analysis_id / "summary.json"
    )
    summary = write_multi_run_summary(
        args.run_dirs,
        output_path=output_path,
        analysis_id=analysis_id,
    )
    print(output_path)
    sample_count = summary["aggregate"]["measured_sample_count"]
    print(f"runs={len(summary['run_ids'])} samples={sample_count}")


def _make_analysis_id() -> str:
    now = datetime.now(timezone.utc)
    token = secrets.token_hex(2)
    return f"analysis-summary-{now:%Y%m%d-%H%M%S}-{token}"


if __name__ == "__main__":
    main()
