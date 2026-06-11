"""Compute proxy-speed summaries from completed runs and cost profiles."""

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

from ens_spec_dec.analysis import write_proxy_speed_report


def main() -> None:
    args = parse_args()
    analysis_id = args.analysis_id or _make_analysis_id()
    output_path = (
        Path(args.output)
        if args.output
        else Path(args.runs_root) / "analysis" / analysis_id / "summary.json"
    )
    report = write_proxy_speed_report(
        run_dirs=args.run_dirs,
        profile_paths=args.profiles,
        ar_baseline_run_dirs=args.ar_baseline_run_dirs,
        output_path=output_path,
        analysis_id=analysis_id,
    )
    print(output_path)
    print(f"profiles={len(report['profiles'])} runs={len(report['run_dirs'])}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Join offline generation artifacts with component profiles."
    )
    parser.add_argument(
        "--run-dirs",
        nargs="+",
        required=True,
        help="Completed SD or relaxed-method offline run directories.",
    )
    parser.add_argument(
        "--profiles",
        nargs="+",
        required=True,
        help="Component profile JSON files.",
    )
    parser.add_argument(
        "--ar-baseline-run-dirs",
        nargs="*",
        default=[],
        help="Optional AR baseline run dirs for length adjustment.",
    )
    parser.add_argument("--output", default=None, help="Output summary path.")
    parser.add_argument("--analysis-id", default=None, help="Analysis id.")
    parser.add_argument(
        "--runs-root",
        default="runs",
        help="Runs root used when --output is omitted.",
    )
    return parser.parse_args()


def _make_analysis_id() -> str:
    now = datetime.now(timezone.utc)
    token = secrets.token_hex(2)
    return f"proxy-speed-{now:%Y%m%d-%H%M%S}-{token}"


if __name__ == "__main__":
    main()
