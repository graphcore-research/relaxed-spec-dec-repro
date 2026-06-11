#!/usr/bin/env python3
"""Write a compact CSV summary from completed run directories."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ens_spec_dec.artifacts import read_manifest, read_summary


FIELDS = [
    "run_id",
    "task_name",
    "decode_mode",
    "model",
    "draft_model",
    "method_name",
    "method_variant",
    "draft_length",
    "n_samples",
    "accuracy",
    "n_correct",
    "total_output_tokens",
    "mean_output_tokens",
    "total_generate_wall_time_s",
    "output_tokens_per_s",
    "delta_num_drafts",
    "delta_num_draft_tokens",
    "delta_num_accepted_tokens",
    "mean_accepted_draft_tokens",
    "mean_acceptance_length_with_bonus",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--output", default="runs/summary.csv")
    args = parser.parse_args()

    rows = rows_from_runs(Path(args.runs_root))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {output} ({len(rows)} rows)")


def rows_from_runs(runs_root: Path) -> list[dict[str, object]]:
    if not runs_root.exists():
        raise SystemExit(f"runs root does not exist: {runs_root}")
    rows = []
    for run_dir in sorted(path for path in runs_root.iterdir() if path.is_dir()):
        summary_path = run_dir / "summary.json"
        manifest_path = run_dir / "manifest.json"
        if not summary_path.is_file() or not manifest_path.is_file():
            continue
        manifest = read_manifest(run_dir)
        summary = read_summary(run_dir)
        method = manifest.config.generation_params.method
        score = summary.score_summary
        efficiency = summary.efficiency_summary
        method_summary = summary.method_summary
        rows.append(
            {
                "run_id": summary.run_id,
                "task_name": summary.task_name,
                "decode_mode": summary.decode_mode.value,
                "model": summary.model,
                "draft_model": summary.draft_model,
                "method_name": method.name,
                "method_variant": method.variant,
                "draft_length": method.draft_length,
                "n_samples": summary.n_samples,
                "accuracy": score.get("accuracy"),
                "n_correct": score.get("n_correct"),
                "total_output_tokens": summary.total_output_tokens,
                "mean_output_tokens": efficiency.get("mean_output_tokens"),
                "total_generate_wall_time_s": summary.total_generate_wall_time_s,
                "output_tokens_per_s": efficiency.get("output_tokens_per_s"),
                "delta_num_drafts": method_summary.get("delta_num_drafts"),
                "delta_num_draft_tokens": method_summary.get("delta_num_draft_tokens"),
                "delta_num_accepted_tokens": method_summary.get(
                    "delta_num_accepted_tokens"
                ),
                "mean_accepted_draft_tokens": method_summary.get(
                    "mean_accepted_draft_tokens"
                ),
                "mean_acceptance_length_with_bonus": method_summary.get(
                    "mean_acceptance_length_with_bonus"
                ),
            }
        )
    return rows


if __name__ == "__main__":
    main()
