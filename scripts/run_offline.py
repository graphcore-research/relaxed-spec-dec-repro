"""Thin script entrypoint for M1.6 offline evaluation runs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ens_spec_dec.offline_run import run_offline_from_yaml
from ens_spec_dec.oom_retry import (
    clear_cuda_cache_after_oom,
    env_int,
    is_oom_error,
    prepare_run_dir_for_oom_retry,
    write_oom_fallback_config,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one offline M1.6 eval.")
    parser.add_argument("config", help="Path to a RunConfig YAML file.")
    parser.add_argument(
        "--runs-root",
        default="runs",
        help="Root directory that will contain runs/<run_id>/",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Optional run_id override without editing the YAML.",
    )
    parser.add_argument(
        "--oom-fallback-batch-size",
        type=int,
        default=env_int("SPEC_DEC_OOM_FALLBACK_BATCH_SIZE"),
        help=(
            "Retry once at this lower batch size after CUDA OOM. "
            "Defaults to half the config batch size."
        ),
    )
    args = parser.parse_args()
    if args.oom_fallback_batch_size is not None and args.oom_fallback_batch_size < 1:
        parser.error("--oom-fallback-batch-size must be >= 1")
    try:
        run_dir = run_offline_from_yaml(
            args.config,
            runs_root=args.runs_root,
            run_id=args.run_id,
        )
    except Exception as exc:
        if not is_oom_error(exc):
            raise
        fallback_config = write_oom_fallback_config(
            args.config,
            fallback_batch_size=args.oom_fallback_batch_size,
        )
        if fallback_config is None:
            raise
        print(
            "OOM retry "
            f"bs {fallback_config.original_batch_size} -> "
            f"{fallback_config.fallback_batch_size}: {fallback_config.path}"
        )
        clear_cuda_cache_after_oom()
        prepare_run_dir_for_oom_retry(args.runs_root, fallback_config.run_id)
        run_dir = run_offline_from_yaml(
            str(fallback_config.path),
            runs_root=args.runs_root,
            run_id=args.run_id,
        )
    print(run_dir)


if __name__ == "__main__":
    main()
