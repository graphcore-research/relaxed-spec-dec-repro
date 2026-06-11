"""Small helpers for benchmark row loading.

Checked-in fixtures stay under `tasks/data/`. Full benchmark files live under
an untracked data root, defaulting to `data/benchmarks/`, so users can download
them locally without bloating the repository.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

DATA_DIR = Path(__file__).with_name("data")
BENCHMARK_DATA_ROOT_ENV = "SPEC_DEC_BENCHMARK_DATA_ROOT"
DEFAULT_BENCHMARK_DATA_ROOT = Path("data/benchmarks")


def load_fixture_rows(filename: str, max_samples: int | None = None) -> list[dict]:
    path = DATA_DIR / filename
    return load_jsonl_rows(path, max_samples)


def benchmark_data_root() -> Path:
    root = os.environ.get(BENCHMARK_DATA_ROOT_ENV)
    return Path(root).expanduser() if root else DEFAULT_BENCHMARK_DATA_ROOT


def load_benchmark_rows(
    task_dir: str,
    filename: str,
    max_samples: int | None = None,
) -> list[dict]:
    path = benchmark_data_root() / task_dir / filename
    if not path.is_file():
        raise FileNotFoundError(
            f"missing benchmark data file {path}; run "
            f"`python3 scripts/prepare_data.py` or set "
            f"{BENCHMARK_DATA_ROOT_ENV}"
        )
    return load_jsonl_rows(path, max_samples)


def load_jsonl_rows(path: str | Path, max_samples: int | None = None) -> list[dict]:
    path = Path(path)
    rows: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(json.loads(line))
    return rows if max_samples is None else rows[:max_samples]


def apply_task_sample_slice(rows: list[dict], task_config: Any) -> list[dict]:
    """Apply a config-level sample slice after loading the full JSONL rows."""
    sample_slice = getattr(task_config, "metadata", {}).get("sample_slice")
    if not isinstance(sample_slice, dict):
        max_samples = getattr(task_config, "max_samples", None)
        return rows if max_samples is None else rows[:max_samples]

    policy = sample_slice.get("policy")
    if policy != "strided":
        raise ValueError(f"unsupported sample_slice policy: {policy!r}")

    count = int(sample_slice["count"])
    offset = int(sample_slice.get("offset", 0))
    if count < 1:
        raise ValueError("strided sample_slice count must be >= 1")
    if offset < 0:
        raise ValueError("strided sample_slice offset must be >= 0")
    if count > len(rows):
        raise ValueError(
            f"strided sample_slice count {count} exceeds row count {len(rows)}"
        )

    # Evenly spread the small sweep slice across the prepared benchmark order.
    indexes = [
        ((index * len(rows)) // count + offset) % len(rows)
        for index in range(count)
    ]
    return [rows[index] for index in indexes]


__all__ = [
    "BENCHMARK_DATA_ROOT_ENV",
    "DATA_DIR",
    "DEFAULT_BENCHMARK_DATA_ROOT",
    "apply_task_sample_slice",
    "benchmark_data_root",
    "load_benchmark_rows",
    "load_fixture_rows",
    "load_jsonl_rows",
]
