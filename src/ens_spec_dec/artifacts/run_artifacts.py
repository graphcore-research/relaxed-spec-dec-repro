"""Local run-dir I/O for one offline evaluation run.

Run directories live at `runs/<run_id>/`, where `run_id` looks like
`run-YYYYMMDD-HHMMSS-<token>`.

This module owns the fixed M1.4 layout:
- `manifest.json`: run metadata, serialized `RunConfig`, and relative paths
- `samples.jsonl`: append-only per-sample `ScoredSample` records during eval
- `summary.json`: optional run-level aggregates written after evaluation
- `logs/`: human-readable runner or backend logs
- `debug/`: extra debug files and later optional token/logit trace artifacts
- `plots/`: saved figures derived from the run

The helpers here initialize the run dir, append sample results during
evaluation, read saved artifacts back, write small run-local debug JSON files,
and expose completed sample ids for a small partial-resume flow.
"""

from __future__ import annotations

import json
import secrets
from datetime import datetime, timezone
from pathlib import Path

from ens_spec_dec.contracts import RunConfig, RunManifest, RunSummary, ScoredSample

RUN_MANIFEST_VERSION = "1"
RUN_ARTIFACT_PATHS = {
    "manifest": "manifest.json",
    "samples": "samples.jsonl",
    "summary": "summary.json",
    "logs_dir": "logs/",
    "debug_dir": "debug/",
    "plots_dir": "plots/",
}


def make_run_id(now: datetime | None = None, token: str | None = None) -> str:
    if now is None:
        now = datetime.now(timezone.utc)
    elif now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    else:
        now = now.astimezone(timezone.utc)

    token = secrets.token_hex(2) if token is None else token
    return f"run-{now:%Y%m%d-%H%M%S}-{token}"


def init_run_dir(
    runs_root: str | Path,
    config: RunConfig,
    created_at_utc: str | None = None,
) -> Path:
    run_dir = Path(runs_root) / config.metadata.run_id
    # TODO: When runner/resume logic lands, make the existing-dir policy
    # explicit: a fresh run should fail if the run dir already exists, while
    # an explicit resume mode should allow reuse only after validating that
    # the existing manifest/config matches the requested run.
    run_dir.mkdir(parents=True, exist_ok=True)

    for key in ("logs_dir", "debug_dir", "plots_dir"):
        _artifact_path(run_dir, RUN_ARTIFACT_PATHS[key]).mkdir(exist_ok=True)

    manifest = RunManifest(
        manifest_version=RUN_MANIFEST_VERSION,
        run_id=config.metadata.run_id,
        created_at_utc=created_at_utc
        or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        config=config,
        artifact_paths=dict(RUN_ARTIFACT_PATHS),
    )
    write_manifest(run_dir, manifest)
    return run_dir


def write_manifest(run_dir: str | Path, manifest: RunManifest) -> None:
    manifest.to_json(_artifact_path(run_dir, RUN_ARTIFACT_PATHS["manifest"]))


def read_manifest(run_dir: str | Path) -> RunManifest:
    return RunManifest.from_json(_artifact_path(run_dir, RUN_ARTIFACT_PATHS["manifest"]))


def append_scored_samples(run_dir: str | Path, samples: list[ScoredSample]) -> None:
    if not samples:
        return

    # Stable per-run artifact: offline_run appends measured scored samples only.
    path = _artifact_path(run_dir, RUN_ARTIFACT_PATHS["samples"])
    with path.open("a", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample.to_dict()))
            handle.write("\n")


def read_scored_samples(run_dir: str | Path) -> list[ScoredSample]:
    path = _artifact_path(run_dir, RUN_ARTIFACT_PATHS["samples"])
    if not path.exists():
        return []

    samples: list[ScoredSample] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        samples.append(ScoredSample.from_dict(json.loads(line)))
    return samples


def write_summary(run_dir: str | Path, summary: RunSummary) -> None:
    summary.to_json(_artifact_path(run_dir, RUN_ARTIFACT_PATHS["summary"]))


def read_summary(run_dir: str | Path) -> RunSummary:
    return RunSummary.from_json(_artifact_path(run_dir, RUN_ARTIFACT_PATHS["summary"]))


def write_debug_json(run_dir: str | Path, filename: str, payload: object) -> Path:
    if not filename.endswith(".json"):
        raise ValueError("debug artifact filename must end with .json")

    # Debug artifacts are diagnostics, not summary.json schema.
    path = _artifact_path(run_dir, RUN_ARTIFACT_PATHS["debug_dir"]) / filename
    text = json.dumps(payload, indent=2)
    path.write_text(f"{text}\n", encoding="utf-8")
    return path


def read_completed_sample_ids(run_dir: str | Path) -> set[str]:
    return {sample.sample.sample_id for sample in read_scored_samples(run_dir)}


def _artifact_path(run_dir: str | Path, relative_path: str) -> Path:
    return Path(run_dir) / relative_path.rstrip("/")


__all__ = [
    "RUN_ARTIFACT_PATHS",
    "RUN_MANIFEST_VERSION",
    "append_scored_samples",
    "init_run_dir",
    "make_run_id",
    "read_completed_sample_ids",
    "read_manifest",
    "read_scored_samples",
    "read_summary",
    "write_debug_json",
    "write_manifest",
    "write_summary",
]
