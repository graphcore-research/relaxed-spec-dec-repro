"""Proxy-speed reducer over offline generations and component profiles.

Offline runs remain the source for quality, output lengths, and acceptance
counters. Component profiles provide reusable cost tables. This reducer joins
those artifacts and writes analysis-level proxy-speed summaries.
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ens_spec_dec.artifacts import read_manifest, read_scored_samples
from ens_spec_dec.contracts import DecodeMode, JsonDict, RunManifest, ScoredSample
from ens_spec_dec.evaluation.summary import reduce_run_summary
from ens_spec_dec.profiling import read_component_profile


def build_proxy_speed_report(
    *,
    run_dirs: list[str | Path],
    profile_paths: list[str | Path],
    ar_baseline_run_dirs: list[str | Path] | None = None,
    analysis_id: str | None = None,
) -> JsonDict:
    if not run_dirs:
        raise ValueError("proxy-speed analysis requires at least one run dir")
    if not profile_paths:
        raise ValueError("proxy-speed analysis requires at least one profile")

    method_runs = [_load_generation_run(path) for path in run_dirs]
    baseline_runs = [
        _load_generation_run(path) for path in (ar_baseline_run_dirs or [])
    ]
    profiles = [
        {"path": str(path), "profile": read_component_profile(path)}
        for path in profile_paths
    ]

    method_aggregate = _aggregate_generation_runs(method_runs)
    baseline = _length_baseline(method_aggregate, baseline_runs)
    profile_results = [
        _profile_result(profile_item, method_aggregate, method_runs, baseline)
        for profile_item in profiles
    ]

    return {
        "summary_kind": "proxy_speed",
        "analysis_id": analysis_id,
        "created_at_utc": _utc_now(),
        "run_dirs": [str(path) for path in run_dirs],
        "profile_paths": [str(path) for path in profile_paths],
        "ar_baseline_run_dirs": [
            str(path) for path in (ar_baseline_run_dirs or [])
        ],
        "aggregate_inputs": _aggregate_payload(method_aggregate),
        "ar_baseline": baseline,
        "profiles": profile_results,
    }


def write_proxy_speed_report(
    *,
    run_dirs: list[str | Path],
    profile_paths: list[str | Path],
    output_path: str | Path,
    ar_baseline_run_dirs: list[str | Path] | None = None,
    analysis_id: str | None = None,
) -> JsonDict:
    report = build_proxy_speed_report(
        run_dirs=run_dirs,
        profile_paths=profile_paths,
        ar_baseline_run_dirs=ar_baseline_run_dirs,
        analysis_id=analysis_id,
    )
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def _profile_result(
    profile_item: JsonDict,
    aggregate: JsonDict,
    runs: list[JsonDict],
    baseline: JsonDict,
) -> JsonDict:
    profile = profile_item["profile"]
    compatibility = _profile_compatibility(profile, aggregate)
    result: JsonDict = {
        "profile_id": profile["profile_id"],
        "profile_path": profile_item["path"],
        "profile_subject": profile.get("profile_subject"),
        "compatibility": compatibility,
        "profile_hardware": dict(profile.get("hardware") or {}),
        "profile_software": dict(profile.get("software") or {}),
        "modes": {},
        "per_run": [],
    }
    if not compatibility["compatible"]:
        return result

    for mode_name, mode in (profile.get("derived_cost_modes") or {}).items():
        mode_result = _mode_result(mode_name, mode, profile, aggregate, baseline)
        result["modes"][mode_name] = mode_result
        result["per_run"].extend(
            _per_run_mode_result(mode_name, mode, profile, run, baseline)
            for run in runs
        )
    return result


def _mode_result(
    mode_name: str,
    mode: JsonDict,
    profile: JsonDict,
    aggregate: JsonDict,
    baseline: JsonDict,
) -> JsonDict:
    round_cost = _round_cost_for_mode(mode, aggregate["draft_length"])
    if round_cost is None:
        return _unavailable_mode(
            mode_name,
            f"profile has no round cost for draft_length={aggregate['draft_length']}",
        )
    return _compute_proxy_row(
        mode_name=mode_name,
        profile=profile,
        aggregate=aggregate,
        round_cost_ms=round_cost,
        baseline=baseline,
    )


def _per_run_mode_result(
    mode_name: str,
    mode: JsonDict,
    profile: JsonDict,
    run: JsonDict,
    baseline: JsonDict,
) -> JsonDict:
    round_cost = _round_cost_for_mode(mode, run["draft_length"])
    if round_cost is None:
        row = _unavailable_mode(
            mode_name,
            f"profile has no round cost for draft_length={run['draft_length']}",
        )
    else:
        row = _compute_proxy_row(
            mode_name=mode_name,
            profile=profile,
            aggregate=run,
            round_cost_ms=round_cost,
            baseline=baseline,
        )
    row["run_id"] = run["run_id"]
    row["run_dir"] = run["run_dir"]
    return row


def _compute_proxy_row(
    *,
    mode_name: str,
    profile: JsonDict,
    aggregate: JsonDict,
    round_cost_ms: float,
    baseline: JsonDict,
) -> JsonDict:
    target_ar_ms = _number_or_none(
        (profile.get("costs_ms") or {}).get("target_ar_token")
    )
    if target_ar_ms is None or target_ar_ms <= 0:
        return _unavailable_mode(mode_name, "profile has no target_ar_token cost")

    drafts, drafts_approx = _draft_count_for_proxy(aggregate)
    if drafts is None or drafts <= 0:
        return _unavailable_mode(
            mode_name,
            "run has no measured drafts and no usable acceptance span fallback",
        )

    output_tokens = aggregate["total_output_tokens"]
    modeled_ar_ms = output_tokens * target_ar_ms
    modeled_spec_ms = drafts * round_cost_ms
    speedup = modeled_ar_ms / modeled_spec_ms if modeled_spec_ms > 0 else None

    row: JsonDict = {
        "available": speedup is not None,
        "mode": mode_name,
        "draft_length": aggregate["draft_length"],
        "round_cost_ms": round_cost_ms,
        "target_ar_token_ms": target_ar_ms,
        "total_output_tokens": output_tokens,
        "total_drafts": drafts,
        "draft_count_is_approximate": drafts_approx,
        "accepted_draft_tokens": aggregate["accepted_draft_tokens"],
        "accepted_span": aggregate["accepted_span"],
        "modeled_ar_ms": modeled_ar_ms,
        "modeled_spec_ms": modeled_spec_ms,
        "token_throughput_proxy_speedup": speedup,
    }
    if baseline.get("available") and speedup is not None:
        length_ratio = baseline["length_ratio_vs_ar"]
        row["length_ratio_vs_ar"] = length_ratio
        row["length_adjusted_dataset_speedup"] = (
            speedup / length_ratio if length_ratio > 0 else None
        )
    return row


def _draft_count_for_proxy(aggregate: JsonDict) -> tuple[float | None, bool]:
    proxy_drafts = aggregate.get("proxy_drafts")
    if proxy_drafts is not None and proxy_drafts > 0:
        return float(proxy_drafts), bool(aggregate.get("drafts_are_approximate"))
    drafts = aggregate["total_drafts"]
    if drafts and drafts > 0:
        return float(drafts), False
    span = aggregate["accepted_span"]
    if span and span > 0:
        return aggregate["total_output_tokens"] / span, True
    return None, False


def _profile_compatibility(profile: JsonDict, aggregate: JsonDict) -> JsonDict:
    compat = profile.get("compatibility") or {}
    warnings = []
    checks = {
        "target_model": aggregate["target_model"],
        "draft_model": aggregate["draft_model"],
        "dtype": aggregate["dtype"],
        "batch_size": aggregate["batch_size"],
    }
    for key, expected in checks.items():
        observed = compat.get(key)
        if observed != expected:
            warnings.append(
                f"{key} mismatch: profile={observed!r} run={expected!r}"
            )

    draft_lengths = compat.get("draft_lengths")
    if isinstance(draft_lengths, list):
        if aggregate["draft_length"] not in [int(item) for item in draft_lengths]:
            warnings.append(
                "draft_length mismatch: "
                f"profile={draft_lengths!r} run={aggregate['draft_length']!r}"
            )
    else:
        warnings.append("profile missing compatibility.draft_lengths")

    return {"compatible": not warnings, "warnings": warnings}


def _length_baseline(
    method_aggregate: JsonDict,
    baseline_runs: list[JsonDict],
) -> JsonDict:
    if not baseline_runs:
        return {
            "available": False,
            "reason": "no AR baseline run dirs provided",
        }
    baseline_aggregate = _aggregate_generation_runs(baseline_runs)
    warnings = []
    if any(run["decode_mode"] != DecodeMode.AR_REF.value for run in baseline_runs):
        warnings.append("one or more baseline runs are not decode_mode=ar_ref")
    if baseline_aggregate["task_name"] != method_aggregate["task_name"]:
        warnings.append(
            "task mismatch: "
            f"method={method_aggregate['task_name']} "
            f"ar={baseline_aggregate['task_name']}"
        )
    if baseline_aggregate["sample_id_counts"] != method_aggregate["sample_id_counts"]:
        warnings.append("measured sample_id multiset differs from AR baseline")

    if warnings:
        return {
            "available": False,
            "warnings": warnings,
            "ar_total_output_tokens": baseline_aggregate["total_output_tokens"],
        }

    ar_tokens = baseline_aggregate["total_output_tokens"]
    method_tokens = method_aggregate["total_output_tokens"]
    length_ratio = method_tokens / ar_tokens if ar_tokens > 0 else None
    return {
        "available": length_ratio is not None,
        "ar_total_output_tokens": ar_tokens,
        "method_total_output_tokens": method_tokens,
        "length_ratio_vs_ar": length_ratio,
        "warnings": [],
    }


def _aggregate_generation_runs(runs: list[JsonDict]) -> JsonDict:
    first = runs[0]
    _validate_compatible_generation_runs(runs)
    sample_ids = Counter()
    total_output_tokens = 0
    total_drafts = 0
    accepted_tokens = 0
    per_pos: list[int] = []
    span_token_weighted_sum = 0.0
    span_token_count = 0
    proxy_drafts = 0.0
    drafts_are_approximate = False
    for run in runs:
        run_tokens = int(run["total_output_tokens"] or 0)
        total_output_tokens += run_tokens
        run_drafts = int(run["total_drafts"] or 0)
        total_drafts += run_drafts
        accepted_tokens += int(run["accepted_draft_tokens"] or 0)
        per_pos = _vector_sum(per_pos, run["accepted_draft_tokens_per_pos"])
        sample_ids.update(run["sample_id_counts"])
        if run["accepted_span"] is not None and run_tokens > 0:
            span_token_weighted_sum += float(run["accepted_span"]) * run_tokens
            span_token_count += run_tokens
        if run_drafts > 0:
            proxy_drafts += run_drafts
        elif run["accepted_span"] is not None and run["accepted_span"] > 0:
            proxy_drafts += run_tokens / float(run["accepted_span"])
            drafts_are_approximate = True

    accepted_span = None
    if total_drafts > 0:
        accepted_span = 1.0 + accepted_tokens / total_drafts
    elif span_token_count > 0:
        accepted_span = span_token_weighted_sum / span_token_count

    return {
        "run_ids": [run["run_id"] for run in runs],
        "task_name": first["task_name"],
        "decode_mode": first["decode_mode"],
        "target_model": first["target_model"],
        "draft_model": first["draft_model"],
        "dtype": first["dtype"],
        "batch_size": first["batch_size"],
        "draft_length": first["draft_length"],
        "total_output_tokens": total_output_tokens,
        "total_drafts": total_drafts,
        "proxy_drafts": proxy_drafts if proxy_drafts > 0 else None,
        "drafts_are_approximate": drafts_are_approximate,
        "accepted_draft_tokens": accepted_tokens,
        "accepted_draft_tokens_per_pos": per_pos,
        "accepted_span": accepted_span,
        "sample_id_counts": dict(sample_ids),
    }


def _validate_compatible_generation_runs(runs: list[JsonDict]) -> None:
    first = runs[0]
    fields = (
        "task_name",
        "decode_mode",
        "target_model",
        "draft_model",
        "dtype",
        "batch_size",
        "draft_length",
    )
    for run in runs[1:]:
        for field in fields:
            if run[field] != first[field]:
                raise ValueError(
                    "cannot aggregate proxy-speed runs with different "
                    f"{field}: {first[field]!r} vs {run[field]!r}"
                )


def _aggregate_payload(aggregate: JsonDict) -> JsonDict:
    payload = dict(aggregate)
    payload["sample_id_counts"] = dict(aggregate["sample_id_counts"])
    return payload


def _load_generation_run(run_dir: str | Path) -> JsonDict:
    run_path = Path(run_dir)
    manifest = read_manifest(run_path)
    summary = reduce_run_summary(run_path)
    samples = read_scored_samples(run_path)
    method_summary = summary.method_summary
    return {
        "run_dir": str(run_path),
        "run_id": manifest.run_id,
        "task_name": summary.task_name,
        "decode_mode": summary.decode_mode.value,
        "target_model": summary.model,
        "draft_model": summary.draft_model,
        "dtype": manifest.config.backend.dtype,
        "batch_size": summary.batch_size,
        "draft_length": manifest.config.generation_params.method.draft_length,
        "total_output_tokens": summary.total_output_tokens or 0,
        "total_drafts": int(method_summary.get("delta_num_drafts") or 0),
        "accepted_draft_tokens": int(
            method_summary.get("delta_num_accepted_tokens") or 0
        ),
        "accepted_draft_tokens_per_pos": _int_list(
            method_summary.get("delta_num_accepted_tokens_per_pos")
        ),
        "accepted_span": (
            method_summary.get("mean_acceptance_length_with_bonus")
            or _sample_acceptance_span(samples)
        ),
        "sample_id_counts": _measured_sample_id_counts(samples, manifest),
    }


def _measured_sample_id_counts(
    samples: list[ScoredSample],
    manifest: RunManifest,
) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for sample in samples:
        phase = sample.metadata.get("phase")
        if phase in (None, "measured") and sample.run_id == manifest.run_id:
            counts[sample.sample.sample_id] += 1
    return dict(counts)


def _sample_acceptance_span(samples: list[ScoredSample]) -> float | None:
    spans = []
    for sample in samples:
        phase = sample.metadata.get("phase")
        if phase not in (None, "measured"):
            continue
        span = sample.generation.method_stats.get(
            "mean_acceptance_length_with_bonus"
        )
        if isinstance(span, (int, float)) and not isinstance(span, bool):
            spans.append(float(span))
    return sum(spans) / len(spans) if spans else None


def _round_cost_for_mode(mode: JsonDict, draft_length: int | None) -> float | None:
    if draft_length is None:
        return None
    costs = mode.get("round_cost_ms_by_d")
    if not isinstance(costs, dict):
        return None
    return _number_or_none(costs.get(str(draft_length)))


def _unavailable_mode(mode_name: str, reason: str) -> JsonDict:
    return {"available": False, "mode": mode_name, "reason": reason}


def _int_list(value: Any) -> list[int]:
    if not isinstance(value, list):
        return []
    return [
        int(item)
        for item in value
        if isinstance(item, (int, float)) and not isinstance(item, bool)
    ]


def _number_or_none(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _vector_sum(left: list[int], right: list[int]) -> list[int]:
    width = max(len(left), len(right))
    return [
        (left[index] if index < len(left) else 0)
        + (right[index] if index < len(right) else 0)
        for index in range(width)
    ]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


__all__ = ["build_proxy_speed_report", "write_proxy_speed_report"]
