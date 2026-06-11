"""Reducers for stable offline summaries over saved run artifacts.

The single-run reducer reads one completed run directory and returns the
existing `RunSummary` contract. The multi-run reducer is an analysis-level
helper for explicit sibling run dirs; it never invokes a backend or rewrites a
sibling run's local summary.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from statistics import mean, stdev
from typing import Any

from ens_spec_dec.artifacts import (
    RUN_ARTIFACT_PATHS,
    read_manifest,
    read_scored_samples,
    write_summary,
)
from ens_spec_dec.contracts import JsonDict, RunManifest, RunSummary, ScoredSample

VLLM_SPEC_METRIC_NAMES = {
    "vllm:spec_decode_num_drafts": "num_drafts",
    "vllm:spec_decode_num_draft_tokens": "num_draft_tokens",
    "vllm:spec_decode_num_accepted_tokens": "num_accepted_tokens",
    "vllm:spec_decode_num_bonus_activations": "num_bonus_activations",
    "vllm:spec_decode_num_target_bonus_activations": (
        "num_target_bonus_activations"
    ),
    "vllm:spec_decode_num_relaxed_bonus_activations": (
        "num_relaxed_bonus_activations"
    ),
    "vllm:spec_decode_num_accepted_tokens_per_pos": (
        "num_accepted_tokens_per_pos"
    ),
}
DELTA_FIELD_NAMES = {
    "num_drafts": "delta_num_drafts",
    "num_draft_tokens": "delta_num_draft_tokens",
    "num_accepted_tokens": "delta_num_accepted_tokens",
    "num_bonus_activations": "delta_num_bonus_activations",
    "num_target_bonus_activations": "delta_num_target_bonus_activations",
    "num_relaxed_bonus_activations": "delta_num_relaxed_bonus_activations",
    "num_accepted_tokens_per_pos": "delta_num_accepted_tokens_per_pos",
}
ENTROPY_STAT_NAMES = (
    "drafter_accepted",
    "verifier_on_accepted",
    "verifier_on_output",
)
ENTROPY_UNIT = "nats"
ENTROPY_DISTRIBUTION = "processed_logits_after_temperature_top_p_top_k"


def reduce_run_summary(run_dir: str | Path) -> RunSummary:
    """Read one completed run dir and compute its stable `RunSummary`."""

    loaded = _load_run_data(run_dir)
    rows = _measured_rows(loaded["samples"], scope_run_id=loaded["manifest"].run_id)
    aggregate = _aggregate_rows(rows, call_scope="single")
    method_summary = _method_summary_from_debug(loaded["debug_payload"])
    entropy_summary = _entropy_summary_from_rows(rows)
    if entropy_summary:
        method_summary["entropy"] = entropy_summary
    score_summary = aggregate["score_summary"]
    efficiency_summary = aggregate["efficiency_summary"]
    config = loaded["manifest"].config
    efficiency_summary["repeat_batch_size"] = config.execution.repeat_batch_size
    efficiency_summary["repeat_seed_offset"] = config.execution.repeat_seed_offset
    efficiency_summary["unique_sample_batch_size"] = (
        config.execution.batch_size // config.execution.repeat_batch_size
    )
    efficiency_summary["excluded_non_measured_count"] = loaded[
        "excluded_non_measured_count"
    ]
    efficiency_summary["legacy_missing_phase_count"] = loaded[
        "legacy_missing_phase_count"
    ]
    if not config.execution.warmup_enabled:
        efficiency_summary["timing_label"] = "no_warmup_timing"

    return RunSummary(
        run_id=loaded["manifest"].run_id,
        task_name=config.task.name,
        measurement_scope=config.metadata.measurement_scope,
        decode_mode=config.metadata.decode_mode,
        backend_name=config.backend.name,
        model=config.backend.model,
        draft_model=config.backend.draft_model,
        batch_size=config.execution.batch_size,
        n_samples=aggregate["measured_sample_count"],
        n_generate_calls=aggregate["n_generate_calls"],
        total_prompt_tokens=aggregate["total_prompt_tokens"],
        total_output_tokens=aggregate["total_output_tokens"],
        total_generate_wall_time_s=aggregate["total_generate_wall_time_s"],
        score_summary=score_summary,
        efficiency_summary=efficiency_summary,
        method_summary=method_summary,
        debug_refs=_summary_debug_refs(loaded["run_dir"], loaded["debug_payload"]),
    )


def regenerate_run_summary(run_dir: str | Path) -> RunSummary:
    """Compute and write `summary.json` for one completed run dir."""

    summary = reduce_run_summary(run_dir)
    write_summary(run_dir, summary)
    return summary


def reduce_multi_run_summary(
    run_dirs: list[str | Path],
    analysis_id: str | None = None,
) -> JsonDict:
    """Reduce explicit sibling run dirs into one analysis-level summary."""

    if not run_dirs:
        raise ValueError("multi-run summary requires at least one run dir")

    loaded_runs = [_load_run_data(run_dir) for run_dir in run_dirs]
    all_rows = []
    per_run = []
    run_breakdowns: JsonDict = {}
    for loaded in loaded_runs:
        manifest = loaded["manifest"]
        rows = _measured_rows(loaded["samples"], scope_run_id=manifest.run_id)
        all_rows.extend(rows)
        summary = _summary_to_multi_run_item(
            reduce_run_summary(loaded["run_dir"]),
            loaded["run_dir"],
            manifest,
        )
        per_run.append(summary)
        run_breakdowns[manifest.run_id] = _repeat_breakdown(rows)

    aggregate = _aggregate_rows(all_rows, call_scope="multi")
    aggregate["efficiency_summary"]["excluded_non_measured_count"] = sum(
        loaded["excluded_non_measured_count"] for loaded in loaded_runs
    )
    aggregate["efficiency_summary"]["legacy_missing_phase_count"] = sum(
        loaded["legacy_missing_phase_count"] for loaded in loaded_runs
    )
    method_summary = _combine_method_summaries(
        [_method_summary_from_debug(loaded["debug_payload"]) for loaded in loaded_runs]
    )
    entropy_summary = _entropy_summary_from_rows(all_rows)
    if entropy_summary:
        method_summary["entropy"] = entropy_summary
    aggregate_payload: JsonDict = {
        "measured_sample_count": aggregate["measured_sample_count"],
        "n_generate_calls": aggregate["n_generate_calls"],
        "total_prompt_tokens": aggregate["total_prompt_tokens"],
        "total_output_tokens": aggregate["total_output_tokens"],
        "total_generate_wall_time_s": aggregate["total_generate_wall_time_s"],
        "output_tokens_per_s": aggregate["efficiency_summary"][
            "output_tokens_per_s"
        ],
        "score_summary": aggregate["score_summary"],
        "efficiency_summary": aggregate["efficiency_summary"],
        "method_summary": method_summary,
    }
    return {
        "summary_kind": "multi_run",
        "analysis_id": analysis_id,
        "run_ids": [loaded["manifest"].run_id for loaded in loaded_runs],
        "run_dirs": [str(loaded["run_dir"]) for loaded in loaded_runs],
        "aggregate": aggregate_payload,
        "per_run": per_run,
        "run_dimension_summary": _run_dimension_summary(per_run),
        "breakdowns": {
            "by_run_id": run_breakdowns,
            "overall": _repeat_breakdown(all_rows),
        },
    }


def write_multi_run_summary(
    run_dirs: list[str | Path],
    output_path: str | Path,
    analysis_id: str | None = None,
) -> JsonDict:
    """Write an analysis-level summary for explicit sibling run dirs."""

    summary = reduce_multi_run_summary(run_dirs, analysis_id=analysis_id)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{json.dumps(summary, indent=2)}\n", encoding="utf-8")
    return summary


def build_vllm_measured_window(
    start_payload: JsonDict,
    end_payload: JsonDict,
) -> JsonDict:
    """Build measured-window spec-decode deltas from two vLLM metric snapshots."""

    start_totals = _spec_metric_totals(start_payload.get("metrics"))
    end_totals = _spec_metric_totals(end_payload.get("metrics"))
    metric_deltas: JsonDict = {}
    if start_totals or end_totals:
        for key in (
            "num_drafts",
            "num_draft_tokens",
            "num_accepted_tokens",
        ):
            metric_deltas[DELTA_FIELD_NAMES[key]] = int(
                end_totals.get(key, 0) - start_totals.get(key, 0)
            )
        for key in (
            "num_bonus_activations",
            "num_target_bonus_activations",
            "num_relaxed_bonus_activations",
        ):
            if key in start_totals or key in end_totals:
                metric_deltas[DELTA_FIELD_NAMES[key]] = int(
                    end_totals.get(key, 0) - start_totals.get(key, 0)
                )
        metric_deltas["delta_num_accepted_tokens_per_pos"] = (
            _vector_delta(
                start_totals.get("num_accepted_tokens_per_pos"),
                end_totals.get("num_accepted_tokens_per_pos"),
            )
        )

    return {
        "metric_deltas": metric_deltas,
        "start_metrics": list(start_payload.get("metrics") or []),
        "end_metrics": list(end_payload.get("metrics") or []),
        "coverage": {
            "start_metrics_present": bool(start_payload.get("metrics")),
            "end_metrics_present": bool(end_payload.get("metrics")),
            "spec_decode_metrics_present": bool(start_totals or end_totals),
            "measured_vllm_deltas_present": bool(metric_deltas),
        },
        "source_note": (
            "Deltas are after warmup and before/after measured generation. "
            "vLLM accepted-token counters exclude bonus/resampled output tokens."
        ),
    }


def _load_run_data(run_dir: str | Path) -> JsonDict:
    run_path = Path(run_dir)
    manifest = read_manifest(run_path)
    samples = read_scored_samples(run_path)
    excluded = 0
    legacy_missing_phase = 0
    for sample in samples:
        phase_present = "phase" in sample.metadata
        phase = sample.metadata.get("phase")
        if not phase_present or phase is None:
            legacy_missing_phase += 1
        elif phase != "measured":
            excluded += 1

    return {
        "run_dir": run_path,
        "manifest": manifest,
        "samples": samples,
        "debug_payload": _read_debug_payload(run_path),
        "excluded_non_measured_count": excluded,
        "legacy_missing_phase_count": legacy_missing_phase,
    }


def _read_debug_payload(run_dir: Path) -> JsonDict | None:
    path = run_dir / RUN_ARTIFACT_PATHS["debug_dir"].rstrip("/") / "vllm_metrics.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _measured_rows(samples: list[ScoredSample], scope_run_id: str) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for index, sample in enumerate(samples):
        phase = sample.metadata.get("phase")
        if phase not in (None, "measured"):
            continue
        rows.append(_row_from_sample(sample, scope_run_id=scope_run_id, index=index))
    return rows


def _row_from_sample(
    sample: ScoredSample,
    scope_run_id: str,
    index: int,
) -> JsonDict:
    generation = sample.generation
    n_out_tokens = generation.n_out_tokens
    output_tokens = (
        len(generation.generated_token_ids)
        if n_out_tokens is None
        else n_out_tokens
    )
    return {
        "run_id": scope_run_id,
        "row_index": index,
        "sample_id": sample.sample.sample_id,
        "output_tokens": output_tokens,
        "used_output_token_fallback": n_out_tokens is None,
        "prompt_tokens": generation.n_prompt_tokens,
        "generate_call_id": generation.generate_call_id,
        "generate_wall_time_s": generation.generate_wall_time_s,
        "hit_max_new_tokens": generation.hit_max_new_tokens,
        "correct": sample.score_payload.get("correct"),
        "has_correct": "correct" in sample.score_payload,
        "exact_match": sample.score_payload.get("exact_match"),
        "score_payload": dict(sample.score_payload),
        "method_stats": dict(generation.method_stats),
        "metadata": dict(sample.metadata),
    }


def _aggregate_rows(rows: list[JsonDict], call_scope: str) -> JsonDict:
    output_tokens = [int(row["output_tokens"]) for row in rows]
    wall_times = [
        row["generate_wall_time_s"]
        for row in rows
        if row["generate_wall_time_s"] is not None
    ]
    prompt_values = [row["prompt_tokens"] for row in rows]
    missing_prompt_count = sum(value is None for value in prompt_values)
    total_prompt_tokens = (
        sum(int(value) for value in prompt_values)
        if rows and missing_prompt_count == 0
        else None
    )
    if not rows and missing_prompt_count == 0:
        total_prompt_tokens = 0

    call_stats = _unique_call_stats(rows, call_scope=call_scope)
    score_summary = _score_summary(rows)
    efficiency_summary = {
        "mean_output_tokens": _mean_or_zero(output_tokens),
        "std_output_tokens": _std_or_zero(output_tokens),
        "mean_generate_wall_time_s": _mean_or_zero(wall_times),
        "std_generate_wall_time_s": _std_or_zero(wall_times),
        "output_tokens_per_s": _tokens_per_s(
            sum(output_tokens),
            call_stats["total_generate_wall_time_s"],
        ),
        "attempt_ids": _unique_metadata_values(rows, "attempt_id"),
        "repeat_indices": _unique_metadata_values(rows, "repeat_index"),
        "repeat_batch_indices": _unique_metadata_values(rows, "repeat_batch_index"),
        "seeds": _unique_metadata_values(rows, "seed"),
        "sample_count_by_attempt": _count_by_metadata(rows, "attempt_id"),
        "sample_count_by_repeat_index": _count_by_metadata(rows, "repeat_index"),
        "sample_count_by_repeat_batch_index": _count_by_metadata(
            rows,
            "repeat_batch_index",
        ),
        "sample_count_by_seed": _count_by_metadata(rows, "seed"),
        "missing_prompt_token_count": missing_prompt_count,
        "available_prompt_token_count": len(rows) - missing_prompt_count,
        "used_output_token_fallback_count": sum(
            bool(row["used_output_token_fallback"]) for row in rows
        ),
        "hit_max_new_tokens_count": sum(
            row["hit_max_new_tokens"] is True for row in rows
        ),
        "hit_max_new_tokens_rate": _hit_rate(rows),
        "missing_hit_max_new_tokens_count": sum(
            row["hit_max_new_tokens"] is None for row in rows
        ),
        "missing_generate_wall_time_count": sum(
            row["generate_wall_time_s"] is None for row in rows
        ),
        "missing_generate_call_id_count": sum(
            row["generate_call_id"] is None for row in rows
        ),
        "missing_generate_call_wall_time_count": call_stats[
            "missing_call_wall_time_count"
        ],
        "wall_time_conflict_count": call_stats["wall_time_conflict_count"],
        "call_id_fallback_count": call_stats["call_id_fallback_count"],
    }
    return {
        "measured_sample_count": len(rows),
        "n_generate_calls": call_stats["n_generate_calls"],
        "total_prompt_tokens": total_prompt_tokens,
        "total_output_tokens": sum(output_tokens),
        "total_generate_wall_time_s": call_stats["total_generate_wall_time_s"],
        "score_summary": score_summary,
        "efficiency_summary": efficiency_summary,
    }


def _unique_call_stats(rows: list[JsonDict], call_scope: str) -> JsonDict:
    call_times: dict[str, float | None] = {}
    conflict_count = 0
    fallback_count = 0
    for row in rows:
        call_id = row["generate_call_id"]
        if call_id is None:
            fallback_count += 1
            call_key = f"{row['run_id']}::missing::{row['row_index']}"
        elif call_scope == "multi":
            call_key = f"{row['run_id']}::{call_id}"
        else:
            call_key = str(call_id)

        wall_time = row["generate_wall_time_s"]
        previous = call_times.get(call_key)
        if call_key not in call_times:
            call_times[call_key] = wall_time
        elif wall_time is not None:
            if previous is None:
                call_times[call_key] = wall_time
            elif previous != wall_time:
                conflict_count += 1
                call_times[call_key] = max(previous, wall_time)

    available_wall_times = [
        value for value in call_times.values() if value is not None
    ]
    return {
        "n_generate_calls": len(call_times),
        "total_generate_wall_time_s": (
            sum(available_wall_times) if available_wall_times else None
        ),
        "missing_call_wall_time_count": sum(
            value is None for value in call_times.values()
        ),
        "wall_time_conflict_count": conflict_count,
        "call_id_fallback_count": fallback_count,
    }


def _score_summary(rows: list[JsonDict]) -> JsonDict:
    scored_rows = [row for row in rows if row["has_correct"]]
    n_correct = sum(bool(row["correct"]) for row in scored_rows)
    exact_matches = [
        float(row["exact_match"])
        for row in rows
        if _is_number(row["exact_match"])
    ]
    summary = {
        "n_scored": len(scored_rows),
        "n_correct": n_correct,
        "accuracy": (n_correct / len(scored_rows)) if scored_rows else None,
        "exact_match_sum": sum(exact_matches),
        "exact_match_mean": _mean_or_none(exact_matches),
        "missing_score_count": len(rows) - len(scored_rows),
    }
    post_summary = _post_think_score_summary(rows)
    if post_summary:
        summary.update(post_summary)
    return summary


def _post_think_score_summary(rows: list[JsonDict]) -> JsonDict:
    scored_rows = [
        row
        for row in rows
        if "post_think_correct" in row["score_payload"]
    ]
    if not scored_rows:
        return {}
    exact_matches = [
        float(row["score_payload"].get("post_think_exact_match"))
        for row in rows
        if _is_number(row["score_payload"].get("post_think_exact_match"))
    ]
    n_correct = sum(
        bool(row["score_payload"].get("post_think_correct"))
        for row in scored_rows
    )
    return {
        "post_think_n_scored": len(scored_rows),
        "post_think_n_correct": n_correct,
        "post_think_accuracy": n_correct / len(scored_rows),
        "post_think_exact_match_sum": sum(exact_matches),
        "post_think_exact_match_mean": _mean_or_none(exact_matches),
        "post_think_missing_score_count": len(rows) - len(scored_rows),
    }


def _hit_rate(rows: list[JsonDict]) -> float | None:
    available = [row for row in rows if row["hit_max_new_tokens"] is not None]
    if not available:
        return None
    return sum(row["hit_max_new_tokens"] is True for row in available) / len(available)


def _entropy_summary_from_rows(rows: list[JsonDict]) -> JsonDict:
    combined = {
        name: {"entropy_sum": 0.0, "token_count": 0}
        for name in ENTROPY_STAT_NAMES
    }
    found_entropy = False
    for row in rows:
        method_stats = row.get("method_stats")
        if not isinstance(method_stats, dict):
            continue
        entropy = method_stats.get("entropy")
        if not isinstance(entropy, dict):
            continue
        found_entropy = True
        for name in ENTROPY_STAT_NAMES:
            item = entropy.get(name)
            if not isinstance(item, dict):
                continue
            entropy_sum = item.get("entropy_sum")
            token_count = item.get("token_count")
            if _is_number(entropy_sum):
                combined[name]["entropy_sum"] += float(entropy_sum)
            if _is_number(token_count):
                combined[name]["token_count"] += int(token_count)

    if not found_entropy:
        return {}

    payload: JsonDict = {
        "version": 1,
        "unit": ENTROPY_UNIT,
        "distribution": ENTROPY_DISTRIBUTION,
    }
    for name in ENTROPY_STAT_NAMES:
        entropy_sum = float(combined[name]["entropy_sum"])
        token_count = int(combined[name]["token_count"])
        payload[name] = {
            "entropy_sum": entropy_sum,
            "token_count": token_count,
            "mean_entropy": entropy_sum / token_count if token_count else None,
        }
    return payload


def _method_summary_from_debug(debug_payload: JsonDict | None) -> JsonDict:
    if debug_payload is None:
        return {
            "coverage": {
                "vllm_metrics_debug_present": False,
                "measured_vllm_deltas_present": False,
            },
            "debug": {"source_metrics": []},
        }

    measured_window = debug_payload.get("measured_window")
    deltas = {}
    if isinstance(measured_window, dict):
        raw_deltas = measured_window.get("metric_deltas")
        if isinstance(raw_deltas, dict):
            deltas = raw_deltas

    method_summary: JsonDict = {
        "coverage": {
            "vllm_metrics_debug_present": True,
            "measured_vllm_deltas_present": bool(deltas),
        },
        "debug": {
            "source_metrics": list(debug_payload.get("metrics") or []),
        },
    }
    if isinstance(measured_window, dict):
        method_summary["debug"]["source_metric_snapshots"] = {
            "measured_window_start": list(
                measured_window.get("start_metrics") or []
            ),
            "measured_window_end": list(measured_window.get("end_metrics") or []),
        }
        method_summary["debug"]["measured_window_coverage"] = dict(
            measured_window.get("coverage") or {}
        )
        method_summary["debug"]["measured_window_source_note"] = (
            measured_window.get("source_note")
        )
    if deltas:
        method_summary.update(_canonical_method_fields_from_deltas(deltas))
    return method_summary


def _canonical_method_fields_from_deltas(deltas: dict[str, Any]) -> JsonDict:
    num_drafts = int(deltas.get("delta_num_drafts") or 0)
    num_draft_tokens = int(deltas.get("delta_num_draft_tokens") or 0)
    num_accepted_tokens = int(deltas.get("delta_num_accepted_tokens") or 0)
    accepted_per_pos = _int_list(deltas.get("delta_num_accepted_tokens_per_pos"))
    num_bonus_activations = int(deltas.get("delta_num_bonus_activations") or 0)
    num_target_bonus_activations = int(
        deltas.get("delta_num_target_bonus_activations") or 0
    )
    num_relaxed_bonus_activations = int(
        deltas.get("delta_num_relaxed_bonus_activations") or 0
    )
    payload: JsonDict = {
        "delta_num_drafts": num_drafts,
        "delta_num_draft_tokens": num_draft_tokens,
        "delta_num_accepted_tokens": num_accepted_tokens,
        "delta_num_accepted_tokens_per_pos": accepted_per_pos,
        "delta_num_bonus_activations": num_bonus_activations,
        "delta_num_target_bonus_activations": num_target_bonus_activations,
        "delta_num_relaxed_bonus_activations": num_relaxed_bonus_activations,
        "acceptance_length_convention": "vllm_bonus_inclusive",
    }
    if num_draft_tokens > 0:
        payload["draft_acceptance_rate"] = num_accepted_tokens / num_draft_tokens
    if num_drafts > 0:
        mean_accepted = num_accepted_tokens / num_drafts
        payload["mean_accepted_draft_tokens"] = mean_accepted
        payload["mean_acceptance_length_with_bonus"] = 1.0 + mean_accepted
        payload["per_position_acceptance_rates"] = [
            value / num_drafts for value in accepted_per_pos
        ]
        payload["bonus_activation_rate"] = num_bonus_activations / num_drafts
        payload["target_bonus_activation_rate"] = (
            num_target_bonus_activations / num_drafts
        )
        payload["relaxed_bonus_activation_rate"] = (
            num_relaxed_bonus_activations / num_drafts
        )
    return payload


def _combine_method_summaries(method_summaries: list[JsonDict]) -> JsonDict:
    combined = {
        "delta_num_drafts": 0,
        "delta_num_draft_tokens": 0,
        "delta_num_accepted_tokens": 0,
        "delta_num_accepted_tokens_per_pos": [],
        "delta_num_bonus_activations": 0,
        "delta_num_target_bonus_activations": 0,
        "delta_num_relaxed_bonus_activations": 0,
    }
    source_metrics = []
    runs_with_deltas = 0
    for summary in method_summaries:
        source_metrics.extend(summary.get("debug", {}).get("source_metrics", []))
        if not summary.get("coverage", {}).get("measured_vllm_deltas_present"):
            continue
        runs_with_deltas += 1
        combined["delta_num_drafts"] += int(summary.get("delta_num_drafts") or 0)
        combined["delta_num_draft_tokens"] += int(
            summary.get("delta_num_draft_tokens") or 0
        )
        combined["delta_num_accepted_tokens"] += int(
            summary.get("delta_num_accepted_tokens") or 0
        )
        combined["delta_num_accepted_tokens_per_pos"] = _vector_sum(
            combined["delta_num_accepted_tokens_per_pos"],
            _int_list(summary.get("delta_num_accepted_tokens_per_pos")),
        )
        combined["delta_num_bonus_activations"] += int(
            summary.get("delta_num_bonus_activations") or 0
        )
        combined["delta_num_target_bonus_activations"] += int(
            summary.get("delta_num_target_bonus_activations") or 0
        )
        combined["delta_num_relaxed_bonus_activations"] += int(
            summary.get("delta_num_relaxed_bonus_activations") or 0
        )

    payload: JsonDict = {
        "coverage": {
            "vllm_metrics_debug_present": bool(source_metrics),
            "measured_vllm_deltas_present": runs_with_deltas > 0,
            "runs_with_measured_vllm_deltas": runs_with_deltas,
            "runs_without_measured_vllm_deltas": (
                len(method_summaries) - runs_with_deltas
            ),
        },
        "debug": {"source_metrics": source_metrics},
    }
    if runs_with_deltas:
        payload.update(_canonical_method_fields_from_deltas(combined))
    return payload


def _spec_metric_totals(metrics: object) -> dict[str, int | list[int]]:
    totals: dict[str, int | list[int]] = {}
    if not isinstance(metrics, list):
        return totals

    for metric in metrics:
        if not isinstance(metric, dict):
            continue
        name = metric.get("name")
        field = VLLM_SPEC_METRIC_NAMES.get(name)
        if field is None:
            continue
        if field == "num_accepted_tokens_per_pos":
            totals[field] = _vector_sum(
                _int_list(totals.get(field)),
                _int_list(metric.get("values")),
            )
            continue
        value = metric.get("value")
        if _is_number(value):
            totals[field] = int(totals.get(field, 0)) + int(value)
    return totals


def _summary_to_multi_run_item(
    summary: RunSummary,
    run_dir: str | Path,
    manifest: RunManifest,
) -> JsonDict:
    return {
        "run_id": summary.run_id,
        "run_dir": str(run_dir),
        "experiment_group": manifest.config.metadata.experiment_group,
        "task_name": summary.task_name,
        "decode_mode": summary.decode_mode.value,
        "model": summary.model,
        "draft_model": summary.draft_model,
        "n_samples": summary.n_samples,
        "n_generate_calls": summary.n_generate_calls,
        "total_prompt_tokens": summary.total_prompt_tokens,
        "total_output_tokens": summary.total_output_tokens,
        "total_generate_wall_time_s": summary.total_generate_wall_time_s,
        "accuracy": summary.score_summary.get("accuracy"),
        "mean_output_tokens": summary.efficiency_summary.get(
            "mean_output_tokens"
        ),
        "std_output_tokens": summary.efficiency_summary.get("std_output_tokens"),
        "mean_generate_wall_time_s": summary.efficiency_summary.get(
            "mean_generate_wall_time_s"
        ),
        "std_generate_wall_time_s": summary.efficiency_summary.get(
            "std_generate_wall_time_s"
        ),
        "output_tokens_per_s": summary.efficiency_summary.get(
            "output_tokens_per_s"
        ),
        "attempt_ids": summary.efficiency_summary.get("attempt_ids", []),
        "repeat_indices": summary.efficiency_summary.get("repeat_indices", []),
        "seeds": summary.efficiency_summary.get("seeds", []),
        "method_summary": dict(summary.method_summary),
    }


def _run_dimension_summary(per_run: list[JsonDict]) -> JsonDict:
    fields = [
        "accuracy",
        "mean_output_tokens",
        "mean_generate_wall_time_s",
        "output_tokens_per_s",
    ]
    payload: JsonDict = {}
    for field in fields:
        values = [
            run[field]
            for run in per_run
            if _is_number(run.get(field))
        ]
        payload[field] = {
            "mean": _mean_or_none(values),
            "std": _std_or_zero(values),
            "n": len(values),
        }
    return payload


def _repeat_breakdown(rows: list[JsonDict]) -> JsonDict:
    return {
        "sample_count_by_run_id": _count_by_field(rows, "run_id"),
        "sample_count_by_attempt": _count_by_metadata(rows, "attempt_id"),
        "sample_count_by_repeat_index": _count_by_metadata(rows, "repeat_index"),
        "sample_count_by_repeat_batch_index": _count_by_metadata(
            rows,
            "repeat_batch_index",
        ),
        "sample_count_by_seed": _count_by_metadata(rows, "seed"),
    }


def _summary_debug_refs(run_dir: Path, debug_payload: JsonDict | None) -> dict[str, str]:
    if debug_payload is None:
        return {}
    debug_path = run_dir / RUN_ARTIFACT_PATHS["debug_dir"].rstrip("/")
    return {"vllm_metrics": str((debug_path / "vllm_metrics.json").relative_to(run_dir))}


def _unique_metadata_values(rows: list[JsonDict], key: str) -> list[Any]:
    values = []
    seen = set()
    for row in rows:
        value = row["metadata"].get(key)
        if value is None:
            continue
        marker = json.dumps(value, sort_keys=True)
        if marker in seen:
            continue
        seen.add(marker)
        values.append(value)
    return values


def _count_by_metadata(rows: list[JsonDict], key: str) -> JsonDict:
    return _string_count(row["metadata"].get(key) for row in rows)


def _count_by_field(rows: list[JsonDict], key: str) -> JsonDict:
    return _string_count(row.get(key) for row in rows)


def _string_count(values) -> JsonDict:
    counts = Counter(_label(value) for value in values)
    return {key: counts[key] for key in sorted(counts)}


def _label(value: Any) -> str:
    if value is None:
        return "missing"
    return str(value)


def _mean_or_zero(values: list[int | float]) -> float:
    return float(mean(values)) if values else 0.0


def _mean_or_none(values: list[int | float]) -> float | None:
    return float(mean(values)) if values else None


def _std_or_zero(values: list[int | float]) -> float:
    return float(stdev(values)) if len(values) > 1 else 0.0


def _tokens_per_s(tokens: int, wall_time_s: float | None) -> float | None:
    if wall_time_s is None or wall_time_s <= 0:
        return None
    return tokens / wall_time_s


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _int_list(value: Any) -> list[int]:
    if not isinstance(value, list):
        return []
    return [int(item) for item in value if _is_number(item)]


def _vector_sum(left: list[int], right: list[int]) -> list[int]:
    width = max(len(left), len(right))
    return [
        (left[index] if index < len(left) else 0)
        + (right[index] if index < len(right) else 0)
        for index in range(width)
    ]


def _vector_delta(left: Any, right: Any) -> list[int]:
    left_values = _int_list(left)
    right_values = _int_list(right)
    width = max(len(left_values), len(right_values))
    return [
        (right_values[index] if index < len(right_values) else 0)
        - (left_values[index] if index < len(left_values) else 0)
        for index in range(width)
    ]


__all__ = [
    "build_vllm_measured_window",
    "reduce_multi_run_summary",
    "reduce_run_summary",
    "regenerate_run_summary",
    "write_multi_run_summary",
]
