"""Component-profile artifacts for proxy-speed analysis.

Profiles are reusable JSON cost tables. Offline generation runs produce
quality, lengths, and acceptance counters; these profile artifacts provide the
hardware/backend costs that proxy-speed analysis joins with those runs.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ens_spec_dec.contracts import JsonDict, RunConfig

COMPONENT_PROFILE_SCHEMA_VERSION = 1
COMPONENT_PROFILE_KIND = "component_profile"
DRAFTER_MODEL_PROFILE_SUBJECT = "drafter_model"

OPTIMISTIC_AR_DRAFTER = "optimistic_ar_drafter"
STOCK_INSIDE_SD_ROUND = "stock_inside_sd_round"
STOCK_INSIDE_SD_DECOMPOSED = "stock_inside_sd_decomposed"


def build_drafter_model_profile(
    *,
    profile_id: str,
    config: RunConfig,
    hardware: JsonDict,
    software: JsonDict,
    profile_mode: str,
    draft_lengths: list[int],
    costs_ms: JsonDict,
    measurement: JsonDict | None = None,
    warnings: list[str] | None = None,
    created_at_utc: str | None = None,
) -> JsonDict:
    """Build a stable vanilla drafter-model component profile."""

    costs = dict(costs_ms)
    profile_warnings = list(warnings or [])
    _derive_drafter_inside_sd_effective(costs, profile_warnings)
    derived_modes = derive_cost_modes(costs, draft_lengths)
    target_ar = _number_or_none(costs.get("target_ar_token"))

    return {
        "schema_version": COMPONENT_PROFILE_SCHEMA_VERSION,
        "artifact_kind": COMPONENT_PROFILE_KIND,
        "profile_subject": DRAFTER_MODEL_PROFILE_SUBJECT,
        "profile_id": profile_id,
        "created_at_utc": created_at_utc or _utc_now(),
        "compatibility": {
            "backend_name": config.backend.name,
            "target_model": config.backend.model,
            "draft_model": config.backend.draft_model,
            "dtype": config.backend.dtype,
            "batch_size": config.execution.batch_size,
            "tensor_parallel_size": config.execution.tensor_parallel_size,
            "draft_lengths": list(draft_lengths),
            "max_model_len": config.backend.options.get("max_model_len"),
            "profile_mode": profile_mode,
        },
        "hardware": dict(hardware),
        "software": dict(software),
        "costs_ms": costs,
        "relative_costs": _relative_costs(costs, draft_lengths, target_ar),
        "derived_cost_modes": derived_modes,
        "measurement": dict(measurement or {}),
        "warnings": profile_warnings,
    }


def derive_cost_modes(costs_ms: JsonDict, draft_lengths: list[int]) -> JsonDict:
    """Return derived round-cost modes from raw profile costs."""

    modes: JsonDict = {}
    target_verify = _float_map(costs_ms.get("target_verify_block_by_d"))
    stock_round = _float_map(costs_ms.get("stock_sd_round_by_d"))
    inside_effective = _float_map(
        costs_ms.get("drafter_inside_sd_effective_by_d")
    )
    target_ar = _number_or_none(costs_ms.get("target_ar_token"))
    drafter_ar = _number_or_none(costs_ms.get("drafter_ar_token"))

    optimistic: dict[str, float] = {}
    if target_ar is not None and drafter_ar is not None:
        for draft_length in draft_lengths:
            optimistic[str(draft_length)] = target_ar + draft_length * drafter_ar
    if optimistic:
        modes[OPTIMISTIC_AR_DRAFTER] = {
            "round_cost_ms_by_d": optimistic,
            "source": "target_ar_token + d * drafter_ar_token",
            "assumption": (
                "target verifier latency is memory-bound, so checking a "
                "parallel token block has the same latency as one verifier token"
            ),
        }

    if stock_round:
        modes[STOCK_INSIDE_SD_ROUND] = {
            "round_cost_ms_by_d": stock_round,
            "source": "measured stock SD wall time divided by measured drafts",
        }

    decomposed: dict[str, float] = {}
    for draft_length in draft_lengths:
        key = str(draft_length)
        verify = target_verify.get(key)
        inside = inside_effective.get(key)
        if verify is not None and inside is not None:
            decomposed[key] = verify + inside
    if decomposed:
        modes[STOCK_INSIDE_SD_DECOMPOSED] = {
            "round_cost_ms_by_d": decomposed,
            "source": "target_verify_block_by_d + drafter_inside_sd_effective_by_d",
        }
    return modes


def read_component_profile(path: str | Path) -> JsonDict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_component_profile(payload)
    return payload


def write_component_profile(path: str | Path, profile: JsonDict) -> None:
    validate_component_profile(profile)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(profile, indent=2) + "\n", encoding="utf-8")


def validate_component_profile(profile: JsonDict) -> None:
    required = (
        "schema_version",
        "artifact_kind",
        "profile_subject",
        "profile_id",
        "compatibility",
        "costs_ms",
        "derived_cost_modes",
    )
    missing = [key for key in required if key not in profile]
    if missing:
        raise ValueError(f"component profile missing required keys: {missing}")
    if profile["schema_version"] != COMPONENT_PROFILE_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported component profile schema: {profile['schema_version']}"
        )
    if profile["artifact_kind"] != COMPONENT_PROFILE_KIND:
        raise ValueError(f"not a component profile: {profile['artifact_kind']}")


def _derive_drafter_inside_sd_effective(
    costs: JsonDict,
    warnings: list[str] | None,
) -> None:
    target_verify = _float_map(costs.get("target_verify_block_by_d"))
    stock_round = _float_map(costs.get("stock_sd_round_by_d"))
    if not stock_round or not target_verify:
        return

    effective: dict[str, float] = {}
    for key, stock_ms in stock_round.items():
        verify_ms = target_verify.get(key)
        if verify_ms is None:
            continue
        raw_effective = stock_ms - verify_ms
        if raw_effective < 0:
            if warnings is not None:
                warnings.append(
                    f"d={key}: stock SD round cost is below verifier-block cost; "
                    "clamped drafter_inside_sd_effective to 0 for reporting"
                )
            raw_effective = 0.0
        effective[key] = raw_effective
    if effective:
        costs["drafter_inside_sd_effective_by_d"] = effective


def _relative_costs(
    costs: JsonDict,
    draft_lengths: list[int],
    target_ar: float | None,
) -> JsonDict:
    if target_ar is None or target_ar <= 0:
        return {}

    relative: JsonDict = {}
    drafter_ar = _number_or_none(costs.get("drafter_ar_token"))
    if drafter_ar is not None:
        relative["drafter_ar_token_over_target_ar"] = drafter_ar / target_ar

    for field in (
        "target_verify_block_by_d",
        "stock_sd_round_by_d",
        "drafter_inside_sd_effective_by_d",
    ):
        values = _float_map(costs.get(field))
        if not values:
            continue
        relative[field.replace("_by_d", "_over_target_ar_by_d")] = {
            str(d): values[str(d)] / target_ar
            for d in draft_lengths
            if str(d) in values
        }
    return relative


def _float_map(value: Any) -> dict[str, float]:
    if not isinstance(value, dict):
        return {}
    result: dict[str, float] = {}
    for key, item in value.items():
        number = _number_or_none(item)
        if number is not None:
            result[str(key)] = number
    return result


def _number_or_none(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


__all__ = [
    "COMPONENT_PROFILE_KIND",
    "COMPONENT_PROFILE_SCHEMA_VERSION",
    "DRAFTER_MODEL_PROFILE_SUBJECT",
    "OPTIMISTIC_AR_DRAFTER",
    "STOCK_INSIDE_SD_DECOMPOSED",
    "STOCK_INSIDE_SD_ROUND",
    "build_drafter_model_profile",
    "derive_cost_modes",
    "read_component_profile",
    "validate_component_profile",
    "write_component_profile",
]
