"""Small helpers for retrying offline runs with a lower batch size after OOM."""

from __future__ import annotations

import gc
import os
from dataclasses import dataclass
from pathlib import Path

from ens_spec_dec.artifacts import RUN_ARTIFACT_PATHS, read_manifest
from ens_spec_dec.contracts import RunConfig


@dataclass(frozen=True, slots=True)
class OomFallbackConfig:
    path: Path
    run_id: str
    original_batch_size: int
    fallback_batch_size: int


def is_oom_error(exc: Exception, context: str | None = None) -> bool:
    message = _exception_chain_text(exc)
    if _contains_oom_text(message):
        return True
    if _is_engine_dead_error(message) and _contains_oom_text(context or ""):
        return True
    return False


def env_int(name: str) -> int | None:
    raw_value = os.environ.get(name)
    if raw_value in (None, ""):
        return None
    return int(raw_value)


def write_oom_fallback_config(
    config_path: str | Path,
    fallback_batch_size: int | None,
) -> OomFallbackConfig | None:
    source_path = Path(config_path)
    config = RunConfig.from_yaml(source_path)
    original_batch_size = config.execution.batch_size
    repeat_batch_size = config.execution.repeat_batch_size
    if fallback_batch_size is None:
        fallback_batch_size = _default_fallback_batch_size(
            original_batch_size,
            repeat_batch_size,
        )

    if fallback_batch_size < 1:
        raise ValueError("fallback_batch_size must be >= 1")
    if fallback_batch_size >= original_batch_size:
        return None

    if fallback_batch_size < repeat_batch_size:
        raise ValueError(
            "OOM fallback batch size must be >= execution.repeat_batch_size"
        )
    if fallback_batch_size % repeat_batch_size != 0:
        raise ValueError(
            "OOM fallback batch size must be divisible by execution.repeat_batch_size"
        )

    config.execution.batch_size = fallback_batch_size
    _lower_max_num_seqs(config, fallback_batch_size)
    _retag_batch_size(config, original_batch_size, fallback_batch_size)
    config.metadata.debug_refs["oom_fallback"] = (
        f"execution.batch_size {original_batch_size} -> {fallback_batch_size}"
    )
    config.debug_refs["oom_fallback_source_config"] = str(source_path)

    fallback_path = source_path.with_name(
        f"{source_path.stem}.oom-bs{fallback_batch_size}{source_path.suffix}"
    )
    config.to_yaml(fallback_path)
    return OomFallbackConfig(
        path=fallback_path,
        run_id=config.metadata.run_id,
        original_batch_size=original_batch_size,
        fallback_batch_size=fallback_batch_size,
    )


def prepare_run_dir_for_oom_retry(runs_root: str | Path, run_id: str) -> None:
    run_dir = Path(runs_root) / run_id
    if not run_dir.exists():
        return

    summary_path = run_dir / RUN_ARTIFACT_PATHS["summary"]
    if summary_path.exists():
        raise ValueError(
            "refusing OOM retry because the target run already has summary.json"
        )

    manifest_path = run_dir / RUN_ARTIFACT_PATHS["manifest"]
    if manifest_path.exists() and read_manifest(run_dir).run_id != run_id:
        raise ValueError("refusing OOM retry because manifest run_id mismatches")

    samples_path = run_dir / RUN_ARTIFACT_PATHS["samples"]
    if samples_path.exists():
        samples_path.unlink()


def clear_cuda_cache_after_oom() -> None:
    gc.collect()
    try:
        import torch
    except Exception:
        return

    try:
        torch.cuda.empty_cache()
    except Exception:
        return


def _lower_max_num_seqs(config: RunConfig, fallback_batch_size: int) -> None:
    max_num_seqs = config.backend.options.get("max_num_seqs")
    if isinstance(max_num_seqs, bool) or not isinstance(max_num_seqs, int):
        return
    if max_num_seqs > fallback_batch_size:
        config.backend.options["max_num_seqs"] = fallback_batch_size


def _default_fallback_batch_size(
    original_batch_size: int,
    repeat_batch_size: int,
) -> int:
    if original_batch_size <= 1:
        return original_batch_size

    half = max(1, original_batch_size // 2)
    if repeat_batch_size <= 1:
        return half

    fallback = (half // repeat_batch_size) * repeat_batch_size
    return max(repeat_batch_size, fallback)


def _retag_batch_size(
    config: RunConfig,
    original_batch_size: int,
    fallback_batch_size: int,
) -> None:
    original_tag = f"bs{original_batch_size}"
    fallback_tag = f"bs{fallback_batch_size}"
    tags = [tag for tag in config.metadata.tags if tag != original_tag]
    for tag in (fallback_tag, f"oom_fallback_from_{original_tag}"):
        if tag not in tags:
            tags.append(tag)
    config.metadata.tags = tags


def _exception_chain_text(exc: BaseException) -> str:
    parts: list[str] = []
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        parts.append(f"{current.__class__.__name__}: {current}")
        notes = getattr(current, "__notes__", None)
        if isinstance(notes, list):
            parts.extend(str(note) for note in notes)
        current = current.__cause__ or current.__context__
    return "\n".join(parts).lower()


def _contains_oom_text(text: str) -> bool:
    lowered = text.lower()
    return (
        "out of memory" in lowered
        or "cuda oom" in lowered
        or "torch.outofmemoryerror" in lowered
        or "cudamemoryallocation" in lowered
    )


def _is_engine_dead_error(text: str) -> bool:
    return "enginedeaderror" in text or "enginecore encountered an issue" in text


__all__ = [
    "OomFallbackConfig",
    "clear_cuda_cache_after_oom",
    "env_int",
    "is_oom_error",
    "prepare_run_dir_for_oom_retry",
    "write_oom_fallback_config",
]
