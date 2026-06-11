"""Minimal offline runner for milestone-1 backend evaluation runs.

This module loads one run config, resolves task prompts, optionally applies a
model chat template, runs optional warmup, calls the backend in measured
attempts, parses and scores outputs, and writes fixed-layout run artifacts plus
the stable artifact-derived summary.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from ens_spec_dec.artifacts import (
    append_scored_samples,
    init_run_dir,
    write_debug_json,
)
from ens_spec_dec.contracts import (
    DecodeMode,
    GenerationParams,
    GenerationResult,
    JsonDict,
    MeasurementScope,
    MethodConfig,
    RunConfig,
    SampleInput,
    SamplingConfig,
    ScoredSample,
    decode_mode_requires_draft_model,
    is_speculative_decode_mode,
)
from ens_spec_dec.engines import BackendRequestConfig, GenerationBackend
from ens_spec_dec.evaluation import (
    build_vllm_measured_window,
    regenerate_run_summary,
)
from ens_spec_dec.tasks import aime_2024, gpqa_diamond, livecodebench_lite_v6

TaskModule = object
BackendFactory = Callable[[BackendRequestConfig], GenerationBackend]

TASK_MODULES: dict[str, TaskModule] = {
    "aime_2024": aime_2024,
    "gpqa_diamond": gpqa_diamond,
    "livecodebench_lite_v6": livecodebench_lite_v6,
}
NO_WARMUP_TIMING_TAG = "no_warmup_timing"


def run_offline_from_yaml(
    config_path: str | Path,
    runs_root: str | Path = "runs",
    run_id: str | None = None,
    backend_factory: BackendFactory | None = None,
) -> Path:
    config = RunConfig.from_yaml(config_path)
    if run_id is not None:
        config.metadata.run_id = run_id
    return run_offline(config, runs_root=runs_root, backend_factory=backend_factory)


def run_offline(
    config: RunConfig,
    runs_root: str | Path = "runs",
    backend_factory: BackendFactory | None = None,
) -> Path:
    run_config = _materialize_runner_config(config)
    _validate_run_config(run_config)
    task_module = _get_task_module(run_config.task.name)
    resolved_generation_params = resolve_generation_params(run_config)
    backend_config = BackendRequestConfig(
        decode_mode=run_config.metadata.decode_mode,
        backend=run_config.backend,
        execution=run_config.execution,
        generation_params=resolved_generation_params,
    )
    if backend_factory is None:
        # Delayed so multi-GPU workers can set CUDA_VISIBLE_DEVICES first.
        from ens_spec_dec.engines.vllm_offline import OfflineVllmBackend

        # OfflineVllmBackend constructs the real vLLM LLM in __init__.
        backend = OfflineVllmBackend(backend_config)
    else:
        backend = backend_factory(backend_config)

    samples = task_module.load_samples(run_config.task)
    prompts = [task_module.build_prompt(sample) for sample in samples]
    formatted_prompts = format_samples_for_backend(samples, prompts, run_config)

    run_dir = init_run_dir(runs_root, run_config)
    attempts = _attempts_for_run(run_config, resolved_generation_params)
    warmup_records = _run_warmup(
        backend=backend,
        config=run_config,
        samples=samples,
        formatted_prompts=formatted_prompts,
        generation_params=_warmup_generation_params(
            attempts[0]["generation_params"],
            run_config,
        ),
        attempt=attempts[0],
    )
    # Warmup exercises the backend but is excluded from stable samples.jsonl.
    if run_config.execution.record_warmup_debug:
        write_debug_json(run_dir, "warmup_samples.json", warmup_records)

    measured_start_debug_payload = backend.get_run_debug_payload()
    unique_sample_batch_size = _unique_sample_batch_size(run_config)
    attempt_groups = _attempt_groups_for_run(run_config, attempts)
    # Only measured batches are scored and appended to stable samples.jsonl.
    for repeat_batch_index, attempt_group in enumerate(attempt_groups):
        for start in range(0, len(samples), unique_sample_batch_size):
            batch_samples = samples[start : start + unique_sample_batch_size]
            batch_prompts = formatted_prompts[
                start : start + unique_sample_batch_size
            ]
            (
                repeat_batch_samples,
                repeat_batch_prompts,
                sample_metadata,
            ) = _repeat_batch_inputs(
                samples=batch_samples,
                formatted_prompts=batch_prompts,
                attempts=attempt_group,
                repeat_batch_index=repeat_batch_index,
                repeat_batch_size=run_config.execution.repeat_batch_size,
            )
            generation_params = _repeat_batch_generation_params(
                attempts=attempt_group,
                unique_sample_count=len(batch_samples),
            )
            batch_result = backend.generate_batch(
                repeat_batch_prompts,
                generation_params,
            )
            scored_samples = _score_batch(
                config=run_config,
                task_module=task_module,
                samples=repeat_batch_samples,
                batch_result=batch_result,
                generation_params=generation_params,
                sample_metadata=sample_metadata,
            )
            append_scored_samples(run_dir, scored_samples)

    measured_end_debug_payload = backend.get_run_debug_payload()
    debug_payload = _runner_debug_payload(
        backend_payload=measured_end_debug_payload,
        measured_start_payload=measured_start_debug_payload,
        measured_end_payload=measured_end_debug_payload,
        config=run_config,
        attempts=attempts,
        attempt_groups=attempt_groups,
        n_warmup_records=len(warmup_records),
    )
    write_debug_json(run_dir, "vllm_metrics.json", debug_payload)
    regenerate_run_summary(run_dir)
    return run_dir


def resolve_generation_params(config: RunConfig) -> GenerationParams:
    defaults = config.task.defaults.generation_params
    current = config.generation_params
    stop = current.sampling.stop or defaults.sampling.stop
    params = dict(defaults.method.params)
    params.update(current.method.params)
    return GenerationParams(
        sampling=SamplingConfig(
            temperature=(
                current.sampling.temperature
                if current.sampling.temperature is not None
                else defaults.sampling.temperature
            ),
            top_p=(
                current.sampling.top_p
                if current.sampling.top_p is not None
                else defaults.sampling.top_p
            ),
            top_k=(
                current.sampling.top_k
                if current.sampling.top_k is not None
                else defaults.sampling.top_k
            ),
            presence_penalty=(
                current.sampling.presence_penalty
                if current.sampling.presence_penalty is not None
                else defaults.sampling.presence_penalty
            ),
            max_new_tokens=(
                current.sampling.max_new_tokens
                if current.sampling.max_new_tokens is not None
                else defaults.sampling.max_new_tokens
            ),
            thinking_budget_tokens=(
                current.sampling.thinking_budget_tokens
                if current.sampling.thinking_budget_tokens is not None
                else defaults.sampling.thinking_budget_tokens
            ),
            stop=list(stop),
            seed=(
                current.sampling.seed
                if current.sampling.seed is not None
                else defaults.sampling.seed
            ),
        ),
        method=MethodConfig(
            name=current.method.name or defaults.method.name,
            variant=current.method.variant or defaults.method.variant,
            method_version=(
                current.method.method_version or defaults.method.method_version
            ),
            draft_length=(
                current.method.draft_length
                if current.method.draft_length is not None
                else defaults.method.draft_length
            ),
            params=params,
        ),
    )


def resolve_sampling_config(config: RunConfig) -> SamplingConfig:
    return resolve_generation_params(config).sampling


def resolve_method_config(config: RunConfig) -> MethodConfig:
    return resolve_generation_params(config).method


def format_prompts_for_backend(prompts: list[str], config: RunConfig) -> list[str]:
    return format_samples_for_backend([], prompts, config)


def format_samples_for_backend(
    samples: list[SampleInput],
    prompts: list[str],
    config: RunConfig,
) -> list[str]:
    prompt_format = config.backend.options.get("prompt_format", "raw")
    if prompt_format == "raw":
        return list(prompts)
    if prompt_format != "chat_model_template":
        raise ValueError(f"unsupported backend.options.prompt_format: {prompt_format}")

    tokenizer = _load_chat_tokenizer(config)
    chat_template_kwargs = config.backend.options.get("chat_template_kwargs") or {}
    if not isinstance(chat_template_kwargs, dict):
        raise ValueError("backend.options.chat_template_kwargs must be a mapping")

    messages_by_prompt = _messages_for_backend(samples, prompts)
    return [
        tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **chat_template_kwargs,
        )
        for messages in messages_by_prompt
    ]


def _messages_for_backend(
    samples: list[SampleInput],
    prompts: list[str],
) -> list[list[JsonDict]]:
    messages_by_prompt: list[list[JsonDict]] = []
    for index, prompt in enumerate(prompts):
        messages = None
        if index < len(samples):
            candidate = samples[index].metadata.get("chat_messages")
            if isinstance(candidate, list):
                messages = candidate
        if messages is None:
            messages = [{"role": "user", "content": prompt}]
        messages_by_prompt.append([dict(message) for message in messages])
    return messages_by_prompt


def _load_chat_tokenizer(config: RunConfig):
    # Delayed to keep raw-prompt runs free of tokenizer/model loading.
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        config.backend.model,
        revision=config.backend.revision,
        trust_remote_code=True,
    )


def _materialize_runner_config(config: RunConfig) -> RunConfig:
    run_config = RunConfig.from_dict(config.to_dict())
    tags = run_config.metadata.tags
    if (
        not run_config.execution.warmup_enabled
        and NO_WARMUP_TIMING_TAG not in tags
    ):
        tags.append(NO_WARMUP_TIMING_TAG)
    return run_config


def _attempts_for_run(
    config: RunConfig,
    base_generation_params: GenerationParams,
) -> list[dict[str, object]]:
    if config.execution.seed_list:
        seeds: list[int | None] = list(config.execution.seed_list)
    else:
        seeds = _repeat_seeds(
            base_seed=base_generation_params.sampling.seed,
            repeat_count=config.execution.repeat_count,
            repeat_seed_offset=config.execution.repeat_seed_offset,
        )

    attempts: list[dict[str, object]] = []
    for index, seed in enumerate(seeds):
        attempts.append(
            {
                "attempt_id": f"attempt-{index + 1:04d}",
                "repeat_index": index,
                "seed": seed,
                "generation_params": _generation_params_with_seed(
                    base_generation_params,
                    seed,
                ),
            }
        )
    return attempts


def _repeat_seeds(
    base_seed: int | None,
    repeat_count: int,
    repeat_seed_offset: int,
) -> list[int | None]:
    if base_seed is None and repeat_count == 1 and repeat_seed_offset == 0:
        return [None] * repeat_count
    seed_base = 0 if base_seed is None else base_seed
    return [
        seed_base + repeat_seed_offset + repeat_index
        for repeat_index in range(repeat_count)
    ]


def _attempt_groups_for_run(
    config: RunConfig,
    attempts: list[dict[str, object]],
) -> list[list[dict[str, object]]]:
    repeat_batch_size = config.execution.repeat_batch_size
    return [
        attempts[start : start + repeat_batch_size]
        for start in range(0, len(attempts), repeat_batch_size)
    ]


def _unique_sample_batch_size(config: RunConfig) -> int:
    return config.execution.batch_size // config.execution.repeat_batch_size


def _repeat_batch_inputs(
    samples: list[SampleInput],
    formatted_prompts: list[str],
    attempts: list[dict[str, object]],
    repeat_batch_index: int,
    repeat_batch_size: int,
) -> tuple[list[SampleInput], list[str], list[JsonDict]]:
    repeat_batch_samples: list[SampleInput] = []
    repeat_batch_prompts: list[str] = []
    sample_metadata: list[JsonDict] = []
    for attempt in attempts:
        metadata = {
            "phase": "measured",
            "attempt_id": attempt["attempt_id"],
            "repeat_index": attempt["repeat_index"],
            "seed": attempt["seed"],
            "repeat_batch_index": repeat_batch_index,
            "repeat_batch_size": repeat_batch_size,
        }
        for sample, prompt in zip(samples, formatted_prompts, strict=True):
            repeat_batch_samples.append(sample)
            repeat_batch_prompts.append(prompt)
            sample_metadata.append(dict(metadata))
    return repeat_batch_samples, repeat_batch_prompts, sample_metadata


def _repeat_batch_generation_params(
    attempts: list[dict[str, object]],
    unique_sample_count: int,
) -> GenerationParams | list[GenerationParams]:
    if len(attempts) == 1:
        return attempts[0]["generation_params"]

    generation_params: list[GenerationParams] = []
    for attempt in attempts:
        params = attempt["generation_params"]
        generation_params.extend([params] * unique_sample_count)
    return generation_params


def _generation_params_with_seed(
    base_generation_params: GenerationParams,
    seed: int | None,
) -> GenerationParams:
    generation_params = GenerationParams.from_dict(base_generation_params.to_dict())
    generation_params.sampling.seed = seed
    return generation_params


def _warmup_generation_params(
    base_generation_params: GenerationParams,
    config: RunConfig,
) -> GenerationParams:
    generation_params = GenerationParams.from_dict(base_generation_params.to_dict())
    cap = config.execution.warmup_max_new_tokens
    max_new_tokens = generation_params.sampling.max_new_tokens
    generation_params.sampling.max_new_tokens = (
        cap if max_new_tokens is None else min(max_new_tokens, cap)
    )
    thinking_budget = generation_params.sampling.thinking_budget_tokens
    warmup_max_new_tokens = generation_params.sampling.max_new_tokens
    if thinking_budget is not None and warmup_max_new_tokens is not None:
        if warmup_max_new_tokens <= 1:
            generation_params.sampling.thinking_budget_tokens = None
        else:
            generation_params.sampling.thinking_budget_tokens = min(
                thinking_budget,
                warmup_max_new_tokens - 1,
            )
    return generation_params


def _run_warmup(
    backend: GenerationBackend,
    config: RunConfig,
    samples: list[SampleInput],
    formatted_prompts: list[str],
    generation_params: GenerationParams,
    attempt: dict[str, object],
) -> list[JsonDict]:
    if not config.execution.warmup_enabled or not samples:
        return []

    records: list[JsonDict] = []
    batch_size = config.execution.batch_size
    warmup_sample_count = min(config.execution.warmup_sample_count, len(samples))
    for start in range(0, warmup_sample_count, batch_size):
        stop = min(start + batch_size, warmup_sample_count)
        batch_samples = samples[start:stop]
        batch_prompts = formatted_prompts[start:stop]
        batch_result = backend.generate_batch(batch_prompts, generation_params)
        if len(batch_samples) != len(batch_result.outputs):
            raise ValueError("backend output count did not match warmup batch size")
        if config.execution.record_warmup_debug:
            records.extend(
                _warmup_debug_records(
                    samples=batch_samples,
                    batch_result=batch_result,
                    attempt=attempt,
                )
            )
    return records


def _warmup_debug_records(
    samples: list[SampleInput],
    batch_result,
    attempt: dict[str, object],
) -> list[JsonDict]:
    records: list[JsonDict] = []
    for sample, output in zip(samples, batch_result.outputs, strict=True):
        records.append(
            {
                "phase": "warmup",
                "sample_id": sample.sample_id,
                "attempt_id": attempt["attempt_id"],
                "repeat_index": attempt["repeat_index"],
                "seed": attempt["seed"],
                "generate_call_id": batch_result.generate_call_id,
                "generate_wall_time_s": batch_result.generate_wall_time_s,
                "generated_text": output.generated_text,
                "generated_token_ids": list(output.generated_token_ids),
                "n_prompt_tokens": output.n_prompt_tokens,
                "n_out_tokens": output.n_out_tokens,
            }
        )
    return records


def _runner_debug_payload(
    backend_payload: JsonDict,
    measured_start_payload: JsonDict,
    measured_end_payload: JsonDict,
    config: RunConfig,
    attempts: list[dict[str, object]],
    attempt_groups: list[list[dict[str, object]]],
    n_warmup_records: int,
) -> JsonDict:
    payload = dict(backend_payload)
    payload["measured_window"] = build_vllm_measured_window(
        measured_start_payload,
        measured_end_payload,
    )
    payload["runner_metadata"] = {
        "warmup_enabled": config.execution.warmup_enabled,
        "warmup_sample_count": (
            config.execution.warmup_sample_count
            if config.execution.warmup_enabled
            else 0
        ),
        "warmup_max_new_tokens": config.execution.warmup_max_new_tokens,
        "record_warmup_debug": config.execution.record_warmup_debug,
        "warmup_debug_record_count": n_warmup_records,
        "repeat_count": config.execution.repeat_count,
        "repeat_batch_size": config.execution.repeat_batch_size,
        "repeat_seed_offset": config.execution.repeat_seed_offset,
        "unique_sample_batch_size": _unique_sample_batch_size(config),
        "seed_list": list(config.execution.seed_list),
        "seeds": [attempt["seed"] for attempt in attempts],
        "measured_attempt_count": len(attempts),
        "measured_repeat_batch_count": len(attempt_groups),
        "attempt_ids": [attempt["attempt_id"] for attempt in attempts],
        "repeat_batch_attempt_ids": [
            [attempt["attempt_id"] for attempt in attempt_group]
            for attempt_group in attempt_groups
        ],
        "note": (
            "Top-level backend metrics are run-local debug diagnostics. "
            "Canonical speculative metrics should use measured_window "
            "deltas, which are snapshotted after warmup and after measured "
            "generation."
        ),
    }
    return payload


def _validate_run_config(config: RunConfig) -> None:
    if config.metadata.measurement_scope is not MeasurementScope.OFFLINE_EVAL:
        raise ValueError("M1.7 only supports measurement_scope=offline_eval")
    if config.execution.batch_size < 1:
        raise ValueError("execution.batch_size must be >= 1")
    if config.execution.repeat_count < 1:
        raise ValueError("execution.repeat_count must be >= 1")
    if config.execution.repeat_batch_size < 1:
        raise ValueError("execution.repeat_batch_size must be >= 1")
    if config.execution.repeat_seed_offset < 0:
        raise ValueError("execution.repeat_seed_offset must be >= 0")
    if config.execution.warmup_enabled and config.execution.warmup_sample_count < 1:
        raise ValueError("execution.warmup_sample_count must be >= 1")
    if config.execution.warmup_enabled and config.execution.warmup_max_new_tokens < 1:
        raise ValueError("execution.warmup_max_new_tokens must be >= 1")
    if config.execution.record_warmup_debug and not config.execution.warmup_enabled:
        raise ValueError("record_warmup_debug requires warmup_enabled")
    if config.execution.repeat_count > 1 and config.execution.seed_list:
        raise ValueError("repeat_count and seed_list cannot both multiply work")
    if config.execution.repeat_batch_size > 1:
        if config.execution.seed_list:
            raise ValueError("repeat_batch_size cannot be used with seed_list")
        if config.execution.repeat_count % config.execution.repeat_batch_size != 0:
            raise ValueError("repeat_count must be divisible by repeat_batch_size")
        if config.execution.batch_size < config.execution.repeat_batch_size:
            raise ValueError("execution.batch_size must be >= repeat_batch_size")
        if config.execution.batch_size % config.execution.repeat_batch_size != 0:
            raise ValueError("execution.batch_size must be divisible by repeat_batch_size")
    if _requires_draft_model(config) and not config.backend.draft_model:
        raise ValueError(
            f"{config.metadata.decode_mode.value} requires backend.draft_model"
        )
    if _uses_speculative_config(config):
        resolved_generation_params = resolve_generation_params(config)
        if resolved_generation_params.method.draft_length is None:
            raise ValueError(
                f"{config.metadata.decode_mode.value} requires method.draft_length"
            )
    resolved_sampling = resolve_sampling_config(config)
    if resolved_sampling.thinking_budget_tokens is not None:
        if resolved_sampling.thinking_budget_tokens < 1:
            raise ValueError("sampling.thinking_budget_tokens must be >= 1")
        if resolved_sampling.max_new_tokens is None:
            raise ValueError(
                "sampling.max_new_tokens is required with thinking_budget_tokens"
            )
        if resolved_sampling.max_new_tokens <= resolved_sampling.thinking_budget_tokens:
            raise ValueError(
                "sampling.max_new_tokens must be greater than "
                "sampling.thinking_budget_tokens"
            )


def _get_task_module(task_name: str):
    try:
        return TASK_MODULES[task_name]
    except KeyError as exc:
        raise ValueError(f"unsupported task adapter: {task_name}") from exc


def _uses_speculative_config(config: RunConfig) -> bool:
    return is_speculative_decode_mode(config.metadata.decode_mode)


def _requires_draft_model(config: RunConfig) -> bool:
    return decode_mode_requires_draft_model(config.metadata.decode_mode)


def _score_batch(
    config: RunConfig,
    task_module,
    samples: list[SampleInput],
    batch_result,
    generation_params: GenerationParams | list[GenerationParams],
    sample_metadata: list[JsonDict],
) -> list[ScoredSample]:
    if len(samples) != len(batch_result.outputs):
        raise ValueError("backend output count did not match batch size")
    if len(samples) != len(sample_metadata):
        raise ValueError("sample metadata count did not match batch size")
    params_by_output = _generation_params_for_outputs(
        generation_params,
        len(batch_result.outputs),
    )

    scored_samples: list[ScoredSample] = []
    for sample, output, params, metadata in zip(
        samples,
        batch_result.outputs,
        params_by_output,
        sample_metadata,
        strict=True,
    ):
        parsed_answer = task_module.parse_answer(output.generated_text)
        score_payload = _score_generation(
            task_module,
            sample.reference,
            output.generated_text,
            parsed_answer,
        )
        method_stats = dict(batch_result.aggregate_method_stats)
        method_stats.update(output.method_stats)
        generation = GenerationResult(
            generated_text=output.generated_text,
            generated_token_ids=list(output.generated_token_ids),
            parsed_answer=parsed_answer,
            n_prompt_tokens=output.n_prompt_tokens,
            n_out_tokens=output.n_out_tokens,
            generate_call_id=batch_result.generate_call_id,
            generate_wall_time_s=batch_result.generate_wall_time_s,
            hit_max_new_tokens=_hit_max_new_tokens(output, params),
            method_stats=method_stats,
        )
        scored_samples.append(
            ScoredSample(
                run_id=config.metadata.run_id,
                task_name=config.task.name,
                measurement_scope=config.metadata.measurement_scope,
                decode_mode=config.metadata.decode_mode,
                sample=sample,
                generation=generation,
                score_payload=score_payload,
                metadata=dict(metadata),
            )
        )
    return scored_samples


def _generation_params_for_outputs(
    generation_params: GenerationParams | list[GenerationParams],
    output_count: int,
) -> list[GenerationParams]:
    if isinstance(generation_params, list):
        if len(generation_params) != output_count:
            raise ValueError("generation params count did not match backend outputs")
        return generation_params
    return [generation_params] * output_count


def _score_generation(
    task_module,
    reference,
    generated_text: str,
    parsed_answer,
) -> JsonDict:
    score_generation = getattr(task_module, "score_generation", None)
    if callable(score_generation):
        return score_generation(reference, generated_text, parsed_answer)
    return task_module.score_answer(reference, parsed_answer)


def _hit_max_new_tokens(output, generation_params: GenerationParams) -> bool | None:
    max_new_tokens = generation_params.sampling.max_new_tokens
    if max_new_tokens is None or output.n_out_tokens is None:
        return None
    return output.n_out_tokens >= max_new_tokens


__all__ = [
    "TASK_MODULES",
    "format_prompts_for_backend",
    "format_samples_for_backend",
    "resolve_generation_params",
    "resolve_method_config",
    "resolve_sampling_config",
    "run_offline",
    "run_offline_from_yaml",
]
