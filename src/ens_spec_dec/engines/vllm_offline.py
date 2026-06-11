"""Offline vLLM wrapper for autoregressive and stock vLLM SD runs.

This module uses stock `vllm.LLM.generate(...)` only. It does not know about
benchmark parsing, scoring, or task-specific prompt conventions.
"""

from __future__ import annotations

import gc
import json
import math
import tempfile
import time
import uuid
from dataclasses import asdict
from typing import Any

# Importing this module imports vLLM, so callers keep this import delayed until
# process CUDA visibility is final.
from vllm import LLM, SamplingParams
from vllm.v1.metrics.reader import Counter, Gauge, Histogram, Metric, Vector

from ens_spec_dec.contracts import (
    DecodeMode,
    GenerationParams,
    JsonDict,
    SamplingConfig,
    decode_mode_requires_draft_model,
)
from ens_spec_dec.engines.backend import (
    BackendRequestConfig,
    BackendSampleOutput,
    GenerateBatchResult,
    GenerationParamsBatch,
)
from ens_spec_dec.fsd_naming import canonical_relaxed_target_method

PROFILE_MEASURED_OPTION = "profile_measured"
PROFILE_PREFIX_OPTION = "profile_prefix"
ENTROPY_MONITORING_OPTION = "entropy_monitoring"
ENTROPY_STAT_NAMES = (
    "drafter_accepted",
    "verifier_on_accepted",
    "verifier_on_output",
)
ENTROPY_UNIT = "nats"
ENTROPY_DISTRIBUTION = "processed_logits_after_temperature_top_p_top_k"
SPECULATIVE_METHOD_BY_DECODE_MODE = {
    DecodeMode.SD_VANILLA: "draft_model",
    DecodeMode.SD_RELAXED: "draft_model",
    DecodeMode.SD_DFLASH: "dflash",
    DecodeMode.SD_MTP: "mtp",
    DecodeMode.SD_MTP_RELAXED: "mtp",
    DecodeMode.SD_EAGLE3: "eagle3",
}
DRAFT_MODEL_STOCHASTIC_OPTIONS = {
    "rejection_sample_method": "probabilistic",
    "draft_sampling_method": "stochastic",
}
QWEN_THINKING_EARLY_STOP_TEXT = (
    "\n\n Considering the limited time by the user, I have to give the "
    "solution based on the thinking directly now.\n</think>\n\n"
)
QWEN_THINK_END_TEXT = "</think>"


class OfflineVllmBackend:
    def __init__(self, config: BackendRequestConfig):
        if config.backend.name != "vllm":
            raise ValueError(f"unsupported backend: {config.backend.name}")
        if config.execution.data_parallel_size not in (None, 1):
            raise ValueError("M1.6 offline vLLM only supports data_parallel_size=1")
        if config.execution.batch_size < 1:
            raise ValueError("execution.batch_size must be >= 1")
        if _requires_draft_model(config) and not config.backend.draft_model:
            raise ValueError(
                f"{config.decode_mode.value} requires backend.draft_model"
            )
        if (
            _uses_speculative_config(config)
            and config.generation_params.method.draft_length is None
        ):
            raise ValueError(
                f"{config.decode_mode.value} requires method.draft_length"
            )

        self._config = config
        self._profile_measured = _profile_measured_enabled(config)
        self._profile_prefix = _profile_prefix(config)
        self._entropy_monitoring = _entropy_monitoring_enabled(config)
        self._entropy_monitoring_path = (
            _make_entropy_monitoring_path()
            if self._entropy_monitoring and _uses_speculative_config(config)
            else None
        )
        self._entropy_monitoring_offset = 0
        self._warmup_generate_call_count = _warmup_generate_call_count(config)
        self._generate_call_count = 0
        self._profiled_generate_call_count = 0
        # Real engine construction happens here; CUDA/device choice is fixed now.
        self._llm = LLM(
            **_build_llm_kwargs(
                config,
                entropy_monitoring_path=self._entropy_monitoring_path,
            )
        )
        self._closed = False

    def close(self) -> None:
        """Release the vLLM engine before constructing a different one."""
        if self._closed:
            return
        self._closed = True

        llm = self._llm
        self._llm = None

        engine = getattr(llm, "llm_engine", None)
        engine_core = getattr(engine, "engine_core", None)
        shutdown = getattr(engine_core, "shutdown", None)
        if shutdown is None:
            shutdown = getattr(engine, "shutdown", None)
        if shutdown is not None:
            try:
                shutdown(timeout=30.0)
            except TypeError:
                shutdown()

        del llm
        _remove_entropy_monitoring_path(self._entropy_monitoring_path)
        self._entropy_monitoring_path = None
        gc.collect()
        _empty_cuda_cache()

    def generate_batch(
        self,
        prompts: list[str],
        generation_params: GenerationParamsBatch,
    ) -> GenerateBatchResult:
        if self._closed:
            raise RuntimeError("vLLM backend is closed")
        _validate_request_params(self._config, prompts, generation_params)
        call_id = f"call-{uuid.uuid4().hex[:8]}"
        should_profile = self._should_profile_generate_call()
        profile_started = False

        try:
            if should_profile:
                self._llm.start_profile(
                    _profile_prefix_for_call(self._profile_prefix, call_id)
                )
                profile_started = True

            started_at = time.perf_counter()
            outputs, method_stats = self._generate_batch_inner(
                prompts,
                generation_params,
            )
            wall_time_s = time.perf_counter() - started_at
        finally:
            if profile_started:
                self._llm.stop_profile()
                self._profiled_generate_call_count += 1
            self._generate_call_count += 1

        return GenerateBatchResult(
            outputs=outputs,
            generate_call_id=call_id,
            generate_wall_time_s=wall_time_s,
            aggregate_method_stats=method_stats,
        )

    def _generate_batch_inner(
        self,
        prompts: list[str],
        generation_params: GenerationParamsBatch,
    ) -> tuple[list[BackendSampleOutput], JsonDict]:
        sampling = _uniform_sampling_config(generation_params)
        if sampling.thinking_budget_tokens is None:
            request_outputs = self._llm.generate(
                prompts,
                _build_sampling_params_batch(generation_params),
                use_tqdm=False,
            )
            entropy_by_request = self._consume_entropy_monitoring_records()
            return _outputs_from_request_outputs(
                request_outputs,
                entropy_by_request=entropy_by_request,
            ), {
                "thinking_budget": {"enabled": False}
            }

        return self._generate_batch_with_thinking_budget(prompts, generation_params)

    def _generate_batch_with_thinking_budget(
        self,
        prompts: list[str],
        generation_params: GenerationParamsBatch,
    ) -> tuple[list[BackendSampleOutput], JsonDict]:
        generation_params_by_prompt = _generation_params_by_prompt(
            prompts,
            generation_params,
        )
        sampling = _uniform_sampling_config(generation_params_by_prompt)
        budget = sampling.thinking_budget_tokens
        max_new_tokens = sampling.max_new_tokens
        if budget is None:
            raise ValueError("thinking budget path requires thinking_budget_tokens")
        if budget < 1:
            raise ValueError("sampling.thinking_budget_tokens must be >= 1")
        if max_new_tokens is None:
            raise ValueError(
                "sampling.max_new_tokens is required with thinking_budget_tokens"
            )
        if max_new_tokens <= budget:
            raise ValueError(
                "sampling.max_new_tokens must be greater than "
                "sampling.thinking_budget_tokens"
            )

        tokenizer = _get_tokenizer(self._llm)
        early_stop_ids = _encode_without_special_tokens(
            tokenizer,
            QWEN_THINKING_EARLY_STOP_TEXT,
        )
        think_end_ids = _encode_without_special_tokens(tokenizer, QWEN_THINK_END_TEXT)
        eos_token_id = getattr(tokenizer, "eos_token_id", None)

        first_outputs = self._llm.generate(
            prompts,
            _build_sampling_params_batch(
                _collapse_generation_params(generation_params_by_prompt),
                max_tokens=budget,
            ),
            use_tqdm=False,
        )
        first_entropy = self._consume_entropy_monitoring_records()
        records: list[_BudgetRecord] = []
        second_prompts: list[dict[str, list[int]]] = []
        second_record_indices: list[int] = []

        for index, request_output in enumerate(first_outputs):
            if not request_output.outputs:
                raise ValueError("vLLM returned a request output with no completions")
            completion = request_output.outputs[0]
            first_token_ids = list(completion.token_ids)
            prompt_token_ids = _prompt_token_ids(
                request_output,
                prompts[index],
                tokenizer,
            )
            finished = _finished_with_eos(completion, first_token_ids, eos_token_id)
            forced = False
            continuation_token_ids = list(prompt_token_ids)
            continuation_token_ids.extend(first_token_ids)

            if not finished:
                if not _contains_subsequence(first_token_ids, think_end_ids):
                    continuation_token_ids.extend(early_stop_ids)
                    forced = True
                remaining_tokens = max_new_tokens - (
                    len(continuation_token_ids) - len(prompt_token_ids)
                )
            else:
                remaining_tokens = 0

            record = _BudgetRecord(
                prompt_token_ids=prompt_token_ids,
                first_text=completion.text,
                first_token_ids=first_token_ids,
                forced_think_end=forced,
                remaining_tokens=max(remaining_tokens, 0),
            )
            record.merge_entropy(
                _entropy_for_request_output(
                    index,
                    request_output,
                    first_entropy,
                    request_count=len(first_outputs),
                )
            )
            records.append(record)
            if remaining_tokens > 0:
                second_prompts.append({"prompt_token_ids": continuation_token_ids})
                second_record_indices.append(index)

        if second_prompts:
            grouped: dict[int, list[tuple[int, dict[str, list[int]]]]] = {}
            for record_index, prompt in zip(
                second_record_indices,
                second_prompts,
                strict=True,
            ):
                grouped.setdefault(records[record_index].remaining_tokens, []).append(
                    (record_index, prompt)
                )
            for remaining_tokens, group in grouped.items():
                group_generation_params = [
                    generation_params_by_prompt[record_index]
                    for record_index, _prompt in group
                ]
                second_outputs = self._llm.generate(
                    [prompt for _record_index, prompt in group],
                    _build_sampling_params_batch(
                        _collapse_generation_params(group_generation_params),
                        max_tokens=remaining_tokens,
                    ),
                    use_tqdm=False,
                )
                second_entropy = self._consume_entropy_monitoring_records()
                zipped_outputs = zip(group, second_outputs, strict=True)
                for output_index, item in enumerate(zipped_outputs):
                    (record_index, _prompt), request_output = item
                    if not request_output.outputs:
                        raise ValueError(
                            "vLLM returned a request output with no completions"
                        )
                    completion = request_output.outputs[0]
                    records[record_index].second_text = completion.text
                    records[record_index].second_token_ids = list(completion.token_ids)
                    records[record_index].merge_entropy(
                        _entropy_for_request_output(
                            output_index,
                            request_output,
                            second_entropy,
                            request_count=len(second_outputs),
                        )
                    )

        outputs = [
            BackendSampleOutput(
                generated_text=record.combined_text,
                generated_token_ids=record.combined_token_ids(early_stop_ids),
                n_prompt_tokens=len(record.prompt_token_ids),
                n_out_tokens=len(record.combined_token_ids(early_stop_ids)),
                method_stats=record.method_stats,
            )
            for record in records
        ]
        return outputs, _thinking_budget_stats(
            records,
            budget,
            max_new_tokens,
            len(early_stop_ids),
        )

    def get_run_debug_payload(self) -> JsonDict:
        metrics = self._serialize_metrics(self._llm.get_metrics())
        spec_metrics = [
            metric for metric in metrics if "spec_decode" in metric["name"]
        ]
        return {
            "backend_name": "vllm",
            "decode_mode": self._config.decode_mode.value,
            "model": self._config.backend.model,
            "draft_model": self._config.backend.draft_model,
            "profiling": {
                "profile_measured": self._profile_measured,
                "profile_prefix": self._profile_prefix,
                "warmup_generate_call_count": self._warmup_generate_call_count,
                "generate_call_count": self._generate_call_count,
                "profiled_generate_call_count": self._profiled_generate_call_count,
            },
            "entropy_monitoring": {
                "enabled": self._entropy_monitoring,
                "sidecar_active": self._entropy_monitoring_path is not None,
            },
            "note": (
                "Aggregate vLLM metrics snapshot for this backend instance. "
                "Per-sample speculative counters are intentionally omitted "
                "because vLLM exposes aggregate metrics only."
            ),
            "metrics": spec_metrics,
        }


    def _consume_entropy_monitoring_records(self) -> dict[str, JsonDict]:
        path = self._entropy_monitoring_path
        if path is None:
            return {}

        try:
            with open(path, encoding="utf-8") as handle:
                handle.seek(self._entropy_monitoring_offset)
                lines = handle.readlines()
                self._entropy_monitoring_offset = handle.tell()
        except FileNotFoundError:
            return {}

        by_request: dict[str, JsonDict] = {}
        for line in lines:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            request_id = record.get("request_id")
            entropy = record.get("entropy")
            if request_id is None or not isinstance(entropy, dict):
                continue
            current = by_request.setdefault(str(request_id), _empty_entropy_payload())
            _merge_entropy_payload(current, entropy)
        return {
            request_id: _finalize_entropy_payload(payload)
            for request_id, payload in by_request.items()
        }

    def _serialize_metrics(self, metrics: list[Metric]) -> list[JsonDict]:
        serialized: list[JsonDict] = []
        for metric in metrics:
            item: JsonDict = {
                "name": metric.name,
                "labels": dict(metric.labels),
                "kind": metric.__class__.__name__.lower(),
            }
            if isinstance(metric, (Counter, Gauge)):
                item["value"] = metric.value
            elif isinstance(metric, Vector):
                item["values"] = list(metric.values)
            elif isinstance(metric, Histogram):
                item["count"] = metric.count
                item["sum"] = metric.sum
                item["buckets"] = dict(metric.buckets)
            else:
                item["raw"] = asdict(metric)
            serialized.append(item)
        return serialized

    def _should_profile_generate_call(self) -> bool:
        return (
            self._profile_measured
            and self._profiled_generate_call_count == 0
            and self._generate_call_count >= self._warmup_generate_call_count
        )


class _BudgetRecord:
    def __init__(
        self,
        prompt_token_ids: list[int],
        first_text: str,
        first_token_ids: list[int],
        forced_think_end: bool,
        remaining_tokens: int,
    ) -> None:
        self.prompt_token_ids = prompt_token_ids
        self.first_text = first_text
        self.first_token_ids = first_token_ids
        self.forced_think_end = forced_think_end
        self.remaining_tokens = remaining_tokens
        self.second_text = ""
        self.second_token_ids: list[int] = []
        self._entropy_accumulator: JsonDict | None = None

    @property
    def combined_text(self) -> str:
        forced_text = QWEN_THINKING_EARLY_STOP_TEXT if self.forced_think_end else ""
        return f"{self.first_text}{forced_text}{self.second_text}"

    def combined_token_ids(self, early_stop_ids: list[int]) -> list[int]:
        forced_ids = early_stop_ids if self.forced_think_end else []
        return [*self.first_token_ids, *forced_ids, *self.second_token_ids]

    def merge_entropy(self, entropy: JsonDict | None) -> None:
        if entropy is None:
            return
        if self._entropy_accumulator is None:
            self._entropy_accumulator = _empty_entropy_payload()
        _merge_entropy_payload(self._entropy_accumulator, entropy)

    @property
    def method_stats(self) -> JsonDict:
        if self._entropy_accumulator is None:
            return {}
        return {"entropy": _finalize_entropy_payload(self._entropy_accumulator)}


def _outputs_from_request_outputs(
    request_outputs: list[Any],
    entropy_by_request: dict[str, JsonDict] | None = None,
) -> list[BackendSampleOutput]:
    outputs: list[BackendSampleOutput] = []
    entropy_by_request = entropy_by_request or {}
    for index, request_output in enumerate(request_outputs):
        if not request_output.outputs:
            raise ValueError("vLLM returned a request output with no completions")

        completion = request_output.outputs[0]
        method_stats: JsonDict = {}
        entropy = _entropy_for_request_output(
            index,
            request_output,
            entropy_by_request,
            request_count=len(request_outputs),
        )
        if entropy is not None:
            method_stats["entropy"] = entropy
        outputs.append(
            BackendSampleOutput(
                generated_text=completion.text,
                generated_token_ids=list(completion.token_ids),
                n_prompt_tokens=_len_or_none(request_output.prompt_token_ids),
                n_out_tokens=len(completion.token_ids),
                method_stats=method_stats,
            )
        )
    return outputs


def _entropy_for_request_output(
    index: int,
    request_output: Any,
    entropy_by_request: dict[str, JsonDict] | None,
    *,
    request_count: int,
) -> JsonDict | None:
    entropy_by_request = entropy_by_request or {}
    request_id = getattr(request_output, "request_id", None)
    if request_id is not None:
        entropy = entropy_by_request.get(str(request_id))
        if entropy is not None:
            return entropy

    entropy_values = list(entropy_by_request.values())
    if len(entropy_values) == request_count:
        return entropy_values[index]
    return None


def _get_tokenizer(llm: object) -> object:
    if hasattr(llm, "get_tokenizer"):
        return llm.get_tokenizer()
    tokenizer = getattr(llm, "tokenizer", None)
    if tokenizer is not None:
        return tokenizer
    raise ValueError("thinking_budget_tokens requires access to the vLLM tokenizer")


def _encode_without_special_tokens(tokenizer: object, text: str) -> list[int]:
    encode = getattr(tokenizer, "encode", None)
    if encode is None:
        raise ValueError("thinking_budget_tokens requires tokenizer.encode")
    return list(encode(text, add_special_tokens=False))


def _prompt_token_ids(
    request_output: object,
    prompt: str,
    tokenizer: object,
) -> list[int]:
    prompt_token_ids = getattr(request_output, "prompt_token_ids", None)
    if prompt_token_ids is not None:
        return list(prompt_token_ids)
    return _encode_without_special_tokens(tokenizer, prompt)


def _finished_with_eos(
    completion: object,
    token_ids: list[int],
    eos_token_id: int | None,
) -> bool:
    finish_reason = getattr(completion, "finish_reason", None)
    if finish_reason in {"stop", "eos_token"}:
        return True
    return eos_token_id is not None and eos_token_id in token_ids


def _contains_subsequence(values: list[int], needle: list[int]) -> bool:
    if not needle:
        return False
    if len(needle) > len(values):
        return False
    last_start = len(values) - len(needle)
    return any(values[start : start + len(needle)] == needle for start in range(last_start + 1))


def _min_max(values: list[int]) -> JsonDict:
    if not values:
        return {"min": 0, "max": 0}
    return {"min": min(values), "max": max(values)}


def _thinking_budget_stats(
    records: list[_BudgetRecord],
    budget: int,
    max_new_tokens: int,
    early_stop_token_count: int,
) -> JsonDict:
    first_counts = [len(record.first_token_ids) for record in records]
    second_counts = [len(record.second_token_ids) for record in records]
    total_counts = [
        len(record.first_token_ids)
        + (early_stop_token_count if record.forced_think_end else 0)
        + len(record.second_token_ids)
        for record in records
    ]
    return {
        "thinking_budget": {
            "enabled": True,
            "budget_tokens": budget,
            "max_new_tokens": max_new_tokens,
            "forced_think_end_count": sum(
                1 for record in records if record.forced_think_end
            ),
            "first_stage_output_tokens": _min_max(first_counts),
            "second_stage_output_tokens": _min_max(second_counts),
            "total_model_output_tokens": _min_max(total_counts),
        }
    }


def _validate_request_params(
    config: BackendRequestConfig,
    prompts: list[str],
    generation_params: GenerationParamsBatch,
) -> None:
    request_params = _generation_params_by_prompt(prompts, generation_params)
    if not _uses_speculative_config(config):
        return

    engine_draft_length = config.generation_params.method.draft_length
    for params in request_params:
        request_draft_length = params.method.draft_length
        if request_draft_length != engine_draft_length:
            raise ValueError(
                "stock vLLM locks speculative draft_length at engine initialization"
            )


def _generation_params_by_prompt(
    prompts: list[str],
    generation_params: GenerationParamsBatch,
) -> list[GenerationParams]:
    if isinstance(generation_params, list):
        if len(generation_params) != len(prompts):
            raise ValueError(
                "per-prompt generation_params length must match prompts length"
            )
        return generation_params
    return [generation_params] * len(prompts)


def _uniform_sampling_config(
    generation_params: GenerationParamsBatch,
) -> SamplingConfig:
    params_list = (
        generation_params
        if isinstance(generation_params, list)
        else [generation_params]
    )
    sampling = params_list[0].sampling
    for params in params_list[1:]:
        other = params.sampling
        if (
            other.thinking_budget_tokens != sampling.thinking_budget_tokens
            or other.max_new_tokens != sampling.max_new_tokens
        ):
            raise ValueError(
                "batched generation_params must share thinking budget and "
                "max_new_tokens"
            )
    return sampling


def _build_sampling_params_batch(
    generation_params: GenerationParamsBatch,
    max_tokens: int | None = None,
) -> SamplingParams | list[SamplingParams]:
    if isinstance(generation_params, list):
        return [
            _build_sampling_params(params, max_tokens=max_tokens)
            for params in generation_params
        ]
    return _build_sampling_params(generation_params, max_tokens=max_tokens)


def _collapse_generation_params(
    generation_params: list[GenerationParams],
) -> GenerationParams | list[GenerationParams]:
    if len(generation_params) == 1:
        return generation_params[0]
    return generation_params


def _build_sampling_params(
    generation_params: GenerationParams,
    max_tokens: int | None = None,
) -> SamplingParams:
    sampling = generation_params.sampling
    top_k = 0 if sampling.top_k is None else sampling.top_k
    stop = sampling.stop or None
    return SamplingParams.from_optional(
        temperature=sampling.temperature,
        top_p=sampling.top_p,
        top_k=top_k,
        presence_penalty=sampling.presence_penalty,
        seed=sampling.seed,
        stop=stop,
        max_tokens=sampling.max_new_tokens if max_tokens is None else max_tokens,
    )


def _build_llm_kwargs(
    config: BackendRequestConfig,
    entropy_monitoring_path: str | None = None,
) -> JsonDict:
    backend = config.backend
    execution = config.execution
    options = dict(backend.options)
    # Runner consumes prompt formatting before this backend boundary.
    options.pop("prompt_format", None)
    options.pop("chat_template_kwargs", None)
    # Repo-local profiling controls wrap measured generate calls. They are not
    # vLLM constructor kwargs. `profiler_config` itself is passed through.
    options.pop(PROFILE_MEASURED_OPTION, None)
    options.pop(PROFILE_PREFIX_OPTION, None)
    options.pop(ENTROPY_MONITORING_OPTION, None)

    llm_kwargs: JsonDict = dict(options)
    llm_kwargs["model"] = backend.model
    llm_kwargs["disable_log_stats"] = False
    if backend.dtype is not None:
        llm_kwargs["dtype"] = backend.dtype
    if backend.revision is not None:
        llm_kwargs["revision"] = backend.revision
    if execution.tensor_parallel_size is not None:
        llm_kwargs["tensor_parallel_size"] = execution.tensor_parallel_size
    if _uses_speculative_config(config):
        spec_config = {
            "method": SPECULATIVE_METHOD_BY_DECODE_MODE[config.decode_mode],
            "model": (
                None
                if config.decode_mode == DecodeMode.SD_MTP_RELAXED
                else backend.draft_model
            ),
            "num_speculative_tokens": (
                config.generation_params.method.draft_length
            ),
        }
        if entropy_monitoring_path is not None:
            spec_config["entropy_monitoring"] = True
            spec_config["entropy_monitoring_path"] = entropy_monitoring_path
        if config.decode_mode == DecodeMode.SD_VANILLA:
            spec_config.update(DRAFT_MODEL_STOCHASTIC_OPTIONS)
        elif config.decode_mode == DecodeMode.SD_RELAXED:
            spec_config.update(_relaxed_speculative_options(config))
        elif config.decode_mode == DecodeMode.SD_MTP_RELAXED:
            spec_config.update(_mtp_relaxed_speculative_options(config))
        llm_kwargs["speculative_config"] = spec_config
    return llm_kwargs


def _mtp_relaxed_speculative_options(config: BackendRequestConfig) -> JsonDict:
    options = _relaxed_speculative_options(config)
    bonus_policy = options.get("relaxed_bonus_token_policy")
    if bonus_policy not in ("target_p", "relaxed_T_qp"):
        raise ValueError(
            "sd_mtp_relaxed supports only bonus_token_policy in "
            "{'target_p', 'relaxed_T_qp'}."
        )
    return options


def _relaxed_speculative_options(config: BackendRequestConfig) -> JsonDict:
    method = config.generation_params.method
    params = method.params
    relaxed_name = canonical_relaxed_target_method(
        params.get("relaxed_target_method") or method.name
    )
    if relaxed_name == "cactus":
        delta = params.get("delta", params.get("cactus_delta"))
        if delta is None:
            raise ValueError("sd_relaxed cactus requires method.params.delta.")

        return {
            **DRAFT_MODEL_STOCHASTIC_OPTIONS,
            "relaxed_target_method": "cactus",
            "cactus_delta": delta,
            "relaxed_bonus_token_policy": params.get(
                "bonus_token_policy", "target_p"
            ),
        }
    if relaxed_name == "ensemble":
        verifier_weight = _relaxed_float_param(params, "verifier_weight", "w_p")
        if verifier_weight is None:
            raise ValueError(
                "sd_relaxed ensemble requires method.params.verifier_weight."
            )
        if not 0.0 <= verifier_weight <= 1.0:
            raise ValueError(
                "sd_relaxed ensemble requires verifier_weight in [0, 1]."
            )

        return {
            **DRAFT_MODEL_STOCHASTIC_OPTIONS,
            "relaxed_target_method": "ensemble",
            "verifier_weight": verifier_weight,
            "relaxed_bonus_token_policy": params.get(
                "bonus_token_policy", "target_p"
            ),
        }
    if relaxed_name == "rfsd":
        divergence = _relaxed_str_param(
            params,
            "divergence_metric",
            "fuzzy_divergence",
            "divergence",
        )
        if divergence is None:
            raise ValueError(
                "sd_relaxed rfsd requires method.params.divergence_metric."
            )
        if divergence not in ("kl", "js"):
            raise ValueError(
                "sd_relaxed rfsd requires divergence_metric in {'kl', 'js'}."
            )

        threshold = _relaxed_float_param(
            params,
            "divergence_threshold",
            "fuzzy_threshold",
            "threshold",
            "tau",
        )
        if threshold is None:
            raise ValueError(
                "sd_relaxed rfsd requires method.params.divergence_threshold."
            )
        if threshold < 0:
            raise ValueError("sd_relaxed rfsd requires divergence_threshold >= 0.")

        return {
            **DRAFT_MODEL_STOCHASTIC_OPTIONS,
            "relaxed_target_method": "rfsd",
            "fuzzy_divergence": divergence,
            "fuzzy_threshold": threshold,
            "relaxed_bonus_token_policy": params.get(
                "bonus_token_policy", "target_p"
            ),
        }
    if relaxed_name == "fsd":
        divergence = _relaxed_str_param(
            params,
            "divergence_metric",
            "fsd_divergence",
            "fuzzy_divergence",
            "divergence",
        )
        if divergence is None:
            raise ValueError("sd_relaxed fsd requires method.params.divergence_metric.")
        if divergence not in ("kl", "js"):
            raise ValueError(
                "sd_relaxed fsd requires divergence_metric in {'kl', 'js'}."
            )

        threshold = _relaxed_float_param(
            params,
            "divergence_threshold",
            "fsd_threshold",
            "fuzzy_threshold",
            "threshold",
            "tau",
        )
        if threshold is None:
            raise ValueError("sd_relaxed fsd requires method.params.divergence_threshold.")
        if threshold < 0:
            raise ValueError("sd_relaxed fsd requires divergence_threshold >= 0.")

        return {
            **DRAFT_MODEL_STOCHASTIC_OPTIONS,
            "relaxed_target_method": "fsd",
            # The overlay reuses the fuzzy divergence controls for FSD/rFSD.
            "fuzzy_divergence": divergence,
            "fuzzy_threshold": threshold,
            "relaxed_bonus_token_policy": params.get(
                "bonus_token_policy", "target_p"
            ),
        }
    if relaxed_name == "spec_cascade_opt":
        alpha = _relaxed_float_param(params, "spec_cascade_alpha", "alpha")
        if alpha is None:
            raise ValueError(
                "sd_relaxed spec_cascade_opt requires "
                "method.params.spec_cascade_alpha."
            )
        gate = _relaxed_str_param(params, "spec_cascade_opt_gate", "gate")
        gate = gate or "processed"
        if gate not in ("processed", "paper"):
            raise ValueError(
                "sd_relaxed spec_cascade_opt requires "
                "spec_cascade_opt_gate in {'processed', 'paper'}."
            )

        return {
            **DRAFT_MODEL_STOCHASTIC_OPTIONS,
            "relaxed_target_method": "spec_cascade_opt",
            "spec_cascade_alpha": alpha,
            "spec_cascade_opt_gate": gate,
            "relaxed_bonus_token_policy": params.get(
                "bonus_token_policy", "target_p"
            ),
        }
    if relaxed_name == "spec_cascade_tok3":
        alpha = _relaxed_float_param(params, "spec_cascade_alpha", "alpha")
        if alpha is None:
            raise ValueError(
                "sd_relaxed spec_cascade_tok3 requires "
                "method.params.spec_cascade_alpha."
            )
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(
                "sd_relaxed spec_cascade_tok3 requires "
                "0 <= spec_cascade_alpha <= 1."
            )

        top_set = _relaxed_str_param(
            params,
            "spec_cascade_tok3_top_set",
            "top_set",
        ) or "paper"
        if top_set not in ("paper", "processed"):
            raise ValueError(
                "sd_relaxed spec_cascade_tok3 requires "
                "spec_cascade_tok3_top_set in {'paper', 'processed'}."
            )

        return {
            **DRAFT_MODEL_STOCHASTIC_OPTIONS,
            "relaxed_target_method": "spec_cascade_tok3",
            "spec_cascade_alpha": alpha,
            "spec_cascade_tok3_top_set": top_set,
            "relaxed_bonus_token_policy": params.get(
                "bonus_token_policy", "target_p"
            ),
        }
    if relaxed_name == "lossy_spec_decode_beta1":
        alpha = _relaxed_float_param(params, "lossy_alpha", "alpha")
        if alpha is None:
            raise ValueError(
                "sd_relaxed lossy_spec_decode_beta1 requires "
                "method.params.lossy_alpha."
            )
        if not 0.0 <= alpha < 1.0:
            raise ValueError(
                "sd_relaxed lossy_spec_decode_beta1 requires 0 <= lossy_alpha < 1."
            )

        return {
            **DRAFT_MODEL_STOCHASTIC_OPTIONS,
            "relaxed_target_method": "lossy_spec_decode_beta1",
            "lossy_alpha": alpha,
            "relaxed_bonus_token_policy": params.get(
                "bonus_token_policy", "target_p"
            ),
        }
    if relaxed_name in ("scd_expert_toppk_gated", "scd_alpha"):
        beta = _relaxed_float_param(params, "scd_beta", "beta")
        if beta is None:
            raise ValueError(
                f"sd_relaxed {relaxed_name} requires method.params.scd_beta."
            )
        if not math.isfinite(beta) or beta < 0:
            raise ValueError(f"sd_relaxed {relaxed_name} requires scd_beta >= 0.")

        explicit_scd_temperature = _relaxed_float_param(
            params,
            "scd_temperature",
            "temperature_cd",
            "t_cd",
        )
        scd_temperature = (
            explicit_scd_temperature
            if explicit_scd_temperature is not None
            else config.generation_params.sampling.temperature
        )
        if scd_temperature is not None and (
            not math.isfinite(scd_temperature) or scd_temperature <= 0
        ):
            raise ValueError(
                f"sd_relaxed {relaxed_name} requires scd_temperature > 0 when set."
            )

        scd_options = {
            **DRAFT_MODEL_STOCHASTIC_OPTIONS,
            "relaxed_target_method": relaxed_name,
            "scd_beta": beta,
            "relaxed_bonus_token_policy": params.get(
                "bonus_token_policy", "relaxed_T_qp"
            ),
        }
        if scd_temperature is not None:
            scd_options["scd_temperature"] = scd_temperature
        if relaxed_name == "scd_alpha":
            alpha = _relaxed_float_param(params, "scd_alpha", "alpha")
            if alpha is None:
                raise ValueError("sd_relaxed scd_alpha requires method.params.scd_alpha.")
            if not math.isfinite(alpha) or not 0.0 < alpha <= 1.0:
                raise ValueError("sd_relaxed scd_alpha requires 0 < scd_alpha <= 1.")
            scd_options["scd_alpha"] = alpha
        elif "scd_alpha" in params or "alpha" in params:
            raise ValueError(
                "sd_relaxed scd_expert_toppk_gated does not use scd_alpha."
            )
        return scd_options
    else:
        raise ValueError(f"Unsupported relaxed speculative method: {relaxed_name!r}.")


def _relaxed_float_param(
    params: JsonDict,
    primary_name: str,
    *alias_names: str,
) -> float | None:
    names = (primary_name, *alias_names)
    values = [
        (name, params[name])
        for name in names
        if name in params
    ]
    if not values:
        return None

    parsed = [(_name, _float_param(_name, value)) for _name, value in values]
    first_name, first_value = parsed[0]
    for name, value in parsed[1:]:
        if value != first_value:
            raise ValueError(
                f"method.params aliases for {primary_name} must match when "
                f"multiple are set; got {first_name}={first_value} and "
                f"{name}={value}."
            )
    return first_value


def _relaxed_str_param(
    params: JsonDict,
    primary_name: str,
    *alias_names: str,
) -> str | None:
    names = (primary_name, *alias_names)
    values = [(name, params[name]) for name in names if name in params]
    if not values:
        return None

    parsed = [(_name, _str_param(_name, value)) for _name, value in values]
    first_name, first_value = parsed[0]
    for name, value in parsed[1:]:
        if value != first_value:
            raise ValueError(
                f"method.params aliases for {primary_name} must match when "
                f"multiple are set; got {first_name}={first_value!r} and "
                f"{name}={value!r}."
            )
    return first_value


def _float_param(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"method.params.{name} must be a number, got {value!r}.")
    return float(value)


def _str_param(name: str, value: object) -> str:
    if not isinstance(value, str):
        raise ValueError(f"method.params.{name} must be a string, got {value!r}.")
    return value


def _empty_entropy_payload() -> JsonDict:
    payload: JsonDict = {
        "version": 1,
        "unit": ENTROPY_UNIT,
        "distribution": ENTROPY_DISTRIBUTION,
    }
    for name in ENTROPY_STAT_NAMES:
        payload[name] = {"entropy_sum": 0.0, "token_count": 0}
    return payload


def _merge_entropy_payload(target: JsonDict, source: JsonDict) -> None:
    for name in ENTROPY_STAT_NAMES:
        item = source.get(name)
        if not isinstance(item, dict):
            continue
        target_item = target[name]
        entropy_sum = item.get("entropy_sum")
        token_count = item.get("token_count")
        if isinstance(entropy_sum, (int, float)) and not isinstance(entropy_sum, bool):
            target_item["entropy_sum"] = float(target_item["entropy_sum"]) + float(
                entropy_sum
            )
        if isinstance(token_count, (int, float)) and not isinstance(token_count, bool):
            target_item["token_count"] = int(target_item["token_count"]) + int(
                token_count
            )


def _finalize_entropy_payload(payload: JsonDict) -> JsonDict:
    finalized: JsonDict = {
        "version": 1,
        "unit": ENTROPY_UNIT,
        "distribution": ENTROPY_DISTRIBUTION,
    }
    for name in ENTROPY_STAT_NAMES:
        item = payload[name]
        entropy_sum = float(item["entropy_sum"])
        token_count = int(item["token_count"])
        finalized[name] = {
            "entropy_sum": entropy_sum,
            "token_count": token_count,
            "mean_entropy": entropy_sum / token_count if token_count else None,
        }
    return finalized


def _make_entropy_monitoring_path() -> str:
    handle = tempfile.NamedTemporaryFile(
        prefix="spec-dec-entropy-",
        suffix=".jsonl",
        delete=False,
    )
    try:
        return handle.name
    finally:
        handle.close()


def _remove_entropy_monitoring_path(path: str | None) -> None:
    if path is None:
        return
    try:
        import os

        os.unlink(path)
    except OSError:
        pass


def _empty_cuda_cache() -> None:
    try:
        import torch
    except Exception:
        return

    try:
        torch.cuda.empty_cache()
    except Exception:
        pass


def _uses_speculative_config(config: BackendRequestConfig) -> bool:
    return config.decode_mode in SPECULATIVE_METHOD_BY_DECODE_MODE


def _requires_draft_model(config: BackendRequestConfig) -> bool:
    return decode_mode_requires_draft_model(config.decode_mode)


def _len_or_none(tokens: list[int] | None) -> int | None:
    return None if tokens is None else len(tokens)


def _profile_measured_enabled(config: BackendRequestConfig) -> bool:
    return bool(config.backend.options.get(PROFILE_MEASURED_OPTION, False))


def _entropy_monitoring_enabled(config: BackendRequestConfig) -> bool:
    return bool(config.backend.options.get(ENTROPY_MONITORING_OPTION, False))


def _profile_prefix(config: BackendRequestConfig) -> str | None:
    value = config.backend.options.get(PROFILE_PREFIX_OPTION)
    return value if isinstance(value, str) else None


def _profile_prefix_for_call(prefix: str | None, call_id: str) -> str:
    if prefix:
        return f"{prefix}-{call_id}"
    return call_id


def _warmup_generate_call_count(config: BackendRequestConfig) -> int:
    execution = config.execution
    if not execution.warmup_enabled or execution.warmup_sample_count < 1:
        return 0
    batch_size = max(execution.batch_size, 1)
    return (execution.warmup_sample_count + batch_size - 1) // batch_size


__all__ = [
    "OfflineVllmBackend",
    "ENTROPY_MONITORING_OPTION",
    "PROFILE_MEASURED_OPTION",
    "PROFILE_PREFIX_OPTION",
]
