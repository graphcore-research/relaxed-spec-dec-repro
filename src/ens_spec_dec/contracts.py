"""Minimal typed contracts for configs and persisted offline artifacts.

Contract hierarchy
==================

Config side
-----------
RunConfig (contains the exact config surface for one run)
  metadata: RunMetadata (contains run id, grouping ids, scope, decode mode)
  task: TaskConfig (contains task identity, dataset slice, task metadata)
    defaults: TaskDefaults (contains task-owned default knobs)
      generation_params: GenerationParams (sampling + method defaults)
  backend: BackendConfig (contains e.g. backend name, model, draft model)
  generation_params: GenerationParams (per-run sampling + method knobs)
  execution: ExecutionConfig (contains batch size, warmup/repeat knobs,
    and parallelism strategy)

Artifact side
-------------
RunManifest (contains the run-dir contract and relative artifact paths)

ScoredSample (contains one sample's input, generation result, and score)
  sample: SampleInput (contains e.g. prompt, reference answer, sample metadata)
  generation: GenerationResult (contains e.g. text, token ids, timings, traces)
  score_payload: task/eval-owned scoring output

RunSummary (contains generic run-level aggregates for comparison/analysis)

Later milestones may add optional distribution-level debug artifacts, such as
token distributions or top-k logits/log-probs, under run-local debug files.

Intended use
------------
- YAML for hand-edited run configs.
- JSON / JSONL for persisted sample and summary artifacts.
- Keep generation metadata separate from scoring data.
- Keep task-owned defaults separate from concrete per-run overrides.
- Keep method-specific knobs, such as speculative-decoding controls, out of
  generic sampling config.
- `generation_params` is the persisted config shape. Legacy `sampling` and
  `method` YAML keys remain readable for old run configs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

import yaml

JsonScalar = None | bool | int | float | str
JsonValue = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
JsonDict = dict[str, JsonValue]


class MeasurementScope(StrEnum):
    # Labels the kind of measurement a run is intended to support.
    OFFLINE_EVAL = "offline_eval"
    PROXY_SPEED = "proxy_speed"
    SERVING_BENCHMARK = "serving_benchmark"


class DecodeMode(StrEnum):
    # Names the decoding path used for a run.
    AR_REF = "ar_ref"
    SD_VANILLA = "sd_vanilla"
    SD_RELAXED = "sd_relaxed"
    SD_DFLASH = "sd_dflash"
    SD_MTP = "sd_mtp"
    SD_MTP_RELAXED = "sd_mtp_relaxed"
    SD_EAGLE3 = "sd_eagle3"


SPECULATIVE_DECODE_MODES = frozenset(
    {
        DecodeMode.SD_VANILLA,
        DecodeMode.SD_RELAXED,
        DecodeMode.SD_DFLASH,
        DecodeMode.SD_MTP,
        DecodeMode.SD_MTP_RELAXED,
        DecodeMode.SD_EAGLE3,
    }
)
DRAFT_MODEL_DECODE_MODES = SPECULATIVE_DECODE_MODES - {
    DecodeMode.SD_MTP,
    DecodeMode.SD_MTP_RELAXED,
}


def is_speculative_decode_mode(decode_mode: DecodeMode | str) -> bool:
    return _decode_mode_or_none(decode_mode) in SPECULATIVE_DECODE_MODES


def decode_mode_requires_draft_model(decode_mode: DecodeMode | str) -> bool:
    return _decode_mode_or_none(decode_mode) in DRAFT_MODEL_DECODE_MODES


def _decode_mode_or_none(decode_mode: DecodeMode | str) -> DecodeMode | None:
    try:
        return DecodeMode(decode_mode)
    except ValueError:
        return None


def _json_dict(data: dict[str, JsonValue] | None) -> JsonDict:
    return {} if data is None else dict(data)


def _str_dict(data: dict[str, str] | None) -> dict[str, str]:
    return {} if data is None else dict(data)


def _str_list(data: list[str] | None) -> list[str]:
    return [] if data is None else list(data)


def _int_list(data: list[int] | None) -> list[int] | None:
    return None if data is None else list(data)


@dataclass(slots=True)
class SamplingConfig:
    # Holds serializable decode and sampling knobs for a concrete run.
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    presence_penalty: float | None = None
    max_new_tokens: int | None = None
    thinking_budget_tokens: int | None = None
    stop: list[str] = field(default_factory=list)
    seed: int | None = None

    def to_dict(self) -> JsonDict:
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "presence_penalty": self.presence_penalty,
            "max_new_tokens": self.max_new_tokens,
            "thinking_budget_tokens": self.thinking_budget_tokens,
            "stop": list(self.stop),
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, data: dict[str, JsonValue] | None) -> "SamplingConfig":
        data = data or {}
        return cls(
            temperature=data.get("temperature"),
            top_p=data.get("top_p"),
            top_k=data.get("top_k"),
            presence_penalty=data.get("presence_penalty"),
            max_new_tokens=data.get("max_new_tokens"),
            thinking_budget_tokens=data.get("thinking_budget_tokens"),
            stop=_str_list(data.get("stop")),
            seed=data.get("seed"),
        )


@dataclass(slots=True)
class ExecutionConfig:
    # Holds runner/backend execution knobs such as batch size and parallelism.
    batch_size: int = 1
    tensor_parallel_size: int | None = None
    data_parallel_size: int | None = None
    warmup_enabled: bool = True
    warmup_sample_count: int = 1
    warmup_max_new_tokens: int = 256
    record_warmup_debug: bool = False
    repeat_count: int = 1
    repeat_batch_size: int = 1
    repeat_seed_offset: int = 0
    seed_list: list[int] = field(default_factory=list)

    def to_dict(self) -> JsonDict:
        return {
            "batch_size": self.batch_size,
            "tensor_parallel_size": self.tensor_parallel_size,
            "data_parallel_size": self.data_parallel_size,
            "warmup_enabled": self.warmup_enabled,
            "warmup_sample_count": self.warmup_sample_count,
            "warmup_max_new_tokens": self.warmup_max_new_tokens,
            "record_warmup_debug": self.record_warmup_debug,
            "repeat_count": self.repeat_count,
            "repeat_batch_size": self.repeat_batch_size,
            "repeat_seed_offset": self.repeat_seed_offset,
            "seed_list": list(self.seed_list),
        }

    @classmethod
    def from_dict(cls, data: dict[str, JsonValue] | None) -> "ExecutionConfig":
        data = data or {}
        return cls(
            batch_size=data.get("batch_size", 1),
            tensor_parallel_size=data.get("tensor_parallel_size"),
            data_parallel_size=data.get("data_parallel_size"),
            warmup_enabled=data.get("warmup_enabled", True),
            warmup_sample_count=data.get("warmup_sample_count", 1),
            warmup_max_new_tokens=data.get("warmup_max_new_tokens", 256),
            record_warmup_debug=data.get("record_warmup_debug", False),
            repeat_count=data.get("repeat_count", 1),
            repeat_batch_size=data.get("repeat_batch_size", 1),
            repeat_seed_offset=data.get("repeat_seed_offset", 0),
            seed_list=list(data.get("seed_list") or []),
        )


@dataclass(slots=True)
class MethodConfig:
    # Holds a stable method id plus method-specific knobs and variant labels.
    name: str | None = None
    variant: str | None = None
    method_version: str | None = None
    draft_length: int | None = None
    params: JsonDict = field(default_factory=dict)

    def to_dict(self) -> JsonDict:
        return {
            "name": self.name,
            "variant": self.variant,
            "method_version": self.method_version,
            "draft_length": self.draft_length,
            "params": dict(self.params),
        }

    @classmethod
    def from_dict(cls, data: dict[str, JsonValue] | None) -> "MethodConfig":
        data = data or {}
        return cls(
            name=data.get("name"),
            variant=data.get("variant"),
            method_version=data.get("method_version"),
            draft_length=data.get("draft_length"),
            params=_json_dict(data.get("params")),
        )


@dataclass(slots=True)
class GenerationParams:
    # Groups per-run decode knobs that can be passed to a generation request.
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    method: MethodConfig = field(default_factory=MethodConfig)

    def to_dict(self) -> JsonDict:
        return {
            "sampling": self.sampling.to_dict(),
            "method": self.method.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, JsonValue] | None) -> "GenerationParams":
        data = data or {}
        return cls(
            sampling=SamplingConfig.from_dict(data.get("sampling")),
            method=MethodConfig.from_dict(data.get("method")),
        )


@dataclass(slots=True)
class TaskDefaults:
    # Groups task-owned default generation params that runs may override.
    generation_params: GenerationParams = field(default_factory=GenerationParams)

    @property
    def sampling(self) -> SamplingConfig:
        return self.generation_params.sampling

    @sampling.setter
    def sampling(self, value: SamplingConfig) -> None:
        self.generation_params.sampling = value

    @property
    def method(self) -> MethodConfig:
        return self.generation_params.method

    @method.setter
    def method(self, value: MethodConfig) -> None:
        self.generation_params.method = value

    def to_dict(self) -> JsonDict:
        return {"generation_params": self.generation_params.to_dict()}

    @classmethod
    def from_dict(cls, data: dict[str, JsonValue] | None) -> "TaskDefaults":
        data = data or {}
        if "generation_params" in data:
            return cls(
                generation_params=GenerationParams.from_dict(
                    data.get("generation_params")
                )
            )
        return cls(
            generation_params=GenerationParams(
                sampling=SamplingConfig.from_dict(data.get("sampling")),
                method=MethodConfig.from_dict(data.get("method")),
            )
        )


@dataclass(slots=True)
class TaskConfig:
    # Describes the task slice plus its owned defaults and task metadata.
    name: str # of backend
    dataset: str | None = None
    split: str | None = None
    max_samples: int | None = None
    defaults: TaskDefaults = field(default_factory=TaskDefaults)
    metadata: JsonDict = field(default_factory=dict)

    def to_dict(self) -> JsonDict:
        return {
            "name": self.name,
            "dataset": self.dataset,
            "split": self.split,
            "max_samples": self.max_samples,
            "defaults": self.defaults.to_dict(),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, JsonValue]) -> "TaskConfig":
        return cls(
            name=data["name"],
            dataset=data.get("dataset"),
            split=data.get("split"),
            max_samples=data.get("max_samples"),
            defaults=TaskDefaults.from_dict(data.get("defaults")),
            metadata=_json_dict(data.get("metadata")),
        )


@dataclass(slots=True)
class BackendConfig:
    # Identifies the backend and model pair used to execute a run.
    name: str
    model: str
    draft_model: str | None = None
    revision: str | None = None
    dtype: str | None = None
    options: JsonDict = field(default_factory=dict)

    def to_dict(self) -> JsonDict:
        return {
            "name": self.name,
            "model": self.model,
            "draft_model": self.draft_model,
            "revision": self.revision,
            "dtype": self.dtype,
            "options": dict(self.options),
        }

    @classmethod
    def from_dict(cls, data: dict[str, JsonValue]) -> "BackendConfig":
        return cls(
            name=data["name"],
            model=data["model"],
            draft_model=data.get("draft_model"),
            revision=data.get("revision"),
            dtype=data.get("dtype"),
            options=_json_dict(data.get("options")),
        )


@dataclass(slots=True)
class RunMetadata:
    # Captures run identity, grouping ids, measurement scope, and debug refs.
    run_id: str
    sweep_id: str | None = None
    experiment_group: str | None = None
    measurement_scope: MeasurementScope = MeasurementScope.OFFLINE_EVAL
    decode_mode: DecodeMode = DecodeMode.AR_REF
    tags: list[str] = field(default_factory=list)
    debug_refs: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> JsonDict:
        return {
            "run_id": self.run_id,
            "sweep_id": self.sweep_id,
            "experiment_group": self.experiment_group,
            "measurement_scope": self.measurement_scope.value,
            "decode_mode": self.decode_mode.value,
            "tags": list(self.tags),
            "debug_refs": dict(self.debug_refs),
        }

    @classmethod
    def from_dict(cls, data: dict[str, JsonValue]) -> "RunMetadata":
        return cls(
            run_id=data["run_id"],
            sweep_id=data.get("sweep_id"),
            experiment_group=data.get("experiment_group"),
            measurement_scope=MeasurementScope(
                data.get("measurement_scope", MeasurementScope.OFFLINE_EVAL)
            ),
            decode_mode=DecodeMode(data.get("decode_mode", DecodeMode.AR_REF)),
            tags=_str_list(data.get("tags")),
            debug_refs=_str_dict(data.get("debug_refs")),
        )


@dataclass(slots=True)
class RunConfig:
    # Pins the exact config surface for one run while preserving task defaults.
    """Exact per-run knobs plus task-owned defaults kept for reference."""

    metadata: RunMetadata
    task: TaskConfig
    backend: BackendConfig
    generation_params: GenerationParams
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    debug_refs: dict[str, str] = field(default_factory=dict)

    @property
    def sampling(self) -> SamplingConfig:
        return self.generation_params.sampling

    @sampling.setter
    def sampling(self, value: SamplingConfig) -> None:
        self.generation_params.sampling = value

    @property
    def method(self) -> MethodConfig:
        return self.generation_params.method

    @method.setter
    def method(self, value: MethodConfig) -> None:
        self.generation_params.method = value

    def to_dict(self) -> JsonDict:
        return {
            "metadata": self.metadata.to_dict(),
            "task": self.task.to_dict(),
            "backend": self.backend.to_dict(),
            "generation_params": self.generation_params.to_dict(),
            "execution": self.execution.to_dict(),
            "debug_refs": dict(self.debug_refs),
        }

    @classmethod
    def from_dict(cls, data: dict[str, JsonValue]) -> "RunConfig":
        if "generation_params" in data:
            generation_params = GenerationParams.from_dict(
                data.get("generation_params")
            )
        else:
            generation_params = GenerationParams(
                sampling=SamplingConfig.from_dict(data.get("sampling")),
                method=MethodConfig.from_dict(data.get("method")),
            )
        return cls(
            metadata=RunMetadata.from_dict(data["metadata"]),
            task=TaskConfig.from_dict(data["task"]),
            backend=BackendConfig.from_dict(data["backend"]),
            generation_params=generation_params,
            execution=ExecutionConfig.from_dict(data.get("execution")),
            debug_refs=_str_dict(data.get("debug_refs")),
        )

    def to_yaml(self, path: str | Path) -> None:
        Path(path).write_text(
            yaml.safe_dump(self.to_dict(), sort_keys=False),
            encoding="utf-8",
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "RunConfig":
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data)


@dataclass(slots=True)
class RunManifest:
    # Describes one local run directory and its portable artifact locations.
    manifest_version: str
    run_id: str
    created_at_utc: str
    config: RunConfig
    artifact_paths: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> JsonDict:
        return {
            "manifest_version": self.manifest_version,
            "run_id": self.run_id,
            "created_at_utc": self.created_at_utc,
            "config": self.config.to_dict(),
            "artifact_paths": dict(self.artifact_paths),
        }

    @classmethod
    def from_dict(cls, data: dict[str, JsonValue]) -> "RunManifest":
        return cls(
            manifest_version=data["manifest_version"],
            run_id=data["run_id"],
            created_at_utc=data["created_at_utc"],
            config=RunConfig.from_dict(data["config"]),
            artifact_paths=_str_dict(data.get("artifact_paths")),
        )

    def to_json(self, path: str | Path) -> None:
        _write_json(path, self.to_dict())

    @classmethod
    def from_json(cls, path: str | Path) -> "RunManifest":
        return cls.from_dict(_read_json(path))


@dataclass(slots=True)
class SampleInput:
    # Carries one task sample before generation or scoring happens.
    sample_id: str
    prompt: str
    source: JsonDict = field(default_factory=dict)
    reference: JsonValue = None
    metadata: JsonDict = field(default_factory=dict)

    def to_dict(self) -> JsonDict:
        return {
            "sample_id": self.sample_id,
            "prompt": self.prompt,
            "source": dict(self.source),
            "reference": self.reference,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, JsonValue]) -> "SampleInput":
        return cls(
            sample_id=data["sample_id"],
            prompt=data["prompt"],
            source=_json_dict(data.get("source")),
            reference=data.get("reference"),
            metadata=_json_dict(data.get("metadata")),
        )


@dataclass(slots=True)
class GenerationResult:
    # Stores generation-only output and backend-call metadata for one sample.
    """Backend output for one sample.

    `generate_wall_time_s` is backend-call wall time, not sample-isolated wall
    time. When one backend generate call serves multiple samples, those samples
    should share the same `generate_call_id` and the same call-level wall time.
    """

    generated_text: str
    generated_token_ids: list[int]
    parsed_answer: JsonValue = None
    n_prompt_tokens: int | None = None
    n_out_tokens: int | None = None
    generate_call_id: str | None = None
    generate_wall_time_s: float | None = None
    q_draft_steps: int | None = None
    v_verify_steps: int | None = None
    accept_lengths: list[int] | None = None
    hit_max_new_tokens: bool | None = None
    method_stats: JsonDict = field(default_factory=dict)
    debug_refs: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> JsonDict:
        return {
            "generated_text": self.generated_text,
            "generated_token_ids": list(self.generated_token_ids),
            "parsed_answer": self.parsed_answer,
            "n_prompt_tokens": self.n_prompt_tokens,
            "n_out_tokens": self.n_out_tokens,
            "generate_call_id": self.generate_call_id,
            "generate_wall_time_s": self.generate_wall_time_s,
            "q_draft_steps": self.q_draft_steps,
            "v_verify_steps": self.v_verify_steps,
            "accept_lengths": _int_list(self.accept_lengths),
            "hit_max_new_tokens": self.hit_max_new_tokens,
            "method_stats": dict(self.method_stats),
            "debug_refs": dict(self.debug_refs),
        }

    @classmethod
    def from_dict(cls, data: dict[str, JsonValue]) -> "GenerationResult":
        return cls(
            generated_text=data["generated_text"],
            generated_token_ids=list(data.get("generated_token_ids", [])),
            parsed_answer=data.get("parsed_answer"),
            n_prompt_tokens=data.get("n_prompt_tokens"),
            n_out_tokens=data.get("n_out_tokens"),
            generate_call_id=data.get("generate_call_id"),
            generate_wall_time_s=data.get("generate_wall_time_s"),
            q_draft_steps=data.get("q_draft_steps"),
            v_verify_steps=data.get("v_verify_steps"),
            accept_lengths=_int_list(data.get("accept_lengths")),
            hit_max_new_tokens=data.get("hit_max_new_tokens"),
            method_stats=_json_dict(data.get("method_stats")),
            debug_refs=_str_dict(data.get("debug_refs")),
        )


@dataclass(slots=True)
class ScoredSample:
    # Wraps one sample's input, generation result, and evaluation payload.
    run_id: str
    task_name: str
    measurement_scope: MeasurementScope
    decode_mode: DecodeMode
    sample: SampleInput
    generation: GenerationResult
    score_payload: JsonDict = field(default_factory=dict)
    metadata: JsonDict = field(default_factory=dict)
    debug_refs: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> JsonDict:
        return {
            "run_id": self.run_id,
            "task_name": self.task_name,
            "measurement_scope": self.measurement_scope.value,
            "decode_mode": self.decode_mode.value,
            "sample": self.sample.to_dict(),
            "generation": self.generation.to_dict(),
            "score_payload": dict(self.score_payload),
            "metadata": dict(self.metadata),
            "debug_refs": dict(self.debug_refs),
        }

    @classmethod
    def from_dict(cls, data: dict[str, JsonValue]) -> "ScoredSample":
        return cls(
            run_id=data["run_id"],
            task_name=data["task_name"],
            measurement_scope=MeasurementScope(data["measurement_scope"]),
            decode_mode=DecodeMode(data["decode_mode"]),
            sample=SampleInput.from_dict(data["sample"]),
            generation=GenerationResult.from_dict(data["generation"]),
            score_payload=_json_dict(data.get("score_payload")),
            metadata=_json_dict(data.get("metadata")),
            debug_refs=_str_dict(data.get("debug_refs")),
        )

    def to_json(self, path: str | Path) -> None:
        _write_json(path, self.to_dict())

    @classmethod
    def from_json(cls, path: str | Path) -> "ScoredSample":
        return cls.from_dict(_read_json(path))


@dataclass(slots=True)
class RunSummary:
    # Stores generic run-level aggregates for later analysis and comparison.
    run_id: str
    task_name: str
    measurement_scope: MeasurementScope
    decode_mode: DecodeMode
    backend_name: str
    model: str
    draft_model: str | None = None
    batch_size: int | None = None
    n_samples: int = 0
    n_generate_calls: int | None = None
    total_prompt_tokens: int | None = None
    total_output_tokens: int | None = None
    total_generate_wall_time_s: float | None = None
    score_summary: JsonDict = field(default_factory=dict)
    efficiency_summary: JsonDict = field(default_factory=dict)
    method_summary: JsonDict = field(default_factory=dict)
    debug_refs: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> JsonDict:
        return {
            "run_id": self.run_id,
            "task_name": self.task_name,
            "measurement_scope": self.measurement_scope.value,
            "decode_mode": self.decode_mode.value,
            "backend_name": self.backend_name,
            "model": self.model,
            "draft_model": self.draft_model,
            "batch_size": self.batch_size,
            "n_samples": self.n_samples,
            "n_generate_calls": self.n_generate_calls,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_generate_wall_time_s": self.total_generate_wall_time_s,
            "score_summary": dict(self.score_summary),
            "efficiency_summary": dict(self.efficiency_summary),
            "method_summary": dict(self.method_summary),
            "debug_refs": dict(self.debug_refs),
        }

    @classmethod
    def from_dict(cls, data: dict[str, JsonValue]) -> "RunSummary":
        return cls(
            run_id=data["run_id"],
            task_name=data["task_name"],
            measurement_scope=MeasurementScope(data["measurement_scope"]),
            decode_mode=DecodeMode(data["decode_mode"]),
            backend_name=data["backend_name"],
            model=data["model"],
            draft_model=data.get("draft_model"),
            batch_size=data.get("batch_size"),
            n_samples=data.get("n_samples", 0),
            n_generate_calls=data.get("n_generate_calls"),
            total_prompt_tokens=data.get("total_prompt_tokens"),
            total_output_tokens=data.get("total_output_tokens"),
            total_generate_wall_time_s=data.get("total_generate_wall_time_s"),
            score_summary=_json_dict(data.get("score_summary")),
            efficiency_summary=_json_dict(data.get("efficiency_summary")),
            method_summary=_json_dict(data.get("method_summary")),
            debug_refs=_str_dict(data.get("debug_refs")),
        )

    def to_json(self, path: str | Path) -> None:
        _write_json(path, self.to_dict())

    @classmethod
    def from_json(cls, path: str | Path) -> "RunSummary":
        return cls.from_dict(_read_json(path))


def write_scored_samples_jsonl(
    path: str | Path, samples: list[ScoredSample]
) -> None:
    lines = [json.dumps(sample.to_dict()) for sample in samples]
    text = "\n".join(lines)
    if text:
        text += "\n"
    Path(path).write_text(text, encoding="utf-8")


def read_scored_samples_jsonl(path: str | Path) -> list[ScoredSample]:
    samples: list[ScoredSample] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        samples.append(ScoredSample.from_dict(json.loads(line)))
    return samples


def _write_json(path: str | Path, data: JsonDict) -> None:
    text = json.dumps(data, indent=2)
    Path(path).write_text(f"{text}\n", encoding="utf-8")


def _read_json(path: str | Path) -> JsonDict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


__all__ = [
    "BackendConfig",
    "DecodeMode",
    "ExecutionConfig",
    "GenerationParams",
    "GenerationResult",
    "MeasurementScope",
    "MethodConfig",
    "RunConfig",
    "RunManifest",
    "RunMetadata",
    "RunSummary",
    "SampleInput",
    "SamplingConfig",
    "ScoredSample",
    "TaskConfig",
    "TaskDefaults",
    "read_scored_samples_jsonl",
    "write_scored_samples_jsonl",
]
