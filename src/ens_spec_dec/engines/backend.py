"""Thin generation-backend contracts for offline benchmark runs.

This module keeps the engine boundary small and benchmark-agnostic.
Tasks own prompt construction, parsing, and scoring. Engines only accept
already-built prompts and return generation outputs plus batch-level metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from ens_spec_dec.contracts import (
    BackendConfig,
    DecodeMode,
    ExecutionConfig,
    GenerationParams,
    JsonDict,
)

GenerationParamsBatch = GenerationParams | list[GenerationParams]


@dataclass(slots=True)
class BackendRequestConfig:
    # Carries the small config surface needed by the backend only.
    decode_mode: DecodeMode
    backend: BackendConfig
    execution: ExecutionConfig
    generation_params: GenerationParams


@dataclass(slots=True)
class BackendSampleOutput:
    # Holds the backend output for one prompt in a batch.
    generated_text: str
    generated_token_ids: list[int]
    n_prompt_tokens: int | None = None
    n_out_tokens: int | None = None
    method_stats: JsonDict = field(default_factory=dict)


@dataclass(slots=True)
class GenerateBatchResult:
    # Holds one backend call result shared across a batch of prompts.
    outputs: list[BackendSampleOutput]
    generate_call_id: str
    generate_wall_time_s: float
    aggregate_method_stats: JsonDict = field(default_factory=dict)


class GenerationBackend(Protocol):
    def generate_batch(
        self, prompts: list[str], generation_params: GenerationParamsBatch
    ) -> GenerateBatchResult: ...

    def get_run_debug_payload(self) -> JsonDict: ...


__all__ = [
    "BackendRequestConfig",
    "BackendSampleOutput",
    "GenerateBatchResult",
    "GenerationParamsBatch",
    "GenerationBackend",
]
