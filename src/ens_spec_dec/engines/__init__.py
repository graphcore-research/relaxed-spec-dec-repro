"""Generation backend invocation modules."""

from ens_spec_dec.engines.backend import (
    BackendRequestConfig,
    BackendSampleOutput,
    GenerateBatchResult,
    GenerationParamsBatch,
    GenerationBackend,
)

try:
    from ens_spec_dec.engines.vllm_offline import OfflineVllmBackend
except ModuleNotFoundError:  # pragma: no cover - exercised in non-vLLM envs.
    OfflineVllmBackend = None

__all__ = [
    "BackendRequestConfig",
    "BackendSampleOutput",
    "GenerateBatchResult",
    "GenerationParamsBatch",
    "GenerationBackend",
    "OfflineVllmBackend",
]
