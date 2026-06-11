# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
import json
import math
import os
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from vllm.logger import init_logger
from vllm.triton_utils import tl, triton
from vllm.v1.outputs import LogprobsLists, LogprobsTensors, SamplerOutput
from vllm.v1.sample.logits_processor.builtin import MinTokensLogitsProcessor
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.ops.bad_words import apply_bad_words_with_drafts
from vllm.v1.sample.ops.penalties import apply_all_penalties
from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p
from vllm.v1.sample.sampler import Sampler
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
from vllm.v1.spec_decode.utils import unconditional_to_conditional_rates

if TYPE_CHECKING:
    from vllm.config.speculative import SpeculativeConfig

logger = init_logger(__name__)

PLACEHOLDER_TOKEN_ID: tl.constexpr = -1
GREEDY_TEMPERATURE: tl.constexpr = 0
# Maximum number of speculative draft tokens allowed per request in a single
# step. This value is chosen to be large enough to handle typical use cases.
MAX_SPEC_LEN = 128
RELAXED_KERNEL_IMPL_ENV = "VLLM_CACTUS_RELAXED_KERNEL_IMPL"
RELAXED_KERNEL_IMPL_DEFAULT = "vanilla"
RELAXED_KERNEL_IMPL_CHOICES = ("vanilla", "fused", "auto")

RELAXED_DENSE_IMPL_ENV = "VLLM_CACTUS_RELAXED_DENSE_IMPL"
RELAXED_DENSE_IMPL_DEFAULT = "vanilla"
RELAXED_DENSE_IMPL_CHOICES = ("vanilla", "inplace")
ENTROPY_STAT_NAMES = (
    "drafter_accepted",
    "verifier_on_accepted",
    "verifier_on_output",
)
ENTROPY_UNIT = "nats"
ENTROPY_DISTRIBUTION = "processed_logits_after_temperature_top_p_top_k"
SCD_DEBUG_ENV = "SPEC_DEC_DEBUG_SCD_NAN"
SCD_DEBUG_PATH_ENV = "SPEC_DEC_DEBUG_SCD_NAN_PATH"
SCD_DEBUG_DEFAULT_PATH = "/tmp/spec-dec-scd-debug/scd_nan_debug.jsonl"


def _scd_debug_enabled() -> bool:
    value = os.getenv(SCD_DEBUG_ENV)
    return value is not None and value.lower() not in ("", "0", "false", "no")


def _scd_debug_tensor_to_list(tensor: torch.Tensor | None) -> list[object] | None:
    if tensor is None:
        return None
    return tensor.detach().cpu().tolist()


def _scd_debug_number(value: torch.Tensor) -> float:
    return float(value.detach().to("cpu", torch.float32).item())


def _scd_debug_tensor_stats(
    name: str,
    tensor: torch.Tensor | None,
    *,
    gate: torch.Tensor | None = None,
    draft_token_ids: torch.Tensor | None = None,
) -> dict[str, object]:
    if tensor is None:
        return {"name": name, "present": False}

    with torch.no_grad():
        data = tensor.detach()
        flat = data.reshape(data.shape[0], -1) if data.ndim > 0 else data.reshape(1, 1)
        nan_mask = torch.isnan(flat)
        inf_mask = torch.isinf(flat)
        finite_mask = torch.isfinite(flat)

        row_has_nan = nan_mask.any(dim=-1)
        row_has_nonfinite = (~finite_mask).any(dim=-1)
        first_nan_row = None
        first_nonfinite_row = None
        nan_rows = torch.nonzero(row_has_nan, as_tuple=False).flatten()
        nonfinite_rows = torch.nonzero(row_has_nonfinite, as_tuple=False).flatten()
        if nan_rows.numel() > 0:
            first_nan_row = int(nan_rows[0].item())
        if nonfinite_rows.numel() > 0:
            first_nonfinite_row = int(nonfinite_rows[0].item())

        finite_values = flat[finite_mask]
        stats: dict[str, object] = {
            "name": name,
            "present": True,
            "shape": list(data.shape),
            "dtype": str(data.dtype),
            "device": str(data.device),
            "nan_count": int(nan_mask.sum().item()),
            "inf_count": int(inf_mask.sum().item()),
            "finite_count": int(finite_mask.sum().item()),
            "first_nan_row": first_nan_row,
            "first_nonfinite_row": first_nonfinite_row,
            "finite_min": (
                _scd_debug_number(finite_values.min())
                if finite_values.numel() > 0
                else None
            ),
            "finite_max": (
                _scd_debug_number(finite_values.max())
                if finite_values.numel() > 0
                else None
            ),
        }

        if gate is not None and gate.shape == data.shape:
            flat_gate = gate.detach().reshape(flat.shape).to(torch.bool)
            stats.update(
                {
                    "gate_true_count": int(flat_gate.sum().item()),
                    "nan_in_gate_count": int(nan_mask[flat_gate].sum().item()),
                    "inf_in_gate_count": int(inf_mask[flat_gate].sum().item()),
                    "finite_in_gate_count": int(finite_mask[flat_gate].sum().item()),
                }
            )

        row = first_nan_row if first_nan_row is not None else first_nonfinite_row
        if row is not None and data.ndim == 2 and draft_token_ids is not None:
            token_ids = draft_token_ids.detach().reshape(-1)
            if row < token_ids.numel():
                token_id = int(token_ids[row].item())
                stats["selected_token_id"] = token_id
                if 0 <= token_id < data.shape[1]:
                    value = data[row, token_id]
                    stats["selected_token_value"] = _scd_debug_number(value)
                    stats["selected_token_is_nan"] = bool(torch.isnan(value).item())
                    stats["selected_token_is_finite"] = bool(
                        torch.isfinite(value).item()
                    )
                    if gate is not None and gate.shape == data.shape:
                        stats["selected_token_in_gate"] = bool(
                            gate[row, token_id].item()
                        )
        return stats


def _write_scd_debug_event(event: dict[str, object]) -> None:
    if not _scd_debug_enabled():
        return
    path = os.getenv(SCD_DEBUG_PATH_ENV, SCD_DEBUG_DEFAULT_PATH)
    try:
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(path, "a", encoding="utf-8") as handle:
            json.dump(event, handle, sort_keys=True)
            handle.write("\n")
    except Exception:
        logger.exception("Failed to write SCD NaN debug event to %s", path)


def _log_scd_debug_tensors(
    stage: str,
    tensors: dict[str, torch.Tensor | None],
    *,
    gate: torch.Tensor | None = None,
    draft_token_ids: torch.Tensor | None = None,
    extra: dict[str, object] | None = None,
) -> None:
    if not _scd_debug_enabled():
        return
    payload: dict[str, object] = {
        "stage": stage,
        "extra": extra or {},
        "tensors": [
            _scd_debug_tensor_stats(
                name,
                tensor,
                gate=gate,
                draft_token_ids=draft_token_ids,
            )
            for name, tensor in tensors.items()
        ],
    }
    _write_scd_debug_event(payload)


def entropy_from_processed_logits(
    logits: torch.Tensor,
    is_greedy: torch.Tensor | None = None,
    all_greedy: bool = False,
) -> torch.Tensor:
    """Return row entropy in nats from processed logits [N, V]."""
    if logits.ndim != 2:
        raise ValueError("entropy logits must have shape [N, V].")
    if logits.shape[0] == 0:
        return logits.new_zeros((0,), dtype=torch.float32)
    if all_greedy:
        return logits.new_zeros((logits.shape[0],), dtype=torch.float32)

    logits = logits.to(torch.float32)
    finite = torch.isfinite(logits)
    has_support = finite.any(dim=-1)
    safe_logits = logits.masked_fill(~finite, float("-inf"))
    log_z = torch.logsumexp(safe_logits, dim=-1)
    log_z = torch.where(has_support, log_z, torch.zeros_like(log_z))
    probs = torch.exp(safe_logits - log_z.unsqueeze(-1))
    probs = torch.where(finite & has_support.unsqueeze(-1), probs, 0.0)
    expected_logit = (probs * logits.masked_fill(~finite, 0.0)).sum(dim=-1)
    entropy = torch.where(has_support, log_z - expected_logit, 0.0)
    entropy = entropy.clamp_min(0.0)
    if is_greedy is not None:
        entropy = torch.where(is_greedy, torch.zeros_like(entropy), entropy)
    return entropy.contiguous()


def _entropy_field(
    entropy_sum: float,
    token_count: int,
) -> dict[str, float | int | None]:
    return {
        "entropy_sum": float(entropy_sum),
        "token_count": int(token_count),
        "mean_entropy": (float(entropy_sum) / token_count) if token_count else None,
    }


def _entropy_payloads_from_outputs(
    output_token_ids: torch.Tensor,
    all_accepted_mask: torch.Tensor,
    num_draft_tokens: list[int],
    draft_entropies: torch.Tensor | None,
    target_entropies: torch.Tensor | None,
    bonus_entropies: torch.Tensor | None,
    vocab_size: int,
) -> list[dict[str, object]] | None:
    """Summarize per-request entropy without storing token-level values."""
    if target_entropies is None:
        return None

    output_rows = output_token_ids.detach().cpu().tolist()
    all_accepted = [bool(value) for value in all_accepted_mask.detach().cpu().tolist()]
    target_values = target_entropies.detach().to("cpu", torch.float32).tolist()
    draft_values = (
        None if draft_entropies is None
        else draft_entropies.detach().to("cpu", torch.float32).tolist()
    )
    bonus_values = (
        None if bonus_entropies is None
        else bonus_entropies.detach().to("cpu", torch.float32).tolist()
    )

    payloads: list[dict[str, object]] = []
    flat_offset = 0
    for row_idx, num_draft in enumerate(num_draft_tokens):
        row = output_rows[row_idx]
        target_slice = target_values[flat_offset : flat_offset + num_draft]
        draft_slice = (
            None if draft_values is None
            else draft_values[flat_offset : flat_offset + num_draft]
        )
        flat_offset += num_draft

        valid_draft_positions = [
            pos
            for pos in range(num_draft)
            if row[pos] != PLACEHOLDER_TOKEN_ID and 0 <= row[pos] < vocab_size
        ]
        if all_accepted[row_idx]:
            accepted_count = num_draft
        else:
            accepted_count = max(len(valid_draft_positions) - 1, 0)
        accepted_positions = valid_draft_positions[:accepted_count]

        drafter_sum = 0.0
        drafter_count = 0
        if draft_slice is not None:
            drafter_sum = sum(float(draft_slice[pos]) for pos in accepted_positions)
            drafter_count = len(accepted_positions)

        verifier_accepted_sum = sum(
            float(target_slice[pos]) for pos in accepted_positions
        )
        verifier_output_sum = sum(
            float(target_slice[pos]) for pos in valid_draft_positions
        )
        verifier_output_count = len(valid_draft_positions)

        bonus_pos = num_draft
        if (
            all_accepted[row_idx]
            and bonus_values is not None
            and bonus_pos < len(row)
            and row[bonus_pos] != PLACEHOLDER_TOKEN_ID
            and 0 <= row[bonus_pos] < vocab_size
        ):
            verifier_output_sum += float(bonus_values[row_idx])
            verifier_output_count += 1

        payload = {
            "version": 1,
            "unit": ENTROPY_UNIT,
            "distribution": ENTROPY_DISTRIBUTION,
            "drafter_accepted": _entropy_field(drafter_sum, drafter_count),
            "verifier_on_accepted": _entropy_field(
                verifier_accepted_sum, len(accepted_positions)
            ),
            "verifier_on_output": _entropy_field(
                verifier_output_sum, verifier_output_count
            ),
        }
        payloads.append(payload)
    return payloads


def use_fused_relaxed_kernel_by_default() -> bool:
    """Return whether fused relaxed kernels should be used for eligible batches.

    The profiled GH200 default is the vLLM-native vanilla path. Set
    VLLM_CACTUS_RELAXED_KERNEL_IMPL=fused or auto to restore the previous
    all-random fused behavior for diagnostics.
    """
    impl = os.environ.get(
        RELAXED_KERNEL_IMPL_ENV, RELAXED_KERNEL_IMPL_DEFAULT
    ).strip().lower()
    if impl not in RELAXED_KERNEL_IMPL_CHOICES:
        raise ValueError(
            f"{RELAXED_KERNEL_IMPL_ENV} must be one of "
            f"{RELAXED_KERNEL_IMPL_CHOICES}, got {impl!r}."
        )
    return impl in ("fused", "auto")


def use_inplace_relaxed_dense_transforms() -> bool:
    """Return whether dense relaxed transforms may reuse scratch/storage."""
    impl = os.environ.get(
        RELAXED_DENSE_IMPL_ENV, RELAXED_DENSE_IMPL_DEFAULT
    ).strip().lower()
    if impl not in RELAXED_DENSE_IMPL_CHOICES:
        raise ValueError(
            f"{RELAXED_DENSE_IMPL_ENV} must be one of "
            f"{RELAXED_DENSE_IMPL_CHOICES}, got {impl!r}."
        )
    return impl == "inplace"


class RejectionSampler(nn.Module):
    """
    The implementation strictly follows the algorithm described in
        https://arxiv.org/abs/2211.17192.
    However, we want to clarify the terminology used in the implementation:
    accepted tokens: tokens that are accepted based on the relationship
            between the "raw" draft and target probabilities.
    recovered tokens: tokens that are sampled based on the adjusted probability
        distribution, which is derived from both the draft and target
        probabilities.
    bonus tokens:
        If all proposed tokens are accepted, the bonus token is added to the
        end of the sequence. The bonus token is only sampled from the target
        probabilities. We pass in the bonus tokens instead of sampling them
        in the rejection sampler to allow for more flexibility in the
        sampling process. For example, we can use top_p, top_k sampling for
        bonus tokens, while spec decode does not support these sampling
        strategies.
    output tokens:
        Tokens are finally generated with the rejection sampler.
        output tokens = accepted tokens + recovered tokens + bonus tokens
    """

    def __init__(
        self,
        sampler: Sampler,
        spec_config: SpeculativeConfig | None = None,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.sampler = sampler
        logprobs_mode = self.sampler.logprobs_mode
        self.is_processed_logprobs_mode = logprobs_mode.startswith("processed")
        self.is_logits_logprobs_mode = logprobs_mode.endswith("logits")

        self.synthetic_conditional_rates: torch.Tensor | None = None
        if (
            spec_config is not None
            and spec_config.rejection_sample_method == "synthetic"
        ):
            assert spec_config.synthetic_acceptance_rates is not None
            self.synthetic_conditional_rates = torch.tensor(
                unconditional_to_conditional_rates(
                    spec_config.synthetic_acceptance_rates
                ),
                dtype=torch.float32,
                device=device,
            )
        self.synthetic_mode = self.synthetic_conditional_rates is not None
        self.relaxed_target_method = (
            "none" if spec_config is None else spec_config.relaxed_target_method
        )
        if self.relaxed_target_method == "fuzzy":
            self.relaxed_target_method = "rfsd"
        self.cactus_delta = None if spec_config is None else spec_config.cactus_delta
        self.verifier_weight = (
            None if spec_config is None else spec_config.verifier_weight
        )
        self.fuzzy_divergence = (
            None if spec_config is None else spec_config.fuzzy_divergence
        )
        self.fuzzy_threshold = (
            None if spec_config is None else spec_config.fuzzy_threshold
        )
        self.spec_cascade_alpha = (
            None if spec_config is None else spec_config.spec_cascade_alpha
        )
        self.spec_cascade_opt_gate = (
            "processed"
            if spec_config is None
            else getattr(spec_config, "spec_cascade_opt_gate", "processed")
        )
        self.spec_cascade_tok3_top_set = (
            "paper" if spec_config is None else spec_config.spec_cascade_tok3_top_set
        )
        self.lossy_alpha = None if spec_config is None else spec_config.lossy_alpha
        self.scd_beta = None if spec_config is None else getattr(
            spec_config, "scd_beta", None
        )
        self.scd_temperature = None if spec_config is None else getattr(
            spec_config, "scd_temperature", None
        )
        self.scd_alpha = None if spec_config is None else getattr(
            spec_config, "scd_alpha", None
        )
        self.relaxed_bonus_token_policy = (
            "target_p"
            if spec_config is None
            else spec_config.relaxed_bonus_token_policy
        )
        self.entropy_monitoring = bool(
            spec_config is not None
            and getattr(spec_config, "entropy_monitoring", False)
        )
        self.last_all_accepted_mask: torch.Tensor | None = None
        self.last_bonus_target_probs: torch.Tensor | None = None
        self.last_bonus_opt_gate_probs: torch.Tensor | None = None
        self.last_bonus_tok3_top_mask: torch.Tensor | None = None
        self.last_bonus_scd_rule_logits: torch.Tensor | None = None
        self.last_bonus_scd_processed_logits: torch.Tensor | None = None

    def forward(
        self,
        metadata: SpecDecodeMetadata,
        # [num_tokens, vocab_size]
        draft_probs: torch.Tensor | None,
        # [num_tokens, vocab_size]
        draft_rule_logits: torch.Tensor | None,
        # [num_tokens]
        draft_entropies: torch.Tensor | None,
        # [num_tokens + batch_size, vocab_size]
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> SamplerOutput:
        """
        Args:
            metadata:
                Metadata for spec decoding.
            draft_probs (Optional[torch.Tensor]):
                Probability distribution for the draft tokens. Shape is
                [num_tokens, vocab_size]. Can be None if probabilities are
                not provided, which is the case for ngram spec decode.
            draft_rule_logits (Optional[torch.Tensor]):
                Draft logits after rule processors and before draft
                temperature/top-p/top-k. Required by SCD relaxed targets.
            draft_entropies (Optional[torch.Tensor]):
                Draft entropy rows aligned with draft_probs, shape [num_tokens].
            logits (torch.Tensor):
                Target model's logits probability distribution.
                Shape is [num_tokens + batch_size, vocab_size]. Here,
                probabilities from different requests are flattened into a
                single tensor because this is the shape of the output logits.
                NOTE: `logits` can be updated in place to save memory.
            sampling_metadata (vllm.v1.sample.metadata.SamplingMetadata):
                Additional metadata needed for sampling, such as temperature,
                top-k/top-p parameters, or other relevant information.
        Returns:
            SamplerOutput:
                Contains the final output token IDs and their logprobs if
                requested.
        """
        assert metadata.max_spec_len <= MAX_SPEC_LEN
        self.last_all_accepted_mask = None
        self.last_bonus_target_probs = None
        self.last_bonus_opt_gate_probs = None
        self.last_bonus_tok3_top_mask = None
        self.last_bonus_scd_rule_logits = None
        self.last_bonus_scd_processed_logits = None

        bonus_logits_indices = metadata.bonus_logits_indices
        target_logits_indices = metadata.target_logits_indices

        # When indexing with a tensor (bonus_logits_indices), PyTorch
        # creates a new tensor with separate storage from the original
        # logits tensor. This means any in-place operations on bonus_logits
        # won't affect the original logits tensor.
        assert logits is not None
        bonus_logits = logits[bonus_logits_indices]
        bonus_logits_for_entropy = (
            bonus_logits.clone() if self.entropy_monitoring else None
        )
        bonus_sampler_output = None
        if self.relaxed_bonus_token_policy == "target_p":
            bonus_sampler_output = self.sampler(
                logits=bonus_logits,
                sampling_metadata=replace(
                    sampling_metadata,
                    max_num_logprobs=-1,
                ),
                predict_bonus_token=True,
                # Override the logprobs mode to return logits because they are
                # needed later to compute the accepted token logprobs.
                logprobs_mode_override="processed_logits"
                if self.is_processed_logprobs_mode
                else "raw_logits",
            )
            bonus_token_ids = bonus_sampler_output.sampled_token_ids
        elif self.relaxed_bonus_token_policy == "relaxed_T_qp":
            if sampling_metadata.max_num_logprobs is not None:
                raise RuntimeError(
                    "relaxed_T_qp bonus sampling does not support logprobs yet."
                )
            needs_bonus_target_probs = self.relaxed_target_method in (
                "cactus",
                "rfsd",
                "fsd",
                "spec_cascade_opt",
                "spec_cascade_tok3",
                "lossy_spec_decode_beta1",
            )
            needs_bonus_scd_logits = self.relaxed_target_method in (
                "scd_expert_toppk_gated",
                "scd_alpha",
            )
            bonus_logits_for_probs = (
                bonus_logits.clone()
                if needs_bonus_target_probs or needs_bonus_scd_logits
                else None
            )
            bonus_sampler_output = self.sampler(
                logits=bonus_logits,
                sampling_metadata=replace(
                    sampling_metadata,
                    max_num_logprobs=-1,
                ),
                predict_bonus_token=True,
                logprobs_mode_override="raw_logits",
            )
            bonus_token_ids = bonus_sampler_output.sampled_token_ids
            if bonus_logits_for_probs is not None:
                if needs_bonus_scd_logits:
                    (
                        _bonus_probs,
                        self.last_bonus_scd_rule_logits,
                        self.last_bonus_scd_processed_logits,
                    ) = bonus_target_probs_and_logits(
                        bonus_logits_for_probs, sampling_metadata, self.sampler
                    )
                elif (
                    self.relaxed_target_method == "spec_cascade_opt"
                    and self.spec_cascade_opt_gate == "paper"
                ):
                    (
                        self.last_bonus_target_probs,
                        bonus_paper_logits,
                        _bonus_processed_logits,
                    ) = bonus_target_probs_and_logits(
                        bonus_logits_for_probs, sampling_metadata, self.sampler
                    )
                    self.last_bonus_opt_gate_probs = bonus_paper_logits.softmax(
                        dim=-1,
                        dtype=torch.float32,
                    )
                elif self.relaxed_target_method == "spec_cascade_tok3":
                    (
                        self.last_bonus_target_probs,
                        bonus_paper_logits,
                        bonus_processed_logits,
                    ) = bonus_target_probs_and_logits(
                        bonus_logits_for_probs, sampling_metadata, self.sampler
                    )
                    bonus_top_logits = (
                        bonus_paper_logits
                        if self.spec_cascade_tok3_top_set == "paper"
                        else bonus_processed_logits
                    )
                    self.last_bonus_tok3_top_mask = (
                        spec_cascade_tok3_top_mask_from_logits(
                            bonus_top_logits,
                            self._require_spec_cascade_alpha(),
                        )
                    )
                else:
                    self.last_bonus_target_probs = bonus_target_probs(
                        bonus_logits_for_probs, sampling_metadata, self.sampler
                    )
        else:
            raise ValueError(
                f"Unknown relaxed_bonus_token_policy: "
                f"{self.relaxed_bonus_token_policy}"
            )

        bonus_entropies = None
        if self.entropy_monitoring and bonus_logits_for_entropy is not None:
            _bonus_probs, _paper_logits, bonus_processed_logits = (
                bonus_target_probs_and_logits(
                    bonus_logits_for_entropy,
                    sampling_metadata,
                    self.sampler,
                )
            )
            bonus_is_greedy = None
            if (
                not sampling_metadata.all_random
                and sampling_metadata.temperature is not None
            ):
                bonus_is_greedy = sampling_metadata.temperature == GREEDY_TEMPERATURE
            bonus_entropies = entropy_from_processed_logits(
                bonus_processed_logits,
                is_greedy=bonus_is_greedy,
                all_greedy=sampling_metadata.all_greedy,
            )

        # Just like `bonus_logits`, `target_logits` is a new tensor with
        # separate storage from the original `logits` tensor. Therefore,
        # it is safe to update `target_logits` in place.
        raw_target_logits = logits[target_logits_indices]
        # Use float32 for the target_logits.
        raw_target_logits = raw_target_logits.to(torch.float32)
        target_logits = raw_target_logits
        if not self.is_processed_logprobs_mode:
            # Clone raw_target_logits before applying processors to preserve
            # the original raw logits for logprobs computation, since
            # apply_logits_processors modifies the tensor in-place.
            target_logits = target_logits.clone()
        target_logits = self.apply_logits_processors(
            target_logits, sampling_metadata, metadata
        )
        opt_needs_paper_logits = (
            self.relaxed_target_method == "spec_cascade_opt"
            and self.spec_cascade_opt_gate == "paper"
        )
        spec_cascade_opt_rule_logits = (
            target_logits.clone() if opt_needs_paper_logits else None
        )
        tok3_needs_paper_logits = (
            self.relaxed_target_method == "spec_cascade_tok3"
            and self.spec_cascade_tok3_top_set == "paper"
        )
        spec_cascade_tok3_rule_logits = (
            target_logits.clone() if tok3_needs_paper_logits else None
        )
        scd_needs_rule_logits = self.relaxed_target_method in (
            "scd_expert_toppk_gated",
            "scd_alpha",
        )
        scd_expert_rule_logits = (
            target_logits.clone() if scd_needs_rule_logits else None
        )
        # [num_tokens, vocab_size]
        # NOTE(woosuk): `target_logits` can be updated in place inside the
        # `apply_sampling_constraints` function.
        target_logits = apply_sampling_constraints(
            target_logits,
            metadata.cu_num_draft_tokens,
            sampling_metadata,
        )
        if scd_needs_rule_logits:
            scd_debug_gate = (
                torch.isfinite(target_logits)
                if self.relaxed_target_method == "scd_expert_toppk_gated"
                else None
            )
            _log_scd_debug_tensors(
                "sampler_pre_rejection_scd",
                {
                    "raw_target_logits": raw_target_logits,
                    "scd_expert_rule_logits": scd_expert_rule_logits,
                    "processed_target_logits": target_logits,
                    "draft_rule_logits": draft_rule_logits,
                    "draft_probs": draft_probs,
                },
                gate=scd_debug_gate,
                draft_token_ids=metadata.draft_token_ids,
                extra={
                    "relaxed_target_method": self.relaxed_target_method,
                    "scd_beta": self.scd_beta,
                    "scd_temperature": self.scd_temperature,
                    "scd_alpha": self.scd_alpha,
                    "num_draft_tokens": list(metadata.num_draft_tokens),
                    "max_spec_len": int(metadata.max_spec_len),
                    "cu_num_draft_tokens": _scd_debug_tensor_to_list(
                        metadata.cu_num_draft_tokens
                    ),
                },
            )

        target_entropies = None
        if self.entropy_monitoring:
            target_is_greedy = None
            if (
                not sampling_metadata.all_random
                and sampling_metadata.temperature is not None
                and target_logits.shape[0] > 0
            ):
                target_is_greedy = expand_batch_to_tokens(
                    sampling_metadata.temperature,
                    metadata.cu_num_draft_tokens,
                    target_logits.shape[0],
                    replace_from=GREEDY_TEMPERATURE,
                    replace_to=1,
                ) == GREEDY_TEMPERATURE
            target_entropies = entropy_from_processed_logits(
                target_logits,
                is_greedy=target_is_greedy,
                all_greedy=sampling_metadata.all_greedy,
            )

        output_token_ids, all_accepted_mask = rejection_sample(
            metadata.draft_token_ids,
            metadata.num_draft_tokens,
            metadata.max_spec_len,
            metadata.cu_num_draft_tokens,
            draft_probs,
            target_logits,
            bonus_token_ids,
            sampling_metadata,
            draft_rule_logits=draft_rule_logits,
            synthetic_mode=self.synthetic_mode,
            synthetic_conditional_rates=self.synthetic_conditional_rates,
            relaxed_target_method=self.relaxed_target_method,
            cactus_delta=self.cactus_delta,
            verifier_weight=self.verifier_weight,
            fuzzy_divergence=self.fuzzy_divergence,
            fuzzy_threshold=self.fuzzy_threshold,
            spec_cascade_alpha=self.spec_cascade_alpha,
            spec_cascade_opt_rule_logits=spec_cascade_opt_rule_logits,
            spec_cascade_opt_gate=self.spec_cascade_opt_gate,
            spec_cascade_tok3_rule_logits=spec_cascade_tok3_rule_logits,
            spec_cascade_tok3_top_set=self.spec_cascade_tok3_top_set,
            lossy_alpha=self.lossy_alpha,
            scd_beta=self.scd_beta,
            scd_temperature=self.scd_temperature,
            scd_alpha=self.scd_alpha,
            scd_expert_rule_logits=scd_expert_rule_logits,
        )
        self.last_all_accepted_mask = all_accepted_mask
        entropy_stats = None
        if self.entropy_monitoring:
            entropy_stats = _entropy_payloads_from_outputs(
                output_token_ids,
                all_accepted_mask,
                metadata.num_draft_tokens,
                draft_entropies,
                target_entropies,
                bonus_entropies,
                target_logits.shape[-1],
            )

        logprobs_tensors = None
        if sampling_metadata.max_num_logprobs is not None:
            assert bonus_sampler_output is not None
            logprobs_tensors = self._get_logprobs_tensors(
                sampling_metadata.max_num_logprobs,
                metadata,
                logits,
                target_logits if self.is_processed_logprobs_mode else raw_target_logits,
                bonus_sampler_output.logprobs_tensors.logprobs,
                output_token_ids,
            )

        sampler_output = SamplerOutput(
            sampled_token_ids=output_token_ids,
            logprobs_tensors=logprobs_tensors,
        )
        sampler_output.entropy_stats = entropy_stats
        return sampler_output

    def apply_relaxed_bonus(
        self,
        sampler_output: SamplerOutput,
        metadata: SpecDecodeMetadata,
        q_bonus_token_ids: torch.Tensor,
        q_bonus_probs: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        q_bonus_rule_logits: torch.Tensor | None = None,
    ) -> None:
        """Patch all-accepted rows with a relaxed target bonus sample.

        Shapes: q_bonus_token_ids [B], q_bonus_probs/rule_logits [B, V].
        The config spelling remains `relaxed_T_qp` for compatibility.
        """
        if self.relaxed_bonus_token_policy != "relaxed_T_qp":
            return
        if self.last_all_accepted_mask is None:
            raise RuntimeError("Missing all-accepted mask for relaxed_T_qp bonus.")

        num_draft_tokens = torch.tensor(
            metadata.num_draft_tokens,
            dtype=torch.int64,
            device=sampler_output.sampled_token_ids.device,
        )
        active_mask = self.last_all_accepted_mask & (num_draft_tokens > 0)
        row_ids = torch.nonzero(active_mask, as_tuple=False).flatten()
        if row_ids.numel() == 0:
            return

        if self.relaxed_target_method == "cactus":
            if self.last_bonus_target_probs is None:
                raise RuntimeError("Missing target bonus probabilities.")
            bonus_token_ids = sample_cactus_bonus_token_ids(
                self.last_bonus_target_probs[row_ids],
                q_bonus_probs[row_ids],
                q_bonus_token_ids[row_ids],
                self._require_cactus_delta(),
                sampling_metadata.generators,
                row_ids,
            )
        elif self.relaxed_target_method == "ensemble":
            target_bonus_token_ids = sampler_output.sampled_token_ids[
                row_ids, num_draft_tokens[row_ids]
            ]
            bonus_token_ids = sample_static_ensemble_bonus_token_ids(
                target_bonus_token_ids,
                q_bonus_token_ids[row_ids],
                self._require_verifier_weight(),
                sampling_metadata.generators,
                row_ids,
            )
        elif self.relaxed_target_method == "rfsd":
            if self.last_bonus_target_probs is None:
                raise RuntimeError("Missing target bonus probabilities.")
            target_bonus_token_ids = sampler_output.sampled_token_ids[
                row_ids, num_draft_tokens[row_ids]
            ]
            bonus_token_ids = sample_rfsd_bonus_token_ids(
                self.last_bonus_target_probs[row_ids],
                q_bonus_probs[row_ids],
                target_bonus_token_ids,
                q_bonus_token_ids[row_ids],
                self._require_fuzzy_divergence(),
                self._require_fuzzy_threshold(),
            )
        elif self.relaxed_target_method == "fsd":
            if self.last_bonus_target_probs is None:
                raise RuntimeError("Missing target bonus probabilities.")
            target_bonus_token_ids = sampler_output.sampled_token_ids[
                row_ids, num_draft_tokens[row_ids]
            ]
            bonus_token_ids = sample_fsd_bonus_token_ids(
                self.last_bonus_target_probs[row_ids],
                q_bonus_probs[row_ids],
                target_bonus_token_ids,
                q_bonus_token_ids[row_ids],
                self._require_fuzzy_divergence(),
                self._require_fuzzy_threshold(),
            )
        elif self.relaxed_target_method == "spec_cascade_opt":
            if self.last_bonus_target_probs is None:
                raise RuntimeError("Missing target bonus probabilities.")
            target_gate_probs = None
            draft_gate_probs = None
            if self.spec_cascade_opt_gate == "paper":
                if self.last_bonus_opt_gate_probs is None:
                    raise RuntimeError("Missing OPT bonus gate probabilities.")
                if q_bonus_rule_logits is None:
                    raise RuntimeError("Missing OPT q-bonus rule logits.")
                target_gate_probs = self.last_bonus_opt_gate_probs[row_ids]
                draft_gate_probs = q_bonus_rule_logits[row_ids].softmax(
                    dim=-1,
                    dtype=torch.float32,
                )
            target_bonus_token_ids = sampler_output.sampled_token_ids[
                row_ids, num_draft_tokens[row_ids]
            ]
            bonus_token_ids = sample_spec_cascade_opt_bonus_token_ids(
                self.last_bonus_target_probs[row_ids],
                q_bonus_probs[row_ids],
                target_bonus_token_ids,
                q_bonus_token_ids[row_ids],
                self._require_spec_cascade_alpha(),
                target_gate_probs=target_gate_probs,
                draft_gate_probs=draft_gate_probs,
            )
        elif self.relaxed_target_method == "spec_cascade_tok3":
            if self.last_bonus_target_probs is None:
                raise RuntimeError("Missing target bonus probabilities.")
            if self.last_bonus_tok3_top_mask is None:
                raise RuntimeError("Missing Tok3 bonus top mask.")
            p_bonus = self.last_bonus_target_probs[row_ids]
            q_bonus = q_bonus_probs[row_ids]
            bonus_top_mask = self.last_bonus_tok3_top_mask[row_ids]
            relaxed_bonus_probs = apply_spec_cascade_tok3_constraint(
                p_bonus,
                q_bonus,
                bonus_top_mask,
            )
            bonus_token_ids = sample_probs(
                relaxed_bonus_probs,
                sampling_metadata.generators,
                row_ids,
            )
        elif self.relaxed_target_method == "lossy_spec_decode_beta1":
            if self.last_bonus_target_probs is None:
                raise RuntimeError("Missing target bonus probabilities.")
            bonus_token_ids = sample_lossy_spec_decode_beta1_bonus_token_ids(
                self.last_bonus_target_probs[row_ids],
                q_bonus_probs[row_ids],
                q_bonus_token_ids[row_ids],
                self._require_lossy_alpha(),
                sampling_metadata.generators,
                row_ids,
            )
        elif self.relaxed_target_method in ("scd_expert_toppk_gated", "scd_alpha"):
            if self.last_bonus_scd_rule_logits is None:
                raise RuntimeError("Missing SCD bonus expert rule logits.")
            if q_bonus_rule_logits is None:
                raise RuntimeError("Missing SCD bonus amateur rule logits.")
            expert_rule_logits = self.last_bonus_scd_rule_logits[row_ids]
            amateur_rule_logits = q_bonus_rule_logits[row_ids]
            if self.relaxed_target_method == "scd_expert_toppk_gated":
                if self.last_bonus_scd_processed_logits is None:
                    raise RuntimeError("Missing SCD bonus processed expert logits.")
                gate = torch.isfinite(self.last_bonus_scd_processed_logits[row_ids])
            else:
                gate = scd_alpha_gate_from_logits(
                    expert_rule_logits,
                    sampling_metadata.temperature[row_ids],
                    self._require_scd_alpha(),
                )
            scd_temperature = self.scd_temperature
            if scd_temperature is None:
                if sampling_metadata.temperature is None:
                    raise RuntimeError("SCD bonus requires sampling temperature.")
                scd_temperature = sampling_metadata.temperature[row_ids]
                scd_temperature = torch.where(
                    scd_temperature == GREEDY_TEMPERATURE,
                    torch.ones_like(scd_temperature),
                    scd_temperature,
                )
            _log_scd_debug_tensors(
                "bonus_pre_scd",
                {
                    "bonus_expert_rule_logits": expert_rule_logits,
                    "bonus_amateur_rule_logits": amateur_rule_logits,
                    "bonus_processed_expert_logits": (
                        self.last_bonus_scd_processed_logits[row_ids]
                        if self.last_bonus_scd_processed_logits is not None
                        else None
                    ),
                    "q_bonus_probs": q_bonus_probs[row_ids],
                },
                gate=gate,
                draft_token_ids=q_bonus_token_ids[row_ids],
                extra={
                    "relaxed_target_method": self.relaxed_target_method,
                    "scd_beta": self.scd_beta,
                    "scd_temperature": (
                        _scd_debug_tensor_to_list(scd_temperature)
                        if isinstance(scd_temperature, torch.Tensor)
                        else scd_temperature
                    ),
                    "row_ids": _scd_debug_tensor_to_list(row_ids),
                    "q_bonus_token_ids": _scd_debug_tensor_to_list(
                        q_bonus_token_ids[row_ids]
                    ),
                },
            )
            relaxed_bonus_probs = apply_scd_constraint(
                expert_rule_logits,
                amateur_rule_logits,
                gate,
                self._require_scd_beta(),
                scd_temperature,
            )
            bonus_token_ids = sample_probs(
                relaxed_bonus_probs,
                sampling_metadata.generators,
                row_ids,
            )
        else:
            raise ValueError(
                f"relaxed_T_qp bonus is not implemented for "
                f"relaxed_target_method={self.relaxed_target_method!r}."
            )
        sampler_output.sampled_token_ids[
            row_ids, num_draft_tokens[row_ids]
        ] = bonus_token_ids.to(torch.int32)

    def _require_cactus_delta(self) -> float:
        if self.cactus_delta is None:
            raise RuntimeError("Cactus relaxed bonus requires cactus_delta.")
        return self.cactus_delta

    def _require_verifier_weight(self) -> float:
        if self.verifier_weight is None:
            raise RuntimeError("Static ensemble relaxed bonus requires verifier_weight.")
        return self.verifier_weight

    def _require_fuzzy_divergence(self) -> str:
        if self.fuzzy_divergence is None:
            raise RuntimeError("rFSD relaxed bonus requires fuzzy_divergence.")
        return self.fuzzy_divergence

    def _require_fuzzy_threshold(self) -> float:
        if self.fuzzy_threshold is None:
            raise RuntimeError("rFSD relaxed bonus requires fuzzy_threshold.")
        return self.fuzzy_threshold

    def _require_spec_cascade_alpha(self) -> float:
        if self.spec_cascade_alpha is None:
            raise RuntimeError("SpecCascades relaxed bonus requires alpha.")
        return self.spec_cascade_alpha

    def _require_lossy_alpha(self) -> float:
        if self.lossy_alpha is None:
            raise RuntimeError("Lossy SD relaxed bonus requires lossy_alpha.")
        return self.lossy_alpha

    def _require_scd_beta(self) -> float:
        if self.scd_beta is None:
            raise RuntimeError("SCD relaxed target requires scd_beta.")
        return self.scd_beta

    def _require_scd_alpha(self) -> float:
        if self.scd_alpha is None:
            raise RuntimeError("SCD alpha relaxed target requires scd_alpha.")
        return self.scd_alpha

    def _get_logprobs_tensors(
        self,
        max_num_logprobs: int,
        metadata: SpecDecodeMetadata,
        logits: torch.Tensor,
        target_logits: torch.Tensor,
        bonus_logits: torch.Tensor,
        sampled_token_ids: torch.Tensor,
    ) -> LogprobsTensors:
        cu_num_sampled_tokens = torch.zeros_like(metadata.cu_num_sampled_tokens)
        cu_num_sampled_tokens[1:] = metadata.cu_num_sampled_tokens[:-1]

        # Collect target and bonus logits.
        bonus_logits_indices = metadata.bonus_logits_indices
        target_logits_indices = metadata.target_logits_indices
        final_logits = torch.zeros_like(logits, dtype=torch.float32)
        final_logits[target_logits_indices] = target_logits.to(torch.float32)
        final_logits[bonus_logits_indices] = bonus_logits.to(torch.float32)

        # NOTE: To avoid cpu-gpu synchronization, we now simply compute indices for
        # all draft tokens, including the rejected ones. The rejected tokens will
        # be filtered out in the `parse_output`.
        logit_start_indices = cu_num_sampled_tokens
        offsets = torch.arange(
            sampled_token_ids.shape[-1],
            device=logit_start_indices.device,
            dtype=logit_start_indices.dtype,
        )
        accepted_logit_indices = (
            logit_start_indices.unsqueeze(1) + offsets.unsqueeze(0)
        ).flatten()
        accepted_logit_indices.clamp_(max=final_logits.shape[0] - 1)
        accepted_tokens = sampled_token_ids.clone().flatten()
        # we replace rejected token ids with 0 to avoid gather_logprobs error
        accepted_tokens[accepted_tokens == PLACEHOLDER_TOKEN_ID] = 0

        # Compute logprobs for accepted tokens.
        accepted_logits = final_logits[accepted_logit_indices]
        accepted_logprobs = (
            accepted_logits
            if self.is_logits_logprobs_mode
            else self.sampler.compute_logprobs(accepted_logits)
        )
        return self.sampler.gather_logprobs(
            accepted_logprobs,
            max_num_logprobs,
            accepted_tokens.to(torch.int64),
        )

    @staticmethod
    def parse_output(
        output_token_ids: torch.Tensor,
        vocab_size: int,
        discard_req_indices: Sequence[int] = (),
        logprobs_tensors: LogprobsTensors | None = None,
    ) -> tuple[list[list[int]], LogprobsLists | None]:
        """Parse the output of the rejection sampler.
        Args:
            output_token_ids: The sampled token IDs in shape
                [batch_size, max_spec_len + 1]. The rejected tokens are
                replaced with `PLACEHOLDER_TOKEN_ID` by the rejection sampler
                and will be filtered out in this function.
            vocab_size: The size of the vocabulary.
            discard_req_indices: Optional row indices to discard tokens in.
            logprobs_tensors: Optional logprobs tensors to filter.
        Returns:
            A list of lists of token IDs.
        """
        output_token_ids_np = output_token_ids.cpu().numpy()
        # Create mask for valid tokens.
        valid_mask = (output_token_ids_np != PLACEHOLDER_TOKEN_ID) & (
            output_token_ids_np < vocab_size
        )
        output_logprobs = None
        if logprobs_tensors is not None:
            cu_num_tokens = [0] + valid_mask.sum(axis=1).cumsum().tolist()
            filtered_tensors = logprobs_tensors.filter(valid_mask.flatten())
            output_logprobs = filtered_tensors.tolists(cu_num_tokens)

        if len(discard_req_indices) > 0:
            valid_mask[discard_req_indices] = False
        outputs = [
            row[valid_mask[i]].tolist() for i, row in enumerate(output_token_ids_np)
        ]
        return outputs, output_logprobs

    def apply_logits_processors(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        metadata: SpecDecodeMetadata,
    ) -> torch.Tensor:
        has_penalties = not sampling_metadata.no_penalties
        any_penalties_or_bad_words = (
            sampling_metadata.bad_words_token_ids or has_penalties
        )

        output_token_ids = sampling_metadata.output_token_ids
        if any_penalties_or_bad_words:
            output_token_ids = self._combine_outputs_with_spec_tokens(
                output_token_ids,
                sampling_metadata.spec_token_ids,
            )

        # Calculate indices of target logits.
        if sampling_metadata.allowed_token_ids_mask is not None or has_penalties:
            num_requests = len(metadata.num_draft_tokens)
            num_draft_tokens = torch.tensor(metadata.num_draft_tokens, device="cpu")
            original_indices = torch.arange(num_requests, device="cpu")
            repeat_indices_cpu = original_indices.repeat_interleave(num_draft_tokens)
            repeat_indices = repeat_indices_cpu.to(
                device=logits.device, non_blocking=True
            )
            logits = self.apply_penalties(
                logits, sampling_metadata, metadata, repeat_indices, output_token_ids
            )

            # Apply allowed token ids.
            if sampling_metadata.allowed_token_ids_mask is not None:
                token_mask = sampling_metadata.allowed_token_ids_mask[repeat_indices]
                logits.masked_fill_(token_mask, float("-inf"))

        # Apply bad words exclusion.
        if bad_words_token_ids := sampling_metadata.bad_words_token_ids:
            apply_bad_words_with_drafts(
                logits, bad_words_token_ids, output_token_ids, metadata.num_draft_tokens
            )

        for processor in sampling_metadata.logitsprocs.non_argmax_invariant:
            if isinstance(processor, MinTokensLogitsProcessor):
                logits = processor.apply_with_spec_decode(
                    logits, metadata.num_draft_tokens
                )

        return logits

    @staticmethod
    def apply_penalties(
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        metadata: SpecDecodeMetadata,
        repeat_indices: torch.Tensor,
        output_token_ids: list[list[int]],
    ) -> torch.Tensor:
        if sampling_metadata.no_penalties:
            return logits

        assert sampling_metadata.prompt_token_ids is not None

        prompt_token_ids = sampling_metadata.prompt_token_ids[repeat_indices]
        presence_penalties = sampling_metadata.presence_penalties[repeat_indices]
        frequency_penalties = sampling_metadata.frequency_penalties[repeat_indices]
        repetition_penalties = sampling_metadata.repetition_penalties[repeat_indices]

        logits = apply_all_penalties(
            logits,
            prompt_token_ids,
            presence_penalties,
            frequency_penalties,
            repetition_penalties,
            output_token_ids,
        )
        return logits

    @staticmethod
    def _combine_outputs_with_spec_tokens(
        output_token_ids: list[list[int]],
        spec_token_ids: list[list[int]] | None = None,
    ) -> list[list[int]]:
        if spec_token_ids is None:
            return output_token_ids

        result = []
        for out, spec in zip(output_token_ids, spec_token_ids):
            if len(spec) == 0:
                continue
            result.append(out)
            for i in range(len(spec) - 1):
                result.append([*result[-1], spec[i]])
        return result


def bonus_target_probs(
    bonus_logits: torch.Tensor,
    sampling_metadata: SamplingMetadata,
    sampler: Sampler,
) -> torch.Tensor:
    """Return processed target bonus probabilities p_bonus [B, V]."""
    return bonus_target_probs_and_logits(
        bonus_logits,
        sampling_metadata,
        sampler,
    )[0]


def bonus_target_probs_and_logits(
    bonus_logits: torch.Tensor,
    sampling_metadata: SamplingMetadata,
    sampler: Sampler,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return target bonus probs plus paper/processed logits.

    The paper logits are after logits processors/legal masks and before
    temperature/top-k/top-p. Processed logits are the actual target sampling
    logits used to form the returned probabilities.
    """
    logits = bonus_logits.to(torch.float32)
    logits = sampler.apply_logits_processors(
        logits, sampling_metadata, predict_bonus_token=True
    )
    paper_logits = logits.clone()
    if sampling_metadata.all_greedy:
        token_ids = logits.argmax(dim=-1)
        probs = torch.zeros_like(logits, dtype=torch.float32)
        probs.scatter_(-1, token_ids.unsqueeze(-1), 1.0)
        return probs.contiguous(), paper_logits.contiguous(), logits.contiguous()

    assert sampling_metadata.temperature is not None
    temperature = sampling_metadata.temperature
    if not sampling_metadata.all_random:
        is_greedy = temperature == GREEDY_TEMPERATURE
        temperature = torch.where(is_greedy, 1.0, temperature)
    else:
        is_greedy = None
    logits.div_(temperature.unsqueeze(-1))
    for processor in sampling_metadata.logitsprocs.argmax_invariant:
        logits = processor.apply(logits)
    logits = apply_top_k_top_p(
        logits,
        sampling_metadata.top_k,
        sampling_metadata.top_p,
    )
    probs = logits.softmax(dim=-1, dtype=torch.float32)
    if is_greedy is not None:
        greedy_token_ids = logits.argmax(dim=-1)
        greedy_probs = torch.zeros_like(probs)
        greedy_probs.scatter_(-1, greedy_token_ids.unsqueeze(-1), 1.0)
        probs = torch.where(is_greedy.unsqueeze(-1), greedy_probs, probs)
    return probs.contiguous(), paper_logits.contiguous(), logits.contiguous()


def sample_probs(
    probs: torch.Tensor,
    generators: dict[int, torch.Generator],
    row_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """Sample from probability rows with vLLM's exponential race trick."""
    q = torch.empty_like(probs)
    q.exponential_()
    if row_ids is None:
        for i, generator in generators.items():
            if i < probs.shape[0]:
                q[i].exponential_(generator=generator)
    else:
        for local_i, row_id in enumerate(row_ids.tolist()):
            generator = generators.get(int(row_id))
            if generator is not None:
                q[local_i].exponential_(generator=generator)
    return probs.div(q).argmax(dim=-1).view(-1)


def scd_alpha_gate_from_logits(
    expert_rule_logits: torch.Tensor,
    expert_temperature: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    """Return SCD's improved-CD plausibility gate from expert logits.

    Shapes: expert_rule_logits [N, V], expert_temperature [N]. A token is
    plausible when z_E[i] >= max(z_E) + expert_temperature * log(alpha).
    """
    if not 0.0 < alpha <= 1.0:
        raise ValueError("scd_alpha must satisfy 0 < alpha <= 1.")
    if expert_rule_logits.ndim != 2:
        raise ValueError("expert_rule_logits must have shape [N, V].")
    if torch.isnan(expert_rule_logits).any():
        raise ValueError("expert_rule_logits must not contain NaNs.")

    expert_temperature = expert_temperature.to(
        device=expert_rule_logits.device,
        dtype=expert_rule_logits.dtype,
    ).view(-1)
    if expert_temperature.shape[0] != expert_rule_logits.shape[0]:
        raise ValueError("expert_temperature must have one value per SCD row.")
    if torch.isnan(expert_temperature).any() or torch.any(expert_temperature <= 0):
        raise ValueError("SCD alpha gate requires positive finite temperatures.")
    if not torch.isfinite(expert_temperature).all():
        raise ValueError("SCD alpha gate requires finite temperatures.")

    finite = torch.isfinite(expert_rule_logits)
    if not finite.any(dim=-1).all():
        raise ValueError("SCD alpha gate requires at least one finite logit per row.")

    row_max = expert_rule_logits.masked_fill(~finite, float("-inf")).max(
        dim=-1, keepdim=True
    ).values
    threshold = row_max + expert_temperature.unsqueeze(-1) * math.log(alpha)
    gate = finite & (expert_rule_logits >= threshold)
    if not gate.any(dim=-1).all():
        raise ValueError("SCD alpha gate produced an empty support row.")
    return gate.contiguous()


def apply_scd_constraint(
    expert_rule_logits: torch.Tensor,
    amateur_rule_logits: torch.Tensor,
    gate: torch.Tensor,
    beta: float,
    scd_temperature: float | torch.Tensor,
) -> torch.Tensor:
    """Return SCD target distribution pi_scd.

    Shapes: expert_rule_logits/amateur_rule_logits/gate are [N, V]. The gate
    masks implausible tokens to zero probability before contrastive softmax
    normalization with scd_temperature.
    """
    if not math.isfinite(beta) or beta < 0:
        raise ValueError("scd_beta must be >= 0.")
    if expert_rule_logits.ndim != 2 or amateur_rule_logits.ndim != 2:
        raise ValueError("SCD logits must have shape [N, V].")
    if gate.ndim != 2:
        raise ValueError("SCD gate must have shape [N, V].")
    if expert_rule_logits.shape != amateur_rule_logits.shape:
        raise ValueError("SCD expert and amateur logits must have the same shape.")
    if expert_rule_logits.shape != gate.shape:
        raise ValueError("SCD gate shape must match the logits.")
    if gate.dtype != torch.bool:
        raise ValueError("SCD gate must be a boolean tensor.")
    if not gate.any(dim=-1).all():
        raise ValueError("SCD gate produced an empty support row.")
    if torch.isnan(expert_rule_logits).any() or torch.isnan(amateur_rule_logits).any():
        raise ValueError("SCD logits must not contain NaNs.")
    if not torch.isfinite(expert_rule_logits[gate]).all():
        raise ValueError("SCD expert logits must be finite on the gate support.")
    if not torch.isfinite(amateur_rule_logits[gate]).all():
        raise ValueError("SCD amateur logits must be finite on the gate support.")

    if isinstance(scd_temperature, torch.Tensor):
        scd_temperature = scd_temperature.to(
            device=expert_rule_logits.device,
            dtype=expert_rule_logits.dtype,
        ).view(-1)
        if scd_temperature.shape[0] not in (1, expert_rule_logits.shape[0]):
            raise ValueError("scd_temperature must be scalar or one value per row.")
        if torch.isnan(scd_temperature).any() or torch.any(scd_temperature <= 0):
            raise ValueError("scd_temperature must be > 0.")
        if not torch.isfinite(scd_temperature).all():
            raise ValueError("scd_temperature must be finite.")
        score_divisor = scd_temperature.view(-1, 1)
    else:
        if not math.isfinite(scd_temperature) or scd_temperature <= 0:
            raise ValueError("scd_temperature must be > 0.")
        score_divisor = scd_temperature

    scores = ((1.0 + beta) * expert_rule_logits - beta * amateur_rule_logits)
    scores = scores / score_divisor
    scores = scores.masked_fill(~gate, float("-inf"))
    if not torch.isfinite(scores[gate]).all():
        raise ValueError("SCD scores must be finite on the gate support.")

    row_max = scores.max(dim=-1, keepdim=True).values
    shifted = torch.where(gate, scores - row_max, float("-inf"))
    unnormalized = shifted.exp()
    row_sums = unnormalized.sum(dim=-1, keepdim=True)
    if not torch.isfinite(row_sums).all() or torch.any(row_sums <= 0):
        raise ValueError("SCD rows must have positive finite mass before softmax.")

    probs = unnormalized / row_sums
    if torch.isnan(probs).any():
        raise ValueError("SCD probabilities must not contain NaNs.")
    prob_sums = probs.sum(dim=-1)
    if not torch.isfinite(prob_sums).all() or torch.any(prob_sums <= 0):
        raise ValueError("SCD rows must have positive finite probability mass.")
    return probs.contiguous()


def apply_scd_constraint_inplace(
    expert_rule_logits: torch.Tensor,
    amateur_rule_logits: torch.Tensor,
    gate: torch.Tensor,
    beta: float,
    scd_temperature: float | torch.Tensor,
) -> torch.Tensor:
    """Return SCD pi in expert_rule_logits storage.

    Inputs are [N, V]. This mutates expert_rule_logits, which is already a
    target-side rule-logit clone in the verification path.
    """
    if not math.isfinite(beta) or beta < 0:
        raise ValueError("scd_beta must be >= 0.")
    if expert_rule_logits.ndim != 2 or amateur_rule_logits.ndim != 2:
        raise ValueError("SCD logits must have shape [N, V].")
    if gate.ndim != 2:
        raise ValueError("SCD gate must have shape [N, V].")
    if expert_rule_logits.shape != amateur_rule_logits.shape:
        raise ValueError("SCD expert and amateur logits must have the same shape.")
    if expert_rule_logits.shape != gate.shape:
        raise ValueError("SCD gate shape must match the logits.")
    if gate.dtype != torch.bool:
        raise ValueError("SCD gate must be a boolean tensor.")
    if not gate.any(dim=-1).all():
        raise ValueError("SCD gate produced an empty support row.")
    if torch.isnan(expert_rule_logits).any() or torch.isnan(amateur_rule_logits).any():
        raise ValueError("SCD logits must not contain NaNs.")
    if not torch.isfinite(expert_rule_logits[gate]).all():
        raise ValueError("SCD expert logits must be finite on the gate support.")
    if not torch.isfinite(amateur_rule_logits[gate]).all():
        raise ValueError("SCD amateur logits must be finite on the gate support.")

    if isinstance(scd_temperature, torch.Tensor):
        scd_temperature = scd_temperature.to(
            device=expert_rule_logits.device,
            dtype=expert_rule_logits.dtype,
        ).view(-1)
        if scd_temperature.shape[0] not in (1, expert_rule_logits.shape[0]):
            raise ValueError("scd_temperature must be scalar or one value per row.")
        if torch.isnan(scd_temperature).any() or torch.any(scd_temperature <= 0):
            raise ValueError("scd_temperature must be > 0.")
        if not torch.isfinite(scd_temperature).all():
            raise ValueError("scd_temperature must be finite.")
        score_divisor = scd_temperature.view(-1, 1)
    else:
        if not math.isfinite(scd_temperature) or scd_temperature <= 0:
            raise ValueError("scd_temperature must be > 0.")
        score_divisor = scd_temperature

    scores = expert_rule_logits
    scores.mul_(1.0 + beta)
    scores.add_(amateur_rule_logits, alpha=-beta)
    scores.div_(score_divisor)
    scores.masked_fill_(~gate, float("-inf"))
    if not torch.isfinite(scores[gate]).all():
        raise ValueError("SCD scores must be finite on the gate support.")

    row_max = scores.max(dim=-1, keepdim=True).values
    scores.sub_(row_max)
    scores.masked_fill_(~gate, float("-inf"))
    scores.exp_()
    scores.masked_fill_(~gate, 0.0)
    row_sums = scores.sum(dim=-1, keepdim=True)
    if not torch.isfinite(row_sums).all() or torch.any(row_sums <= 0):
        raise ValueError("SCD rows must have positive finite mass before softmax.")

    scores.div_(row_sums)
    if torch.isnan(scores).any():
        raise ValueError("SCD probabilities must not contain NaNs.")
    prob_sums = scores.sum(dim=-1)
    if not torch.isfinite(prob_sums).all() or torch.any(prob_sums <= 0):
        raise ValueError("SCD rows must have positive finite probability mass.")
    return scores.contiguous()


def static_ensemble_accept_ratio(
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor,
    draft_token_ids: torch.Tensor,
    verifier_weight: float,
) -> torch.Tensor:
    """Return pi_static(q, p)(x) / q(x) for sampled draft tokens x."""
    assert target_probs.ndim == 2
    assert draft_probs.ndim == 2
    assert draft_token_ids.ndim == 1
    assert target_probs.shape == draft_probs.shape
    assert target_probs.shape[0] == draft_token_ids.shape[0]

    if verifier_weight == 0.0:
        return torch.ones_like(draft_token_ids, dtype=target_probs.dtype)

    selected = draft_token_ids.to(torch.int64).unsqueeze(-1)
    selected_target_probs = target_probs.gather(-1, selected).squeeze(-1)
    selected_draft_probs = draft_probs.gather(-1, selected).squeeze(-1)

    ratio = torch.zeros_like(selected_target_probs)
    positive = selected_draft_probs > 0
    ratio[positive] = (
        (1.0 - verifier_weight)
        + verifier_weight
        * selected_target_probs[positive]
        / selected_draft_probs[positive]
    )
    return ratio


def lossy_spec_decode_beta1_accept_ratio(
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor,
    draft_token_ids: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    """Return beta=1 lossy-SD accept ratios for sampled draft tokens.

    Shapes: target_probs/draft_probs are [N, V], draft_token_ids is [N].
    Inputs are processed p/q probability rows, after temperature and sampling
    constraints. The beta=1 recovery row is the vanilla (p - q)+ residual.
    """
    assert target_probs.ndim == 2
    assert draft_probs.ndim == 2
    assert draft_token_ids.ndim == 1
    assert target_probs.shape == draft_probs.shape
    assert target_probs.shape[0] == draft_token_ids.shape[0]
    if not 0.0 <= alpha < 1.0:
        raise ValueError("lossy_alpha must satisfy 0 <= alpha < 1.")

    selected = draft_token_ids.to(torch.int64).unsqueeze(-1)
    selected_target_probs = target_probs.gather(-1, selected).squeeze(-1)
    selected_draft_probs = draft_probs.gather(-1, selected).squeeze(-1)

    ratio = torch.zeros_like(selected_target_probs)
    positive = selected_draft_probs > 0
    ratio[positive] = (
        selected_target_probs[positive]
        / ((1.0 - alpha) * selected_draft_probs[positive])
    )
    return ratio


def lossy_spec_decode_beta1_residual_probs(
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor,
) -> torch.Tensor:
    """Return normalized beta=1 lossy-SD residual rows, (p - q)+."""
    assert target_probs.ndim == 2
    assert draft_probs.ndim == 2
    assert target_probs.shape == draft_probs.shape

    residual = (target_probs - draft_probs).clamp_min(0.0)
    row_sums = residual.sum(dim=-1, keepdim=True)
    return torch.where(
        row_sums > 0.0,
        residual / row_sums.clamp_min(torch.finfo(residual.dtype).tiny),
        residual,
    )


def sample_lossy_spec_decode_beta1_bonus_token_ids(
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor,
    q_bonus_token_ids: torch.Tensor,
    alpha: float,
    generators: dict[int, torch.Generator],
    row_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run one lossy beta1 accept/residual step for relaxed bonus tokens."""
    q_bonus_token_ids = q_bonus_token_ids.view(-1).to(torch.int64)
    accept_ratio = lossy_spec_decode_beta1_accept_ratio(
        target_probs,
        draft_probs,
        q_bonus_token_ids,
        alpha,
    ).to(torch.float64)

    uniforms = torch.empty(
        q_bonus_token_ids.shape,
        dtype=torch.float64,
        device=target_probs.device,
    )
    uniforms.uniform_()
    if row_ids is None:
        for i, generator in generators.items():
            if i < uniforms.shape[0]:
                uniforms[i].uniform_(generator=generator)
    else:
        for local_i, row_id in enumerate(row_ids.tolist()):
            generator = generators.get(int(row_id))
            if generator is not None:
                uniforms[local_i].uniform_(generator=generator)

    residual_probs = lossy_spec_decode_beta1_residual_probs(
        target_probs,
        draft_probs,
    )
    recovered_token_ids = sample_probs(residual_probs, generators, row_ids)
    return torch.where(
        accept_ratio >= uniforms,
        q_bonus_token_ids.to(recovered_token_ids.dtype),
        recovered_token_ids,
    )


def cactus_accept_ratio(
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor,
    draft_token_ids: torch.Tensor,
    delta: float,
) -> torch.Tensor:
    """Return Cactus proposal-conditioned H_x[x] / q(x)."""
    assert target_probs.ndim == 2
    assert draft_probs.ndim == 2
    assert draft_token_ids.ndim == 1
    assert target_probs.shape == draft_probs.shape
    assert target_probs.shape[0] == draft_token_ids.shape[0]

    selected = draft_token_ids.to(torch.int64).unsqueeze(-1)
    selected_target_probs = target_probs.gather(-1, selected).squeeze(-1)
    selected_draft_probs = draft_probs.gather(-1, selected).squeeze(-1)
    boost = torch.sqrt(
        2 * delta * selected_target_probs * (1 - selected_target_probs)
    )
    relaxed_selected = (selected_target_probs + boost).clamp(min=0.0, max=1.0)

    ratio = torch.zeros_like(selected_target_probs)
    positive = selected_draft_probs > 0
    ratio[positive] = relaxed_selected[positive] / selected_draft_probs[positive]
    return ratio


def cactus_residual_probs(
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor,
    draft_token_ids: torch.Tensor,
    delta: float,
) -> torch.Tensor:
    """Return normalized Cactus rejection residual rows, (H_x - q)+."""
    relaxed = apply_cactus_constraint(
        target_probs,
        draft_probs,
        draft_token_ids.to(torch.int64),
        delta,
    )
    residual = (relaxed - draft_probs).clamp_min(0.0)
    row_sums = residual.sum(dim=-1, keepdim=True)
    return torch.where(
        row_sums > 0.0,
        residual / row_sums.clamp_min(torch.finfo(residual.dtype).tiny),
        residual,
    )


def sample_cactus_bonus_token_ids(
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor,
    q_bonus_token_ids: torch.Tensor,
    delta: float,
    generators: dict[int, torch.Generator],
    row_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run one Cactus-style accept/residual step for relaxed bonus tokens."""
    q_bonus_token_ids = q_bonus_token_ids.view(-1).to(torch.int64)
    assert target_probs.ndim == 2
    assert draft_probs.ndim == 2
    assert target_probs.shape == draft_probs.shape
    assert target_probs.shape[0] == q_bonus_token_ids.shape[0]

    accept_ratio = cactus_accept_ratio(
        target_probs,
        draft_probs,
        q_bonus_token_ids,
        delta,
    ).to(torch.float64)
    uniforms = torch.empty(
        q_bonus_token_ids.shape,
        dtype=torch.float64,
        device=target_probs.device,
    )
    uniforms.uniform_()
    if row_ids is None:
        for i, generator in generators.items():
            if i < uniforms.shape[0]:
                uniforms[i].uniform_(generator=generator)
    else:
        for local_i, row_id in enumerate(row_ids.tolist()):
            generator = generators.get(int(row_id))
            if generator is not None:
                uniforms[local_i].uniform_(generator=generator)

    residual_probs = cactus_residual_probs(
        target_probs,
        draft_probs,
        q_bonus_token_ids,
        delta,
    )
    recovered_token_ids = sample_probs(residual_probs, generators, row_ids)
    return torch.where(
        accept_ratio >= uniforms,
        q_bonus_token_ids.to(recovered_token_ids.dtype),
        recovered_token_ids,
    )


def static_ensemble_residual_probs(
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor,
) -> torch.Tensor:
    """Return normalized static-ensemble rejection residual rows.

    For F = w_p * p + (1 - w_p) * q and w_p > 0, the rejection residual is
    proportional to (p - q)+, so the normalized residual is independent of w_p.
    """
    assert target_probs.ndim == 2
    assert draft_probs.ndim == 2
    assert target_probs.shape == draft_probs.shape

    residual = (target_probs - draft_probs).clamp_min(0.0)
    row_sums = residual.sum(dim=-1, keepdim=True)
    return torch.where(
        row_sums > 0,
        residual / row_sums.clamp_min(torch.finfo(residual.dtype).tiny),
        residual,
    )


def fsd_divergence_scores(
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor,
    divergence: str,
) -> torch.Tensor:
    """Return D(p, q) for FSD-family rows.

    This repo-local FSD family uses p=target/verifier and q=draft. The KL
    option is KL(p || q); this direction is important for zero q support.
    """
    assert target_probs.ndim == 2
    assert draft_probs.ndim == 2
    assert target_probs.shape == draft_probs.shape

    p = target_probs.to(torch.float32)
    q = draft_probs.to(torch.float32)
    tiny = torch.finfo(torch.float32).tiny
    safe_p = p.clamp_min(tiny)
    safe_q = q.clamp_min(tiny)

    if divergence == "kl":
        return (p * (safe_p.log() - safe_q.log())).sum(dim=-1)
    if divergence == "js":
        m = 0.5 * (p + q)
        safe_m = m.clamp_min(tiny)
        kl_p_m = (p * (safe_p.log() - safe_m.log())).sum(dim=-1)
        kl_q_m = (q * (safe_q.log() - safe_m.log())).sum(dim=-1)
        return 0.5 * (kl_p_m + kl_q_m)
    raise ValueError(f"Unknown FSD-family divergence: {divergence!r}.")


def fsd_divergence_scores_inplace(
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor,
    divergence: str,
) -> torch.Tensor:
    """Return D(p, q) with lower transient [N, V] allocation pressure.

    Inputs are [N, V] probability rows. The inputs are not mutated; the
    elementwise logs and products happen in mutable scratch tensors.
    """
    assert target_probs.ndim == 2
    assert draft_probs.ndim == 2
    assert target_probs.shape == draft_probs.shape

    p = target_probs.to(torch.float32)
    q = draft_probs.to(torch.float32)
    tiny = torch.finfo(torch.float32).tiny

    if divergence == "kl":
        log_ratio = p.clamp_min(tiny)
        log_ratio.log_()
        safe_q = q.clamp_min(tiny)
        safe_q.log_()
        log_ratio.sub_(safe_q)
        log_ratio.mul_(p)
        return log_ratio.sum(dim=-1)

    if divergence == "js":
        p_log_ratio = p.clamp_min(tiny)
        p_log_ratio.log_()
        q_log_ratio = q.clamp_min(tiny)
        q_log_ratio.log_()
        mixture_log = p.add(q).mul_(0.5)
        mixture_log.clamp_min_(tiny)
        mixture_log.log_()
        p_log_ratio.sub_(mixture_log)
        q_log_ratio.sub_(mixture_log)
        p_log_ratio.mul_(p)
        q_log_ratio.mul_(q)
        p_log_ratio.add_(q_log_ratio)
        p_log_ratio.mul_(0.5)
        return p_log_ratio.sum(dim=-1)

    raise ValueError(f"Unknown FSD-family divergence: {divergence!r}.")


def rfsd_use_draft_mask(
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor,
    divergence: str,
    threshold: float,
) -> torch.Tensor:
    """Return rows where pi_rfsd(q, p) selects q instead of p."""
    if threshold < 0:
        raise ValueError("divergence_threshold must be >= 0.")
    return fsd_divergence_scores(target_probs, draft_probs, divergence) <= threshold


def fsd_accept_mask(
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor,
    divergence: str,
    threshold: float,
) -> torch.Tensor:
    """Return exact-FSD accepted rows using the paper's strict D(p, q) < T."""
    if threshold < 0:
        raise ValueError("fsd_threshold must be >= 0.")
    return fsd_divergence_scores(target_probs, draft_probs, divergence) < threshold


def rfsd_use_draft_mask_inplace(
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor,
    divergence: str,
    threshold: float,
) -> torch.Tensor:
    """Lower-peak version of rfsd_use_draft_mask."""
    if threshold < 0:
        raise ValueError("divergence_threshold must be >= 0.")
    scores = fsd_divergence_scores_inplace(target_probs, draft_probs, divergence)
    return scores <= threshold


def fsd_accept_mask_inplace(
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor,
    divergence: str,
    threshold: float,
) -> torch.Tensor:
    """Lower-peak version of fsd_accept_mask."""
    if threshold < 0:
        raise ValueError("fsd_threshold must be >= 0.")
    scores = fsd_divergence_scores_inplace(target_probs, draft_probs, divergence)
    return scores < threshold


def apply_rfsd_constraint(
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor,
    divergence: str,
    threshold: float,
) -> torch.Tensor:
    """Return pi_rfsd(q, p) = q if D(p, q) <= threshold else p.

    No verifier-confidence / max-p gate is applied in this repo-local variant.
    """
    use_q = rfsd_use_draft_mask(
        target_probs,
        draft_probs,
        divergence,
        threshold,
    )
    return torch.where(use_q.unsqueeze(-1), draft_probs, target_probs).contiguous()


def apply_rfsd_constraint_inplace(
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor,
    divergence: str,
    threshold: float,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return pi_rfsd while reusing `out`/target storage when possible."""
    use_q = rfsd_use_draft_mask_inplace(
        target_probs,
        draft_probs,
        divergence,
        threshold,
    )
    relaxed = target_probs if out is None else out
    if relaxed is not target_probs:
        relaxed.copy_(target_probs)
    if use_q.any():
        relaxed[use_q] = draft_probs[use_q]
    return relaxed.contiguous()


def sample_rfsd_bonus_token_ids(
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor,
    target_bonus_token_ids: torch.Tensor,
    q_bonus_token_ids: torch.Tensor,
    divergence: str,
    threshold: float,
) -> torch.Tensor:
    """Choose the already sampled q or p bonus token using the rFSD gate."""
    target_bonus_token_ids = target_bonus_token_ids.view(-1)
    q_bonus_token_ids = q_bonus_token_ids.view(-1).to(target_bonus_token_ids.dtype)
    assert target_bonus_token_ids.shape == q_bonus_token_ids.shape

    use_q = rfsd_use_draft_mask(target_probs, draft_probs, divergence, threshold)
    return torch.where(use_q, q_bonus_token_ids, target_bonus_token_ids)


def fuzzy_divergence_scores(
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor,
    divergence: str,
) -> torch.Tensor:
    """Legacy alias for fsd_divergence_scores."""

    return fsd_divergence_scores(target_probs, draft_probs, divergence)


def fuzzy_use_draft_mask(
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor,
    divergence: str,
    threshold: float,
) -> torch.Tensor:
    """Legacy alias for rfsd_use_draft_mask."""

    return rfsd_use_draft_mask(target_probs, draft_probs, divergence, threshold)


def apply_fuzzy_constraint(
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor,
    divergence: str,
    threshold: float,
) -> torch.Tensor:
    """Legacy alias for apply_rfsd_constraint."""

    return apply_rfsd_constraint(target_probs, draft_probs, divergence, threshold)


def sample_fuzzy_bonus_token_ids(
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor,
    target_bonus_token_ids: torch.Tensor,
    q_bonus_token_ids: torch.Tensor,
    divergence: str,
    threshold: float,
) -> torch.Tensor:
    """Legacy alias for sample_rfsd_bonus_token_ids."""

    return sample_rfsd_bonus_token_ids(
        target_probs,
        draft_probs,
        target_bonus_token_ids,
        q_bonus_token_ids,
        divergence,
        threshold,
    )


def sample_fsd_bonus_token_ids(
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor,
    target_bonus_token_ids: torch.Tensor,
    q_bonus_token_ids: torch.Tensor,
    divergence: str,
    threshold: float,
) -> torch.Tensor:
    """Choose q bonus on strict FSD gate acceptance, otherwise target p."""
    target_bonus_token_ids = target_bonus_token_ids.view(-1)
    q_bonus_token_ids = q_bonus_token_ids.view(-1).to(target_bonus_token_ids.dtype)
    assert target_bonus_token_ids.shape == q_bonus_token_ids.shape

    accept = fsd_accept_mask(target_probs, draft_probs, divergence, threshold)
    return torch.where(accept, q_bonus_token_ids, target_bonus_token_ids)


def spec_cascade_tok3_top_mask_from_logits(
    logits: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    """Return TokenV3 T_alpha from logits without a softmax.

    Shapes: logits [N, V]. A token is in the top set when
    p(v) >= max_u p(u) * (1 - alpha), equivalently
    logits(v) >= max_u logits(u) + log(1 - alpha).
    """
    assert logits.ndim == 2
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("spec_cascade_alpha must satisfy 0 <= alpha <= 1.")

    finite = torch.isfinite(logits)
    if alpha == 0.0:
        finite_logits = logits.masked_fill(~finite, float("-inf"))
        token_ids = finite_logits.argmax(dim=-1)
        top_mask = torch.zeros_like(finite)
        top_mask.scatter_(-1, token_ids.unsqueeze(-1), True)
        return top_mask & finite
    if alpha == 1.0:
        return finite

    threshold = logits.max(dim=-1, keepdim=True).values + math.log1p(-alpha)
    return finite & (logits >= threshold)


def apply_spec_cascade_tok3_constraint(
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor,
    top_mask: torch.Tensor,
) -> torch.Tensor:
    """Return TokenV3 pi_tok3 = q on top tokens plus eta * p everywhere."""
    assert target_probs.ndim == 2
    assert draft_probs.ndim == 2
    assert top_mask.ndim == 2
    assert target_probs.shape == draft_probs.shape == top_mask.shape

    eta = torch.where(top_mask, 0.0, draft_probs).sum(dim=-1, keepdim=True)
    relaxed = torch.where(top_mask, draft_probs, 0.0) + eta * target_probs
    return relaxed.contiguous()


def apply_spec_cascade_tok3_constraint_inplace(
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor,
    top_mask: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return Tok3 pi while reusing `out`/target storage when possible.

    Tok3 still needs dense pi rows. This avoids a separate pi tensor by writing
    eta * p into the target/output rows and then adding q on the top set.
    """
    assert target_probs.ndim == 2
    assert draft_probs.ndim == 2
    assert top_mask.ndim == 2
    assert target_probs.shape == draft_probs.shape == top_mask.shape

    outside_q = draft_probs.clone()
    outside_q.masked_fill_(top_mask, 0.0)
    eta = outside_q.sum(dim=-1, keepdim=True)

    relaxed = target_probs if out is None else out
    if relaxed is not target_probs:
        relaxed.copy_(target_probs)
    relaxed.mul_(eta)
    if top_mask.any():
        relaxed[top_mask] += draft_probs[top_mask]
    return relaxed.contiguous()


def spec_cascade_opt_defer_mask(
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor,
    alpha: float,
    target_gate_probs: torch.Tensor | None = None,
    draft_gate_probs: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return rows where SpecCascades [OPT] defers from q to p.

    Shapes: probability rows are [N, V]. TV is always computed on processed
    sampler distributions. Optional gate rows let paper mode use pre-sampling
    max probabilities without changing acceptance/residual probabilities.
    """
    assert target_probs.ndim == 2
    assert draft_probs.ndim == 2
    assert target_probs.shape == draft_probs.shape
    if target_gate_probs is None:
        target_gate_probs = target_probs
    if draft_gate_probs is None:
        draft_gate_probs = draft_probs
    assert target_gate_probs.shape == target_probs.shape
    assert draft_gate_probs.shape == draft_probs.shape

    target_max = target_gate_probs.max(dim=-1).values
    draft_max = draft_gate_probs.max(dim=-1).values
    tv_distance = (target_probs - draft_probs).clamp_min(0.0).sum(dim=-1)
    return draft_max < target_max - alpha * tv_distance


def apply_spec_cascade_opt_constraint(
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor,
    alpha: float,
    target_gate_probs: torch.Tensor | None = None,
    draft_gate_probs: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return pi_opt(q, p) = q or p for SpecCascades [OPT] rows."""
    defer_to_target = spec_cascade_opt_defer_mask(
        target_probs,
        draft_probs,
        alpha,
        target_gate_probs=target_gate_probs,
        draft_gate_probs=draft_gate_probs,
    )
    return torch.where(
        defer_to_target.unsqueeze(-1),
        target_probs,
        draft_probs,
    ).contiguous()


def spec_cascade_opt_defer_mask_inplace(
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor,
    alpha: float,
    target_gate_probs: torch.Tensor | None = None,
    draft_gate_probs: torch.Tensor | None = None,
) -> torch.Tensor:
    """Lower-peak version of spec_cascade_opt_defer_mask."""
    assert target_probs.ndim == 2
    assert draft_probs.ndim == 2
    assert target_probs.shape == draft_probs.shape
    if target_gate_probs is None:
        target_gate_probs = target_probs
    if draft_gate_probs is None:
        draft_gate_probs = draft_probs
    assert target_gate_probs.shape == target_probs.shape
    assert draft_gate_probs.shape == draft_probs.shape

    target_max = target_gate_probs.max(dim=-1).values
    draft_max = draft_gate_probs.max(dim=-1).values
    tv_distance = target_probs.to(torch.float32) - draft_probs.to(torch.float32)
    tv_distance.clamp_min_(0.0)
    tv_distance = tv_distance.sum(dim=-1)
    return draft_max < target_max - alpha * tv_distance


def apply_spec_cascade_opt_constraint_inplace(
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor,
    alpha: float,
    target_gate_probs: torch.Tensor | None = None,
    draft_gate_probs: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return pi_opt while reusing `out`/target storage when possible."""
    defer_to_target = spec_cascade_opt_defer_mask_inplace(
        target_probs,
        draft_probs,
        alpha,
        target_gate_probs=target_gate_probs,
        draft_gate_probs=draft_gate_probs,
    )
    relaxed = target_probs if out is None else out
    if relaxed is not target_probs:
        relaxed.copy_(target_probs)
    use_q = ~defer_to_target
    if use_q.any():
        relaxed[use_q] = draft_probs[use_q]
    return relaxed.contiguous()


def sample_spec_cascade_opt_bonus_token_ids(
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor,
    target_bonus_token_ids: torch.Tensor,
    q_bonus_token_ids: torch.Tensor,
    alpha: float,
    target_gate_probs: torch.Tensor | None = None,
    draft_gate_probs: torch.Tensor | None = None,
) -> torch.Tensor:
    """Choose the already sampled q or p bonus token using the OPT gate."""
    target_bonus_token_ids = target_bonus_token_ids.view(-1)
    q_bonus_token_ids = q_bonus_token_ids.view(-1).to(target_bonus_token_ids.dtype)
    assert target_bonus_token_ids.shape == q_bonus_token_ids.shape

    defer_to_target = spec_cascade_opt_defer_mask(
        target_probs,
        draft_probs,
        alpha,
        target_gate_probs=target_gate_probs,
        draft_gate_probs=draft_gate_probs,
    )
    return torch.where(defer_to_target, target_bonus_token_ids, q_bonus_token_ids)


def sample_static_ensemble_bonus_token_ids(
    target_bonus_token_ids: torch.Tensor,
    q_bonus_token_ids: torch.Tensor,
    verifier_weight: float,
    generators: dict[int, torch.Generator],
    row_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """Sample a token from w_p * p_bonus + (1 - w_p) * q_bonus.

    The target and drafter bonus tokens have already been sampled from p_bonus
    and q_bonus respectively, so a Bernoulli verifier/drafter choice is enough.
    """
    target_bonus_token_ids = target_bonus_token_ids.view(-1)
    q_bonus_token_ids = q_bonus_token_ids.view(-1).to(target_bonus_token_ids.dtype)
    assert target_bonus_token_ids.shape == q_bonus_token_ids.shape

    if verifier_weight <= 0.0:
        return q_bonus_token_ids
    if verifier_weight >= 1.0:
        return target_bonus_token_ids

    uniforms = torch.empty(
        target_bonus_token_ids.shape,
        dtype=torch.float32,
        device=target_bonus_token_ids.device,
    )
    uniforms.uniform_()
    if row_ids is None:
        for i, generator in generators.items():
            if i < uniforms.shape[0]:
                uniforms[i].uniform_(generator=generator)
    else:
        for local_i, row_id in enumerate(row_ids.tolist()):
            generator = generators.get(int(row_id))
            if generator is not None:
                uniforms[local_i].uniform_(generator=generator)
    return torch.where(
        uniforms < verifier_weight,
        target_bonus_token_ids,
        q_bonus_token_ids,
    )


def rejection_sample(
    # [num_tokens]
    draft_token_ids: torch.Tensor,
    # [batch_size]
    num_draft_tokens: list[int],
    max_spec_len: int,
    # [batch_size]
    cu_num_draft_tokens: torch.Tensor,
    # [num_tokens, vocab_size]
    draft_probs: torch.Tensor | None,
    # [num_tokens, vocab_size]
    target_logits: torch.Tensor,
    # [batch_size, 1]
    bonus_token_ids: torch.Tensor,
    sampling_metadata: SamplingMetadata,
    # [num_tokens, vocab_size]
    draft_rule_logits: torch.Tensor | None = None,
    synthetic_mode: bool = False,
    synthetic_conditional_rates: torch.Tensor | None = None,
    relaxed_target_method: str = "none",
    cactus_delta: float | None = None,
    verifier_weight: float | None = None,
    fuzzy_divergence: str | None = None,
    fuzzy_threshold: float | None = None,
    spec_cascade_alpha: float | None = None,
    spec_cascade_opt_rule_logits: torch.Tensor | None = None,
    spec_cascade_opt_gate: str = "processed",
    spec_cascade_tok3_rule_logits: torch.Tensor | None = None,
    spec_cascade_tok3_top_set: str = "paper",
    lossy_alpha: float | None = None,
    scd_beta: float | None = None,
    scd_temperature: float | None = None,
    scd_alpha: float | None = None,
    scd_expert_rule_logits: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert draft_token_ids.ndim == 1
    assert draft_probs is None or draft_probs.ndim == 2
    assert draft_rule_logits is None or draft_rule_logits.ndim == 2
    assert cu_num_draft_tokens.ndim == 1
    assert target_logits.ndim == 2
    assert spec_cascade_opt_rule_logits is None or (
        spec_cascade_opt_rule_logits.ndim == 2
    )

    batch_size = len(num_draft_tokens)
    num_tokens = draft_token_ids.shape[0]
    vocab_size = target_logits.shape[-1]
    device = target_logits.device
    assert draft_token_ids.is_contiguous()
    assert draft_probs is None or draft_probs.is_contiguous()
    assert draft_rule_logits is None or draft_rule_logits.is_contiguous()
    assert spec_cascade_opt_rule_logits is None or (
        spec_cascade_opt_rule_logits.is_contiguous()
    )
    assert bonus_token_ids.is_contiguous()
    assert target_logits.shape == (num_tokens, vocab_size)

    fsd_mode = relaxed_target_method == "fsd"
    spec_cascade_opt_mode = relaxed_target_method == "spec_cascade_opt"
    spec_cascade_tok3_mode = relaxed_target_method == "spec_cascade_tok3"
    lossy_mode = relaxed_target_method == "lossy_spec_decode_beta1"
    scd_expert_toppk_mode = relaxed_target_method == "scd_expert_toppk_gated"
    scd_alpha_mode = relaxed_target_method == "scd_alpha"
    scd_mode = scd_expert_toppk_mode or scd_alpha_mode

    # Create output buffer.
    output_token_ids = torch.full(
        (batch_size, max_spec_len + 1),
        PLACEHOLDER_TOKEN_ID,
        dtype=torch.int32,  # Consistent with SamplerOutput.sampled_token_ids.
        device=device,
    )
    all_accepted_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)

    if sampling_metadata.all_greedy:
        is_greedy = None
    else:
        is_greedy = sampling_metadata.temperature == GREEDY_TEMPERATURE
    if (fsd_mode or scd_mode) and not sampling_metadata.all_random:
        # vLLM's warmup sampler can set all_random=False while every row is
        # still stochastic; only true greedy rows are outside these dense paths.
        has_greedy_request = sampling_metadata.all_greedy
        if is_greedy is not None:
            has_greedy_request = bool(torch.any(is_greedy).item())
        if has_greedy_request:
            raise ValueError(
                f"{relaxed_target_method} rejection sampling currently "
                "requires stochastic sampling."
            )

    # Generate uniform probabilities before either kernel because synthetic
    # mode needs them in the greedy kernel too.  Skip only when all requests
    # are greedy *and* synthetic mode is off (the standard fast-path).
    # [num_tokens]
    uniform_probs: torch.Tensor | None = None
    needs_uniform_probs = synthetic_mode or (
        not sampling_metadata.all_greedy and not fsd_mode
    )
    if needs_uniform_probs:
        uniform_probs = generate_uniform_probs(
            num_tokens,
            num_draft_tokens,
            sampling_metadata.generators,
            device,
        )

    if not sampling_metadata.all_random:
        # Rejection sampling for greedy sampling requests.
        verifier_argmax = target_logits.argmax(dim=-1)
        rejection_greedy_sample_kernel[(batch_size,)](
            output_token_ids,
            all_accepted_mask,
            cu_num_draft_tokens,
            draft_token_ids,
            verifier_argmax,
            bonus_token_ids,
            is_greedy,
            max_spec_len,
            uniform_probs,
            synthetic_conditional_rates,
            SYNTHETIC_MODE=synthetic_mode,
        )
        if sampling_metadata.all_greedy:
            return output_token_ids, all_accepted_mask

    # verifier_probs is processed p. Some methods materialize an effective pi
    # row for the generic rejector; fused/specialized methods keep p and encode
    # the relaxed rule in kernel flags or side tensors.
    use_inplace_dense_transforms = use_inplace_relaxed_dense_transforms()
    if use_inplace_dense_transforms and scd_mode:
        verifier_probs = None
        rejection_target_probs = None
    else:
        verifier_probs = target_logits.softmax(dim=-1, dtype=torch.float32)
        rejection_target_probs = verifier_probs
    cactus_mode = relaxed_target_method == "cactus"
    ensemble_mode = relaxed_target_method == "ensemble"
    rfsd_mode = relaxed_target_method in ("rfsd", "fuzzy")
    fsd_family_mode = rfsd_mode or fsd_mode
    use_fused_relaxed_kernel = use_fused_relaxed_kernel_by_default()
    use_fused_fsd_family = (
        use_fused_relaxed_kernel
        and fsd_family_mode
        and sampling_metadata.all_random
        and not synthetic_mode
    )
    use_fused_spec_cascade_opt = (
        use_fused_relaxed_kernel
        and spec_cascade_opt_mode
        and spec_cascade_opt_gate == "processed"
        and sampling_metadata.all_random
        and not synthetic_mode
    )
    fsd_accept: torch.Tensor | None = None
    ensemble_verifier_weight = 1.0
    cactus_rejection_delta = 0.0
    lossy_rejection_alpha = 0.0
    if cactus_mode:
        if draft_probs is None:
            raise ValueError("Cactus rejection sampling requires draft_probs.")
        if cactus_delta is None:
            raise ValueError("Cactus rejection sampling requires cactus_delta.")
        cactus_rejection_delta = cactus_delta
    elif ensemble_mode:
        if draft_probs is None:
            raise ValueError("Static ensemble rejection sampling requires draft_probs.")
        if verifier_weight is None:
            raise ValueError(
                "Static ensemble rejection sampling requires verifier_weight."
            )
        ensemble_verifier_weight = verifier_weight
    elif rfsd_mode:
        if draft_probs is None:
            raise ValueError("rFSD rejection sampling requires draft_probs.")
        if fuzzy_divergence is None:
            raise ValueError("rFSD rejection sampling requires fuzzy_divergence.")
        if fuzzy_threshold is None:
            raise ValueError("rFSD rejection sampling requires fuzzy_threshold.")
        if not use_fused_fsd_family:
            assert verifier_probs is not None
            if use_inplace_dense_transforms:
                rejection_target_probs = apply_rfsd_constraint_inplace(
                    verifier_probs,
                    draft_probs,
                    fuzzy_divergence,
                    fuzzy_threshold,
                    out=verifier_probs,
                )
            else:
                rejection_target_probs = apply_rfsd_constraint(
                    verifier_probs,
                    draft_probs,
                    fuzzy_divergence,
                    fuzzy_threshold,
                )
    elif fsd_mode:
        if draft_probs is None:
            raise ValueError("FSD rejection sampling requires draft_probs.")
        if fuzzy_divergence is None:
            raise ValueError("FSD rejection sampling requires fuzzy_divergence.")
        if fuzzy_threshold is None:
            raise ValueError("FSD rejection sampling requires fuzzy_threshold.")
        if not use_fused_fsd_family:
            assert verifier_probs is not None
            accept_mask_fn = (
                fsd_accept_mask_inplace
                if use_inplace_dense_transforms else fsd_accept_mask
            )
            fsd_accept = accept_mask_fn(
                verifier_probs,
                draft_probs,
                fuzzy_divergence,
                fuzzy_threshold,
            ).contiguous()
    elif spec_cascade_opt_mode:
        if draft_probs is None:
            raise ValueError(
                "SpecCascades [OPT] rejection sampling requires draft_probs."
            )
        if spec_cascade_alpha is None:
            raise ValueError(
                "SpecCascades [OPT] rejection sampling requires "
                "spec_cascade_alpha."
            )
        if spec_cascade_opt_gate not in ("processed", "paper"):
            raise ValueError("spec_cascade_opt_gate must be 'processed' or 'paper'.")
        verifier_gate_probs = None
        draft_gate_probs = None
        if spec_cascade_opt_gate == "paper":
            if draft_rule_logits is None:
                raise ValueError(
                    "SpecCascades [OPT] paper gate requires draft_rule_logits."
                )
            if spec_cascade_opt_rule_logits is None:
                raise ValueError(
                    "SpecCascades [OPT] paper gate requires "
                    "spec_cascade_opt_rule_logits."
                )
            if draft_rule_logits.shape != target_logits.shape:
                raise ValueError("OPT draft rule logits must match target rows.")
            if spec_cascade_opt_rule_logits.shape != target_logits.shape:
                raise ValueError("OPT target rule logits must match target rows.")
            verifier_gate_probs = spec_cascade_opt_rule_logits.softmax(
                dim=-1, dtype=torch.float32
            )
            draft_gate_probs = draft_rule_logits.softmax(dim=-1, dtype=torch.float32)
        if not use_fused_spec_cascade_opt:
            assert verifier_probs is not None
            if use_inplace_dense_transforms:
                rejection_target_probs = apply_spec_cascade_opt_constraint_inplace(
                    verifier_probs,
                    draft_probs,
                    spec_cascade_alpha,
                    target_gate_probs=verifier_gate_probs,
                    draft_gate_probs=draft_gate_probs,
                    out=verifier_probs,
                )
            else:
                rejection_target_probs = apply_spec_cascade_opt_constraint(
                    verifier_probs,
                    draft_probs,
                    spec_cascade_alpha,
                    target_gate_probs=verifier_gate_probs,
                    draft_gate_probs=draft_gate_probs,
                )
    elif spec_cascade_tok3_mode:
        if draft_probs is None:
            raise ValueError(
                "SpecCascades Tok3 rejection sampling requires draft_probs."
            )
        if spec_cascade_alpha is None:
            raise ValueError(
                "SpecCascades Tok3 rejection sampling requires "
                "spec_cascade_alpha."
            )
        if not 0.0 <= spec_cascade_alpha <= 1.0:
            raise ValueError(
                "spec_cascade_alpha must satisfy 0 <= alpha <= 1 for Tok3."
            )
        if spec_cascade_tok3_top_set not in ("paper", "processed"):
            raise ValueError(
                "spec_cascade_tok3_top_set must be 'paper' or 'processed'."
            )
        top_logits = (
            target_logits
            if spec_cascade_tok3_top_set == "processed"
            else spec_cascade_tok3_rule_logits
        )
        if top_logits is None:
            raise ValueError(
                "SpecCascades Tok3 paper top-set mode requires "
                "spec_cascade_tok3_rule_logits."
            )
        top_mask = spec_cascade_tok3_top_mask_from_logits(
            top_logits,
            spec_cascade_alpha,
        )
        assert verifier_probs is not None
        if use_inplace_dense_transforms:
            rejection_target_probs = apply_spec_cascade_tok3_constraint_inplace(
                verifier_probs,
                draft_probs,
                top_mask,
                out=verifier_probs,
            )
        else:
            rejection_target_probs = apply_spec_cascade_tok3_constraint(
                verifier_probs,
                draft_probs,
                top_mask,
            )
    elif lossy_mode:
        if draft_probs is None:
            raise ValueError("Lossy SD rejection sampling requires draft_probs.")
        if lossy_alpha is None:
            raise ValueError("Lossy SD rejection sampling requires lossy_alpha.")
        if not 0.0 <= lossy_alpha < 1.0:
            raise ValueError("lossy_alpha must satisfy 0 <= alpha < 1.")
        lossy_rejection_alpha = lossy_alpha
    elif scd_mode:
        if draft_probs is None:
            raise ValueError("SCD rejection sampling requires draft_probs.")
        if draft_rule_logits is None:
            raise ValueError("SCD rejection sampling requires draft rule logits.")
        if scd_expert_rule_logits is None:
            raise ValueError("SCD rejection sampling requires expert rule logits.")
        if scd_beta is None:
            raise ValueError("SCD rejection sampling requires scd_beta.")
        if draft_rule_logits.shape != target_logits.shape:
            raise ValueError("SCD draft rule logits must match target rows.")
        if scd_expert_rule_logits.shape != target_logits.shape:
            raise ValueError("SCD expert rule logits must match target rows.")

        if scd_expert_toppk_mode:
            gate = torch.isfinite(target_logits)
        else:
            if scd_alpha is None:
                raise ValueError("SCD alpha rejection sampling requires scd_alpha.")
            assert sampling_metadata.temperature is not None
            temperature = expand_batch_to_tokens(
                sampling_metadata.temperature,
                cu_num_draft_tokens,
                num_tokens,
                replace_from=GREEDY_TEMPERATURE,
                replace_to=1,
            )
            gate = scd_alpha_gate_from_logits(
                scd_expert_rule_logits,
                temperature,
                scd_alpha,
            )
        if scd_temperature is None:
            assert sampling_metadata.temperature is not None
            scd_temperature = expand_batch_to_tokens(
                sampling_metadata.temperature,
                cu_num_draft_tokens,
                num_tokens,
                replace_from=GREEDY_TEMPERATURE,
                replace_to=1,
            )
        if use_inplace_dense_transforms:
            rejection_target_probs = apply_scd_constraint_inplace(
                scd_expert_rule_logits,
                draft_rule_logits,
                gate,
                scd_beta,
                scd_temperature,
            )
        else:
            rejection_target_probs = apply_scd_constraint(
                scd_expert_rule_logits,
                draft_rule_logits,
                gate,
                scd_beta,
                scd_temperature,
            )
    assert rejection_target_probs is not None
    assert rejection_target_probs.is_contiguous()

    if use_fused_fsd_family:
        assert draft_probs is not None
        assert fuzzy_divergence is not None
        assert fuzzy_threshold is not None
        return fused_fsd_family_rejection_sample(
            output_token_ids,
            all_accepted_mask,
            draft_token_ids,
            num_draft_tokens,
            cu_num_draft_tokens,
            draft_probs,
            verifier_probs,
            bonus_token_ids,
            uniform_probs,
            sampling_metadata,
            max_spec_len,
            exact_fsd=fsd_mode,
            divergence=fuzzy_divergence,
            threshold=fuzzy_threshold,
        )

    if use_fused_spec_cascade_opt:
        assert draft_probs is not None
        assert spec_cascade_alpha is not None
        return fused_spec_cascade_opt_rejection_sample(
            output_token_ids,
            all_accepted_mask,
            draft_token_ids,
            num_draft_tokens,
            cu_num_draft_tokens,
            draft_probs,
            verifier_probs,
            bonus_token_ids,
            uniform_probs,
            sampling_metadata,
            max_spec_len,
            alpha=spec_cascade_alpha,
        )

    # Sample recovered tokens for each position.
    # [num_tokens]
    if fsd_mode:
        recovered_token_ids = sample_target_tokens(
            num_draft_tokens,
            verifier_probs,
            sampling_metadata,
        )
    elif ensemble_mode and ensemble_verifier_weight == 0.0:
        recovered_token_ids = torch.empty_like(draft_token_ids)
    else:
        recovered_token_ids = sample_recovered_tokens(
            max_spec_len,
            num_draft_tokens,
            cu_num_draft_tokens,
            draft_token_ids,
            draft_probs,
            rejection_target_probs,
            sampling_metadata,
            device,
            cactus_mode=cactus_mode,
            cactus_delta=cactus_rejection_delta,
        )

    # Rejection sampling for random sampling requests.
    if uniform_probs is None:
        uniform_probs = torch.empty((num_tokens,), dtype=torch.float64, device=device)
    rejection_random_sample_kernel[(batch_size,)](
        output_token_ids,
        all_accepted_mask,
        cu_num_draft_tokens,
        draft_token_ids,
        draft_probs,
        rejection_target_probs,
        bonus_token_ids,
        recovered_token_ids,
        uniform_probs,
        fsd_accept,
        is_greedy,
        max_spec_len,
        vocab_size,
        synthetic_conditional_rates,
        NO_DRAFT_PROBS=draft_probs is None,
        SYNTHETIC_MODE=synthetic_mode,
        CACTUS_MODE=cactus_mode,
        CACTUS_DELTA=cactus_rejection_delta,
        ENSEMBLE_MODE=ensemble_mode,
        ENSEMBLE_VERIFIER_WEIGHT=ensemble_verifier_weight,
        FSD_MODE=fsd_mode,
        LOSSY_MODE=lossy_mode,
        LOSSY_ALPHA=lossy_rejection_alpha,
    )
    return output_token_ids, all_accepted_mask


def apply_cactus_constraint(
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor,
    draft_token_ids: torch.Tensor,
    delta: float,
) -> torch.Tensor:
    """Return Cactus' proposal-conditioned row H_x, not marginal pi_cactus.

    Shapes: target_probs/draft_probs are [N, V], draft_token_ids is [N].
    The public Cactus implementation boosts the verifier probability of the
    proposed token and rescales all other verifier probabilities. Only
    draft_token_ids are used; draft_probs is accepted for call-site symmetry.
    """
    del draft_probs
    assert target_probs.ndim == 2
    assert draft_token_ids.ndim == 1
    assert target_probs.shape[0] == draft_token_ids.shape[0]

    selected = draft_token_ids.to(torch.int64).unsqueeze(-1)
    selected_target_probs = target_probs.gather(-1, selected).squeeze(-1)
    boost = torch.sqrt(2 * delta * selected_target_probs * (1 - selected_target_probs))
    relaxed_selected = (selected_target_probs + boost).clamp(min=0.0, max=1.0)

    scale = ((1 - relaxed_selected) / (1 - selected_target_probs)).nan_to_num(
        nan=1.0, posinf=1.0, neginf=0.0
    )
    relaxed = target_probs * scale.unsqueeze(-1)
    relaxed.scatter_(-1, selected, relaxed_selected.unsqueeze(-1))
    relaxed.clamp_(min=0.0, max=1.0)
    return relaxed.contiguous()


def apply_sampling_constraints(
    logits: torch.Tensor,  # [num_tokens, vocab_size]
    cu_num_draft_tokens: torch.Tensor,  # [batch_size]
    sampling_metadata: SamplingMetadata,
) -> torch.Tensor:
    """Process logits based on sampling metadata.

    This function applies temperature scaling to the logits,
    as well as top-k and top-p. For greedy decoding, it returns
    the original logits.

    Args:
        logits: Input logits tensor to be processed.
        cu_num_draft_tokens: Cumulative number of draft tokens.
        sampling_metadata: Metadata containing sampling parameters such as
            temperature and whether greedy sampling is used.

    Returns:
        torch.Tensor: Processed logits if non-greedy sampling is used,
        otherwise returns the original logits.
    """
    assert logits.ndim == 2
    assert cu_num_draft_tokens.ndim == 1
    if sampling_metadata.all_greedy:
        return logits

    num_tokens = logits.shape[0]
    temperature = expand_batch_to_tokens(
        sampling_metadata.temperature,
        cu_num_draft_tokens,
        num_tokens,
        replace_from=GREEDY_TEMPERATURE,
        replace_to=1,
    )
    # NOTE(woosuk): Update `logits` in place to avoid allocating a new tensor.
    logits.div_(temperature.unsqueeze(-1))

    # Get expanded top_k and top_p tensors.
    top_k = None
    if sampling_metadata.top_k is not None:
        top_k = expand_batch_to_tokens(
            sampling_metadata.top_k,
            cu_num_draft_tokens,
            num_tokens,
        )
    top_p = None
    if sampling_metadata.top_p is not None:
        top_p = expand_batch_to_tokens(
            sampling_metadata.top_p,
            cu_num_draft_tokens,
            num_tokens,
        )

    # NOTE(woosuk): `apply_top_k_top_p` uses sorting to calculate the mask,
    # which is slow for large vocab sizes. This may cause performance issues.
    return apply_top_k_top_p(logits, top_k, top_p)


def expand_batch_to_tokens(
    x: torch.Tensor,  # [batch_size]
    cu_num_tokens: torch.Tensor,  # [batch_size]
    num_tokens: int,
    replace_from: int = 0,
    replace_to: int = 0,
) -> torch.Tensor:
    """Expand [batch_size] tensor to [num_tokens] tensor based on the number of
    tokens per batch in cu_num_tokens.

    For example, if x = [a, b, c] and cu_num_tokens = [2, 5, 6], then
    num_tokens = 6, and expanded_x = [a, a, b, b, b, c].

    Args:
        x: [batch_size] tensor to expand.
        cu_num_tokens: [batch_size] tensor containing the cumulative number of
            tokens per batch. Each element represents the total number of
            tokens up to and including that batch.
        num_tokens: Total number of tokens.
        replace_from: int = 0
            Value to be replaced if it is found in x.
        replace_to: int = 0
            Value to replace with when replace_from is found.
    Returns:
        expanded_x: [num_tokens] tensor.
    """
    batch_size = x.shape[0]
    assert cu_num_tokens.shape[0] == batch_size
    expanded_x = x.new_empty(num_tokens)
    expand_kernel[(batch_size,)](
        expanded_x,
        x,
        cu_num_tokens,
        replace_from,
        replace_to,
        MAX_NUM_TOKENS=MAX_SPEC_LEN,  # To avoid recompilation.
    )
    return expanded_x


def generate_uniform_probs(
    num_tokens: int,
    num_draft_tokens: list[int],
    generators: dict[int, torch.Generator],
    device: torch.device,
) -> torch.Tensor:
    """
    Generates a batch of uniform random samples, with optional seeding
    if available.

    This method creates a tensor of shape `(num_tokens, )` filled
    with uniform random values in the range [0, 1). If `generators` is provided,
    the requests with their own seeds will use the provided `torch.Generator`
    for reproducibility. The samples for the other requests will be generated
    without a seed.

    Args:
        num_tokens: int
            Total number of tokens.
        num_draft_tokens: List[List[int]]
            Number of draft tokens per request.
        generators: Optional[Dict[int, torch.Generator]]
            A dictionary mapping indices in the batch to
            `torch.Generator` objects.
        device: torch.device
            The device on which to allocate the tensor.
    Returns:
        uniform_rand: torch.Tensor
            A tensor of shape `(num_tokens, )` containing uniform
            random values in the range [0, 1).
    """
    # NOTE(woosuk): We deliberately use float64 instead of float32 here
    # because when using float32, there's a non-negligible chance that
    # uniform_prob is sampled to be exact 0.0 as reported in
    # https://github.com/pytorch/pytorch/issues/16706. Using float64
    # mitigates the issue.
    uniform_probs = torch.rand(
        (num_tokens,),
        dtype=torch.float64,
        device=device,
    )
    start_idx = 0
    for req_idx, n in enumerate(num_draft_tokens):
        # Do not generate random numbers for requests with no draft tokens.
        # This can be important for reproducibility.
        if n == 0:
            continue
        end_idx = start_idx + n
        generator = generators.get(req_idx)
        if generator is not None:
            uniform_probs[start_idx:end_idx].uniform_(generator=generator)
        start_idx = end_idx
    return uniform_probs


def sample_target_tokens(
    num_draft_tokens: list[int],
    target_probs: torch.Tensor,
    sampling_metadata: SamplingMetadata,
) -> torch.Tensor:
    """Sample exact-FSD rejection tokens directly from target p rows [N, V]."""
    q = torch.empty_like(target_probs)
    q.exponential_()

    start_idx = 0
    for req_idx, n in enumerate(num_draft_tokens):
        end_idx = start_idx + n
        if n > 0:
            generator = sampling_metadata.generators.get(req_idx)
            if generator is not None:
                q[start_idx:end_idx].exponential_(generator=generator)
        start_idx = end_idx

    return target_probs.div(q).argmax(dim=-1).view(-1)


def generate_recovery_inv_q(
    batch_size: int,
    vocab_size: int,
    num_draft_tokens: list[int],
    generators: dict[int, torch.Generator],
    device: torch.device,
) -> torch.Tensor:
    """Return exponential-race inverse samples [B, V] for recovery scans."""
    q = torch.empty(
        (batch_size, vocab_size),
        dtype=torch.float32,
        device=device,
    )
    q.exponential_()
    for i, generator in generators.items():
        if i < batch_size and num_draft_tokens[i] > 0:
            q[i].exponential_(generator=generator)
    return q.reciprocal()


def fused_fsd_family_rejection_sample(
    output_token_ids: torch.Tensor,
    all_accepted_mask: torch.Tensor,
    draft_token_ids: torch.Tensor,
    num_draft_tokens: list[int],
    cu_num_draft_tokens: torch.Tensor,
    draft_probs: torch.Tensor,
    target_probs: torch.Tensor,
    bonus_token_ids: torch.Tensor,
    uniform_probs: torch.Tensor | None,
    sampling_metadata: SamplingMetadata,
    max_spec_len: int,
    exact_fsd: bool,
    divergence: str,
    threshold: float,
    recovery_inv_q: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused FSD/rFSD verification for all-random draft-model rows.

    The kernel computes rowwise divergence, decides the draft prefix, and scans
    vocab only for the first rejected position in each request.
    """
    assert target_probs.ndim == 2
    assert draft_probs.ndim == 2
    assert target_probs.shape == draft_probs.shape
    if divergence not in ("kl", "js"):
        raise ValueError(f"Unknown FSD-family divergence: {divergence!r}.")
    if uniform_probs is None:
        uniform_probs = torch.empty(
            (draft_token_ids.shape[0],),
            dtype=torch.float64,
            device=target_probs.device,
        )
    if recovery_inv_q is None:
        recovery_inv_q = generate_recovery_inv_q(
            len(num_draft_tokens),
            target_probs.shape[-1],
            num_draft_tokens,
            sampling_metadata.generators,
            target_probs.device,
        )

    BLOCK_SIZE = 8192
    fused_fsd_family_rejection_sample_kernel[(len(num_draft_tokens),)](
        output_token_ids,
        all_accepted_mask,
        cu_num_draft_tokens,
        draft_token_ids,
        draft_probs,
        target_probs,
        bonus_token_ids,
        uniform_probs,
        recovery_inv_q,
        max_spec_len,
        target_probs.shape[-1],
        threshold,
        BLOCK_SIZE,
        EXACT_FSD=exact_fsd,
        DIVERGENCE_IS_JS=divergence == "js",
    )
    return output_token_ids, all_accepted_mask


def fused_spec_cascade_opt_rejection_sample(
    output_token_ids: torch.Tensor,
    all_accepted_mask: torch.Tensor,
    draft_token_ids: torch.Tensor,
    num_draft_tokens: list[int],
    cu_num_draft_tokens: torch.Tensor,
    draft_probs: torch.Tensor,
    target_probs: torch.Tensor,
    bonus_token_ids: torch.Tensor,
    uniform_probs: torch.Tensor | None,
    sampling_metadata: SamplingMetadata,
    max_spec_len: int,
    alpha: float,
    recovery_inv_q: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused SpecCascades [OPT] verification for all-random rows.

    Each token row computes max(p), max(q), TV(p, q), then either accepts the
    q-sampled draft token directly or runs vanilla SD against p.
    """
    assert target_probs.ndim == 2
    assert draft_probs.ndim == 2
    assert target_probs.shape == draft_probs.shape
    if uniform_probs is None:
        uniform_probs = torch.empty(
            (draft_token_ids.shape[0],),
            dtype=torch.float64,
            device=target_probs.device,
        )
    if recovery_inv_q is None:
        recovery_inv_q = generate_recovery_inv_q(
            len(num_draft_tokens),
            target_probs.shape[-1],
            num_draft_tokens,
            sampling_metadata.generators,
            target_probs.device,
        )

    BLOCK_SIZE = 8192
    fused_spec_cascade_opt_rejection_sample_kernel[(len(num_draft_tokens),)](
        output_token_ids,
        all_accepted_mask,
        cu_num_draft_tokens,
        draft_token_ids,
        draft_probs,
        target_probs,
        bonus_token_ids,
        uniform_probs,
        recovery_inv_q,
        max_spec_len,
        target_probs.shape[-1],
        alpha,
        BLOCK_SIZE,
    )
    return output_token_ids, all_accepted_mask


def sample_recovered_tokens(
    max_spec_len: int,
    num_draft_tokens: list[int],
    # [batch_size]
    cu_num_draft_tokens: torch.Tensor,
    # [num_tokens]
    draft_token_ids: torch.Tensor,
    # [num_tokens, vocab_size]
    draft_probs: torch.Tensor | None,
    # [num_tokens, vocab_size]
    target_probs: torch.Tensor,
    sampling_metadata: SamplingMetadata,
    device: torch.device,
    cactus_mode: bool = False,
    cactus_delta: float = 0.0,
) -> torch.Tensor:
    # NOTE(woosuk): Create only one distribution for each request.
    batch_size = len(num_draft_tokens)
    vocab_size = target_probs.shape[-1]
    inv_q = generate_recovery_inv_q(
        batch_size,
        vocab_size,
        num_draft_tokens,
        sampling_metadata.generators,
        device,
    )

    recovered_token_ids = torch.empty_like(draft_token_ids)
    BLOCK_SIZE = 8192
    sample_recovered_tokens_kernel[(batch_size, max_spec_len)](
        recovered_token_ids,
        cu_num_draft_tokens,
        draft_token_ids,
        draft_probs,
        target_probs,
        inv_q,
        vocab_size,
        BLOCK_SIZE,
        NO_DRAFT_PROBS=draft_probs is None,
        CACTUS_MODE=cactus_mode,
        CACTUS_DELTA=cactus_delta,
    )
    return recovered_token_ids


# NOTE(woosuk): Avoid specialization to prevent unnecessary recompilation.
@triton.jit(do_not_specialize=["max_spec_len"])
def fused_fsd_family_rejection_sample_kernel(
    output_token_ids_ptr,  # [batch_size, max_spec_len + 1]
    all_accepted_mask_ptr,  # [batch_size]
    cu_num_draft_tokens_ptr,  # [batch_size]
    draft_token_ids_ptr,  # [num_tokens]
    draft_probs_ptr,  # [num_tokens, vocab_size]
    target_probs_ptr,  # [num_tokens, vocab_size]
    bonus_token_ids_ptr,  # [batch_size]
    uniform_probs_ptr,  # [num_tokens], ignored for exact FSD
    recovery_inv_q_ptr,  # [batch_size, vocab_size]
    max_spec_len,
    vocab_size,
    threshold,
    BLOCK_SIZE: tl.constexpr,
    EXACT_FSD: tl.constexpr,
    DIVERGENCE_IS_JS: tl.constexpr,
):
    req_idx = tl.program_id(0)
    start_idx = 0 if req_idx == 0 else tl.load(cu_num_draft_tokens_ptr + req_idx - 1)
    end_idx = tl.load(cu_num_draft_tokens_ptr + req_idx)
    num_draft_tokens = end_idx - start_idx

    rejected = False
    for pos in range(num_draft_tokens):
        if not rejected:
            token_idx = start_idx + pos
            draft_token_id = tl.load(draft_token_ids_ptr + token_idx).to(tl.int32)

            divergence = 0.0
            for v in range(0, vocab_size, BLOCK_SIZE):
                vocab_offset = v + tl.arange(0, BLOCK_SIZE)
                vocab_mask = vocab_offset < vocab_size
                target_prob = tl.load(
                    target_probs_ptr + token_idx * vocab_size + vocab_offset,
                    mask=vocab_mask,
                    other=0.0,
                )
                draft_prob = tl.load(
                    draft_probs_ptr + token_idx * vocab_size + vocab_offset,
                    mask=vocab_mask,
                    other=0.0,
                )

                safe_target = tl.maximum(target_prob, 1.0e-30)
                safe_draft = tl.maximum(draft_prob, 1.0e-30)
                if DIVERGENCE_IS_JS:
                    mixed_prob = 0.5 * (target_prob + draft_prob)
                    safe_mixed = tl.maximum(mixed_prob, 1.0e-30)
                    term = 0.5 * target_prob * (
                        tl.log(safe_target) - tl.log(safe_mixed)
                    )
                    term += 0.5 * draft_prob * (
                        tl.log(safe_draft) - tl.log(safe_mixed)
                    )
                else:
                    term = target_prob * (tl.log(safe_target) - tl.log(safe_draft))
                divergence += tl.sum(tl.where(vocab_mask, term, 0.0), axis=0)

            if EXACT_FSD:
                gate_accepted = divergence < threshold
            else:
                gate_accepted = divergence <= threshold

            if gate_accepted:
                token_id = draft_token_id
            else:
                accepted = False
                if not EXACT_FSD:
                    selected_draft_prob = tl.load(
                        draft_probs_ptr
                        + token_idx * vocab_size
                        + draft_token_id
                    )
                    selected_target_prob = tl.load(
                        target_probs_ptr
                        + token_idx * vocab_size
                        + draft_token_id
                    )
                    uniform_prob = tl.load(uniform_probs_ptr + token_idx)
                    accepted = (
                        selected_draft_prob > 0.0
                        and selected_target_prob / selected_draft_prob >= uniform_prob
                    )

                if accepted:
                    token_id = draft_token_id
                else:
                    rejected = True
                    max_val = float("-inf")
                    recovered_id = 0
                    for v in range(0, vocab_size, BLOCK_SIZE):
                        vocab_offset = v + tl.arange(0, BLOCK_SIZE)
                        vocab_mask = vocab_offset < vocab_size
                        target_prob = tl.load(
                            target_probs_ptr + token_idx * vocab_size + vocab_offset,
                            mask=vocab_mask,
                            other=0.0,
                        )
                        if EXACT_FSD:
                            prob = target_prob
                        else:
                            draft_prob = tl.load(
                                draft_probs_ptr
                                + token_idx * vocab_size
                                + vocab_offset,
                                mask=vocab_mask,
                                other=0.0,
                            )
                            prob = tl.maximum(target_prob - draft_prob, 0.0)

                        inv_q = tl.load(
                            recovery_inv_q_ptr + req_idx * vocab_size + vocab_offset,
                            mask=vocab_mask,
                            other=0.0,
                        )
                        score = tl.where(vocab_mask, prob * inv_q, float("-inf"))
                        local_max, local_id = tl.max(
                            score, axis=0, return_indices=True
                        )

                        if local_max > max_val:
                            max_val = local_max
                            recovered_id = v + local_id
                    token_id = recovered_id

            tl.store(
                output_token_ids_ptr + req_idx * (max_spec_len + 1) + pos,
                token_id,
            )

    if not rejected:
        bonus_token_id = tl.load(bonus_token_ids_ptr + req_idx)
        tl.store(
            output_token_ids_ptr + req_idx * (max_spec_len + 1) + num_draft_tokens,
            bonus_token_id,
        )
    tl.store(all_accepted_mask_ptr + req_idx, rejected == False)


# NOTE(woosuk): Avoid specialization to prevent unnecessary recompilation.
@triton.jit(do_not_specialize=["max_spec_len"])
def fused_spec_cascade_opt_rejection_sample_kernel(
    output_token_ids_ptr,  # [batch_size, max_spec_len + 1]
    all_accepted_mask_ptr,  # [batch_size]
    cu_num_draft_tokens_ptr,  # [batch_size]
    draft_token_ids_ptr,  # [num_tokens]
    draft_probs_ptr,  # [num_tokens, vocab_size]
    target_probs_ptr,  # [num_tokens, vocab_size]
    bonus_token_ids_ptr,  # [batch_size]
    uniform_probs_ptr,  # [num_tokens]
    recovery_inv_q_ptr,  # [batch_size, vocab_size]
    max_spec_len,
    vocab_size,
    alpha,
    BLOCK_SIZE: tl.constexpr,
):
    req_idx = tl.program_id(0)
    start_idx = 0 if req_idx == 0 else tl.load(cu_num_draft_tokens_ptr + req_idx - 1)
    end_idx = tl.load(cu_num_draft_tokens_ptr + req_idx)
    num_draft_tokens = end_idx - start_idx

    rejected = False
    for pos in range(num_draft_tokens):
        if not rejected:
            token_idx = start_idx + pos
            draft_token_id = tl.load(draft_token_ids_ptr + token_idx).to(tl.int32)

            target_max = 0.0
            draft_max = 0.0
            tv_distance = 0.0
            max_residual_score = float("-inf")
            recovered_id = 0

            for v in range(0, vocab_size, BLOCK_SIZE):
                vocab_offset = v + tl.arange(0, BLOCK_SIZE)
                vocab_mask = vocab_offset < vocab_size
                target_prob = tl.load(
                    target_probs_ptr + token_idx * vocab_size + vocab_offset,
                    mask=vocab_mask,
                    other=0.0,
                )
                draft_prob = tl.load(
                    draft_probs_ptr + token_idx * vocab_size + vocab_offset,
                    mask=vocab_mask,
                    other=0.0,
                )
                positive_delta = tl.maximum(target_prob - draft_prob, 0.0)

                target_max = tl.maximum(
                    target_max,
                    tl.max(tl.where(vocab_mask, target_prob, 0.0), axis=0),
                )
                draft_max = tl.maximum(
                    draft_max,
                    tl.max(tl.where(vocab_mask, draft_prob, 0.0), axis=0),
                )
                tv_distance += tl.sum(
                    tl.where(vocab_mask, positive_delta, 0.0), axis=0
                )

                inv_q = tl.load(
                    recovery_inv_q_ptr + req_idx * vocab_size + vocab_offset,
                    mask=vocab_mask,
                    other=0.0,
                )
                residual_score = tl.where(
                    vocab_mask,
                    positive_delta * inv_q,
                    float("-inf"),
                )
                local_max, local_id = tl.max(
                    residual_score, axis=0, return_indices=True
                )
                if local_max > max_residual_score:
                    max_residual_score = local_max
                    recovered_id = v + local_id

            defer_to_target = draft_max < target_max - alpha * tv_distance
            if not defer_to_target:
                token_id = draft_token_id
            else:
                selected_draft_prob = tl.load(
                    draft_probs_ptr + token_idx * vocab_size + draft_token_id
                )
                selected_target_prob = tl.load(
                    target_probs_ptr + token_idx * vocab_size + draft_token_id
                )
                uniform_prob = tl.load(uniform_probs_ptr + token_idx)
                accepted = (
                    selected_draft_prob > 0.0
                    and selected_target_prob / selected_draft_prob >= uniform_prob
                )
                if accepted:
                    token_id = draft_token_id
                else:
                    rejected = True
                    token_id = recovered_id

            tl.store(
                output_token_ids_ptr + req_idx * (max_spec_len + 1) + pos,
                token_id,
            )

    if not rejected:
        bonus_token_id = tl.load(bonus_token_ids_ptr + req_idx)
        tl.store(
            output_token_ids_ptr + req_idx * (max_spec_len + 1) + num_draft_tokens,
            bonus_token_id,
        )
    tl.store(all_accepted_mask_ptr + req_idx, rejected == False)


# NOTE(woosuk): Avoid specialization to prevent unnecessary recompilation.
@triton.jit(do_not_specialize=["max_spec_len"])
def rejection_greedy_sample_kernel(
    output_token_ids_ptr,  # [batch_size, max_spec_len + 1]
    all_accepted_mask_ptr,  # [batch_size]
    cu_num_draft_tokens_ptr,  # [batch_size]
    draft_token_ids_ptr,  # [num_tokens]
    target_argmax_ptr,  # [num_tokens]
    bonus_token_ids_ptr,  # [batch_size]
    is_greedy_ptr,  # [batch_size] or None
    max_spec_len,
    uniform_probs_ptr,  # [num_tokens] or None (synthetic mode only)
    synthetic_conditional_rates_ptr,  # [num_speculative_tokens] or None
    SYNTHETIC_MODE: tl.constexpr,
):
    req_idx = tl.program_id(0)
    # FIXME(woosuk): Because is_greedy_ptr is not None at profiling run,
    # re-compilation may happen during runtime when is_greedy_ptr is None.
    is_greedy = True if is_greedy_ptr is None else tl.load(is_greedy_ptr + req_idx)
    if not is_greedy:
        # Early exit for non-greedy sampling requests.
        return

    start_idx = 0 if req_idx == 0 else tl.load(cu_num_draft_tokens_ptr + req_idx - 1)
    end_idx = tl.load(cu_num_draft_tokens_ptr + req_idx)
    num_draft_tokens = end_idx - start_idx

    rejected = False
    for pos in range(num_draft_tokens):
        if not rejected:
            draft_token_id = tl.load(
                draft_token_ids_ptr + start_idx + pos
            ).to(tl.int32)
            target_argmax_id = tl.load(target_argmax_ptr + start_idx + pos).to(tl.int32)
            if SYNTHETIC_MODE:
                uniform_prob = tl.load(uniform_probs_ptr + start_idx + pos)
                rate = tl.load(synthetic_conditional_rates_ptr + pos)
                accepted = uniform_prob < rate
                token_id = draft_token_id if accepted else target_argmax_id
                rejected = not accepted
            else:
                token_id = target_argmax_id
                rejected = draft_token_id != target_argmax_id
            tl.store(
                output_token_ids_ptr + req_idx * (max_spec_len + 1) + pos,
                token_id,
            )

    if not rejected:
        # If all tokens are accepted, append the bonus token.
        bonus_token_id = tl.load(bonus_token_ids_ptr + req_idx)
        tl.store(
            output_token_ids_ptr + req_idx * (max_spec_len + 1) + num_draft_tokens,
            bonus_token_id,
        )
    tl.store(all_accepted_mask_ptr + req_idx, rejected == False)


# NOTE(woosuk): Avoid specialization to prevent unnecessary recompilation.
@triton.jit(do_not_specialize=["max_spec_len"])
def rejection_random_sample_kernel(
    output_token_ids_ptr,  # [batch_size, max_spec_len + 1]
    all_accepted_mask_ptr,  # [batch_size]
    cu_num_draft_tokens_ptr,  # [batch_size]
    draft_token_ids_ptr,  # [num_tokens]
    draft_probs_ptr,  # [num_tokens, vocab_size] or None
    target_probs_ptr,  # [num_tokens, vocab_size]
    bonus_token_ids_ptr,  # [batch_size]
    recovered_token_ids_ptr,  # [num_tokens]
    uniform_probs_ptr,  # [num_tokens]
    fsd_accept_mask_ptr,  # [num_tokens] or None
    is_greedy_ptr,  # [batch_size]
    max_spec_len,
    vocab_size,
    synthetic_conditional_rates_ptr,  # [num_speculative_tokens] or None
    NO_DRAFT_PROBS: tl.constexpr,
    SYNTHETIC_MODE: tl.constexpr,
    CACTUS_MODE: tl.constexpr,
    CACTUS_DELTA: tl.constexpr,
    ENSEMBLE_MODE: tl.constexpr,
    ENSEMBLE_VERIFIER_WEIGHT: tl.constexpr,
    FSD_MODE: tl.constexpr,
    LOSSY_MODE: tl.constexpr,
    LOSSY_ALPHA: tl.constexpr,
):
    req_idx = tl.program_id(0)
    is_greedy = tl.load(is_greedy_ptr + req_idx)
    if is_greedy:
        # Early exit for greedy sampling requests.
        return

    start_idx = 0 if req_idx == 0 else tl.load(cu_num_draft_tokens_ptr + req_idx - 1)
    end_idx = tl.load(cu_num_draft_tokens_ptr + req_idx)
    num_draft_tokens = end_idx - start_idx

    rejected = False
    for pos in range(num_draft_tokens):
        if not rejected:
            draft_token_id = tl.load(
                draft_token_ids_ptr + start_idx + pos
            ).to(tl.int32)
            if SYNTHETIC_MODE:
                uniform_prob = tl.load(uniform_probs_ptr + start_idx + pos)
                rate = tl.load(synthetic_conditional_rates_ptr + pos)
                accepted = uniform_prob < rate
            elif FSD_MODE:
                accepted = tl.load(fsd_accept_mask_ptr + start_idx + pos)
            else:
                if NO_DRAFT_PROBS:
                    draft_prob = 1
                else:
                    draft_prob = tl.load(
                        draft_probs_ptr
                        + (start_idx + pos) * vocab_size
                        + draft_token_id
                    )
                target_prob = tl.load(
                    target_probs_ptr + (start_idx + pos) * vocab_size + draft_token_id
                )
                uniform_prob = tl.load(uniform_probs_ptr + start_idx + pos)
                # NOTE(woosuk): While the draft probability should never be 0,
                # we check it to avoid NaNs. If it happens to be 0, we reject.
                if CACTUS_MODE:
                    boost = tl.sqrt(
                        2.0 * CACTUS_DELTA * target_prob * (1.0 - target_prob)
                    )
                    relaxed_target_prob = tl.minimum(target_prob + boost, 1.0)
                    accepted = (
                        draft_prob > 0
                        and relaxed_target_prob / draft_prob >= uniform_prob
                    )
                elif ENSEMBLE_MODE and ENSEMBLE_VERIFIER_WEIGHT == 0.0:
                    accepted = True
                elif ENSEMBLE_MODE:
                    safe_draft_prob = tl.maximum(draft_prob, 1.0e-30)
                    accept_ratio = (
                        (1.0 - ENSEMBLE_VERIFIER_WEIGHT)
                        + ENSEMBLE_VERIFIER_WEIGHT * target_prob / safe_draft_prob
                    )
                    accepted = draft_prob > 0 and accept_ratio >= uniform_prob
                elif LOSSY_MODE:
                    safe_draft_prob = tl.maximum(draft_prob, 1.0e-30)
                    accept_ratio = (
                        target_prob / ((1.0 - LOSSY_ALPHA) * safe_draft_prob)
                    )
                    accepted = draft_prob > 0 and accept_ratio >= uniform_prob
                else:
                    accepted = (
                        draft_prob > 0 and target_prob / draft_prob >= uniform_prob
                    )
            if accepted:
                token_id = draft_token_id
            else:
                rejected = True
                token_id = tl.load(
                    recovered_token_ids_ptr + start_idx + pos
                ).to(tl.int32)
            tl.store(
                output_token_ids_ptr + req_idx * (max_spec_len + 1) + pos, token_id
            )

    if not rejected:
        # If all tokens are accepted, append the bonus token.
        bonus_token_id = tl.load(bonus_token_ids_ptr + req_idx)
        tl.store(
            output_token_ids_ptr + req_idx * (max_spec_len + 1) + num_draft_tokens,
            bonus_token_id,
        )
    tl.store(all_accepted_mask_ptr + req_idx, rejected == False)


# NOTE(woosuk): Avoid specialization to prevent unnecessary recompilation.
@triton.jit(do_not_specialize=["replace_from", "replace_to"])
def expand_kernel(
    output_ptr,  # [num_tokens]
    input_ptr,  # [batch_size]
    cu_num_tokens_ptr,  # [batch_size]
    replace_from,
    replace_to,
    MAX_NUM_TOKENS: tl.constexpr,
):
    req_idx = tl.program_id(0)
    if req_idx == 0:  # noqa: SIM108
        start_idx = 0
    else:
        start_idx = tl.load(cu_num_tokens_ptr + req_idx - 1)
    end_idx = tl.load(cu_num_tokens_ptr + req_idx)
    num_tokens = end_idx - start_idx

    src_val = tl.load(input_ptr + req_idx)
    src_val = tl.where(src_val == replace_from, replace_to, src_val)
    offset = tl.arange(0, MAX_NUM_TOKENS)
    tl.store(output_ptr + start_idx + offset, src_val, mask=offset < num_tokens)


@triton.jit
def sample_recovered_tokens_kernel(
    output_token_ids_ptr,  # [num_tokens]
    cu_num_draft_tokens_ptr,  # [batch_size]
    draft_token_ids_ptr,  # [num_tokens]
    draft_probs_ptr,  # [num_tokens, vocab_size] or None
    target_probs_ptr,  # [num_tokens, vocab_size]
    inv_q_ptr,  # [batch_size, vocab_size]
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
    NO_DRAFT_PROBS: tl.constexpr,
    CACTUS_MODE: tl.constexpr,
    CACTUS_DELTA: tl.constexpr,
):
    req_idx = tl.program_id(0)
    start_idx = 0 if req_idx == 0 else tl.load(cu_num_draft_tokens_ptr + req_idx - 1)
    end_idx = tl.load(cu_num_draft_tokens_ptr + req_idx)
    num_draft_tokens = end_idx - start_idx

    # Early exit for out-of-range positions.
    pos = tl.program_id(1)
    if pos >= num_draft_tokens:
        return

    token_idx = start_idx + pos

    if NO_DRAFT_PROBS or CACTUS_MODE:
        draft_token_id = tl.load(draft_token_ids_ptr + token_idx).to(tl.int32)
    if CACTUS_MODE:
        selected_target_prob = tl.load(
            target_probs_ptr + token_idx * vocab_size + draft_token_id
        )
        boost = tl.sqrt(
            2.0
            * CACTUS_DELTA
            * selected_target_prob
            * (1.0 - selected_target_prob)
        )
        relaxed_selected = tl.minimum(selected_target_prob + boost, 1.0)
        denom = 1.0 - selected_target_prob
        scale = tl.where(denom > 0.0, (1.0 - relaxed_selected) / denom, 1.0)

    max_val = float("-inf")
    recovered_id = 0
    for v in range(0, vocab_size, BLOCK_SIZE):
        vocab_offset = v + tl.arange(0, BLOCK_SIZE)
        vocab_mask = vocab_offset < vocab_size

        if NO_DRAFT_PROBS:
            prob = tl.load(
                target_probs_ptr + token_idx * vocab_size + vocab_offset,
                mask=(vocab_mask & (vocab_offset != draft_token_id)),
                other=0.0,
            )
        else:
            draft_prob = tl.load(
                draft_probs_ptr + token_idx * vocab_size + vocab_offset,
                mask=vocab_mask,
                other=0.0,
            )
            target_prob = tl.load(
                target_probs_ptr + token_idx * vocab_size + vocab_offset,
                mask=vocab_mask,
                other=0.0,
            )
            if CACTUS_MODE:
                relaxed_target_prob = target_prob * scale
                relaxed_target_prob = tl.where(
                    vocab_offset == draft_token_id,
                    relaxed_selected,
                    relaxed_target_prob,
                )
                target_prob = tl.minimum(tl.maximum(relaxed_target_prob, 0.0), 1.0)
            prob = tl.maximum(target_prob - draft_prob, 0.0)
            # NOTE(woosuk): We don't need `prob = prob / tl.sum(prob)` here because
            # `tl.argmax` will select the maximum value.

        inv_q = tl.load(
            inv_q_ptr + req_idx * vocab_size + vocab_offset,
            mask=vocab_mask,
            other=0.0,
        )

        # Local tile reduction
        score = prob * inv_q
        local_max, local_id = tl.max(score, axis=0, return_indices=True)

        if local_max > max_val:
            max_val = local_max
            recovered_id = v + local_id

    tl.store(output_token_ids_ptr + token_idx, recovered_id)
