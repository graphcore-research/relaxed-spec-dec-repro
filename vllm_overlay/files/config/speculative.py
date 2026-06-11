# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ast
import copy
import math
from typing import TYPE_CHECKING, Any, Literal, get_args

from pydantic import Field, SkipValidation, model_validator
from typing_extensions import Self

from vllm.config import LoadConfig
from vllm.config.kernel import MoEBackend
from vllm.config.model import ModelConfig
from vllm.config.parallel import ParallelConfig
from vllm.config.utils import config
from vllm.logger import init_logger
from vllm.transformers_utils.config import get_hf_text_config
from vllm.utils.hashing import safe_hash
from vllm.utils.import_utils import LazyLoader, has_arctic_inference

if TYPE_CHECKING:
    from transformers import PretrainedConfig

    import vllm.model_executor.layers.quantization as me_quant
else:
    PretrainedConfig = Any

    me_quant = LazyLoader(
        "model_executor", globals(), "vllm.model_executor.layers.quantization"
    )

logger = init_logger(__name__)

MTPModelTypes = Literal[
    "deepseek_mtp",
    "mimo_mtp",
    "glm4_moe_mtp",
    "glm4_moe_lite_mtp",
    "glm_ocr_mtp",
    "ernie_mtp",
    "nemotron_h_mtp",
    "exaone_moe_mtp",
    "exaone4_5_mtp",
    "qwen3_next_mtp",
    "qwen3_5_mtp",
    "longcat_flash_mtp",
    "mtp",
    "pangu_ultra_moe_mtp",
    "step3p5_mtp",
    "hy_v3_mtp",
]
NgramGPUTypes = Literal["ngram_gpu"]
DFlashModelTypes = Literal["dflash"]
EagleModelTypes = Literal[
    "eagle", "eagle3", "extract_hidden_states", MTPModelTypes, DFlashModelTypes
]
SpeculativeMethod = Literal[
    "ngram",
    "medusa",
    "mlp_speculator",
    "draft_model",
    "suffix",
    EagleModelTypes,
    NgramGPUTypes,
]
RejectionSampleMethod = Literal["strict", "probabilistic", "synthetic"]
RelaxedTargetMethod = Literal[
    "none",
    "cactus",
    "ensemble",
    "rfsd",
    "fuzzy",
    "fsd",
    "spec_cascade_opt",
    "spec_cascade_tok3",
    "lossy_spec_decode_beta1",
    "scd_expert_toppk_gated",
    "scd_alpha",
]
FuzzyDivergence = Literal["kl", "js"]
RelaxedBonusTokenPolicy = Literal["target_p", "relaxed_T_qp"]
SpecCascadeOptGate = Literal["processed", "paper"]
SpecCascadeTok3TopSet = Literal["paper", "processed"]
DraftSamplingMethod = Literal["greedy", "stochastic"]


@config
class SpeculativeConfig:
    """Configuration for speculative decoding."""

    enforce_eager: bool | None = None
    """Override the default enforce_eager from model_config"""
    # General speculative decoding control
    num_speculative_tokens: int = Field(default=None, gt=0)  # type: ignore[assignment]
    """The number of speculative tokens, if provided. It will default to the
    number in the draft model config if present, otherwise, it is required."""
    model: str | None = None
    """The name of the draft model, eagle head, or additional weights, if
    provided."""
    method: SpeculativeMethod | None = None
    """The name of the speculative method to use. If users provide and set the
    `model` param, the speculative method type will be detected automatically
    if possible, if `model` param is not provided, the method name must be
    provided.

    If using `ngram` method, the related configuration `prompt_lookup_max` and
    `prompt_lookup_min` should be considered."""
    draft_tensor_parallel_size: int | None = Field(default=None, ge=1)
    """The degree of the tensor parallelism for the draft model. Can only be 1
    or the same as the target model's tensor parallel size."""
    tensor_parallel_size: int | None = None
    """Users should pass "draft_tensor_parallel_size". This parameter's purpose is to
    warn users when they mistakenly provide the wrong argument."""

    # Draft model configuration
    quantization: me_quant.QuantizationMethods | str | None = None
    """Quantization method that was used to quantize the draft model weights.
    If `None`, we assume the model weights are not quantized. Note that it only
    takes effect when using the draft model-based speculative method."""
    moe_backend: MoEBackend | None = None
    """MoE backend to use for the draft model. When `None`, the draft model
    inherits the target model's `--moe-backend` setting. Useful when the
    drafter and generator require different MoE kernels (e.g. quantized
    generator with unquantized drafter)."""
    max_model_len: int | None = Field(default=None, ge=1)
    """The maximum model length of the draft model. Used when testing the
    ability to skip speculation for some sequences."""
    revision: str | None = None
    """The specific model version to use for the draft model. It can be a
    branch name, a tag name, or a commit id. If unspecified, will use the
    default version."""
    code_revision: str | None = None
    """The specific revision to use for the draft model code on Hugging Face
    Hub. It can be a branch name, a tag name, or a commit id. If unspecified,
    will use the default version."""

    # Advanced control
    disable_padded_drafter_batch: bool = False
    """Disable input padding for speculative decoding. If set to True,
    speculative input batches can contain sequences of different lengths,
    which may only be supported by certain attention backends. This currently
    only affects the EAGLE method of speculation."""
    use_local_argmax_reduction: bool = False
    """Use vocab-parallel local argmax instead of all-gathering full logits
    for draft token generation. Reduces communication from O(vocab_size) to
    O(2 * tp_size) per token. Only applies to greedy draft selection in
    non-tree speculation."""
    entropy_monitoring: bool = False
    """Record per-request speculative entropy summaries to a sidecar."""
    entropy_monitoring_path: str | None = None
    """JSONL sidecar path used when entropy_monitoring is enabled."""

    # Ngram proposer configuration
    prompt_lookup_max: int | None = Field(default=None, ge=1)
    """Maximum size of ngram token window when using Ngram proposer, required
    when method is set to ngram."""
    prompt_lookup_min: int | None = Field(default=None, ge=1)
    """Minimum size of ngram token window when using Ngram proposer, if
    provided. Defaults to 1."""

    # Alternative drafting strategies
    speculative_token_tree: str | None = None
    """Specifies the tree structure for speculative token generation.
    """
    parallel_drafting: bool = False
    """Enable parallel drafting, where all speculative tokens are generated
    in parallel rather than sequentially. This can improve performance but
    requires the speculative model be trained to support parallel drafting.
    Only compatible with EAGLE and draft model methods."""

    # required configuration params passed from engine
    target_model_config: SkipValidation[ModelConfig] = None  # type: ignore
    """The configuration of the target model."""
    target_parallel_config: SkipValidation[ParallelConfig] = None  # type: ignore
    """The parallel configuration for the target model."""

    # params generated in the post-init stage
    draft_model_config: SkipValidation[ModelConfig] = None  # type: ignore
    """The configuration of the draft model initialized internal."""
    draft_parallel_config: SkipValidation[ParallelConfig] = None  # type: ignore
    """The parallel configuration for the draft model initialized internal."""

    # Suffix decoding configuration
    suffix_decoding_max_tree_depth: int = 24
    """The maximum depth of the suffix decoding global and prompt trees. The
    tree depth limits the sum of the prefix match and speculation lengths."""

    suffix_decoding_max_cached_requests: int = 10000
    """The maximum number of requests to cache in the global suffix tree. If
    exceeded, will trigger eviction in FIFO order. If set to 0, the global
    suffix tree is disabled and past responses are not cached (prompt trees
    are still used)."""

    suffix_decoding_max_spec_factor: float = 1.0
    """The maximum spec factor for suffix decoding. The spec factor controls
    speculation lengths based on the prefix match length: max_spec_tokens =
    max_spec_factor * prefix_match_length."""

    suffix_decoding_min_token_prob: float = 0.1
    """The minimum token probability for suffix decoding. Will only speculate
    tokens with estimated probability (based on frequency counts) greater than
    or equal to this value."""

    draft_load_config: LoadConfig | None = None
    """Load config for the draft model. If not specified, will use the load
    config from the target model."""

    rejection_sample_method: RejectionSampleMethod = "strict"
    """Whether to use strict (target and draft sampled tokens match exactly)
    or probabilistic rejection sampling. Both respect the target model
    distribution, but the latter yields a higher acceptance rate at the cost
    of more memory to cache draft logits."""

    draft_sampling_method: DraftSamplingMethod = "greedy"
    """How draft-model token ids are selected. vLLM's default draft-model path
    is greedy. Stochastic draft sampling caches draft probabilities so
    probabilistic/relaxed rejection samplers can use q correctly."""

    relaxed_target_method: RelaxedTargetMethod = "none"
    """Relaxed target distribution used by generalized speculative sampling.
    'cactus' applies the Cactus proposal-conditioned target distribution.
    'ensemble' applies the static verifier/drafter probability ensemble.
    'spec_cascade_opt' applies the SpecCascades [OPT] q-vs-p gate.
    'spec_cascade_tok3' applies the SpecCascades TokenV3 tokenwise target.
    'lossy_spec_decode_beta1' applies lossy SD with beta fixed to 1.
    'scd_expert_toppk_gated' and 'scd_alpha' apply dense SCD contrastive
    verifier distributions."""

    cactus_delta: float | None = None
    """Cactus KL-budget parameter. Required when relaxed_target_method is
    'cactus'. delta=0 recovers vanilla probabilistic rejection sampling."""

    verifier_weight: float | None = None
    """Verifier/target probability weight for relaxed_target_method='ensemble'.
    F(q, p) = verifier_weight * p + (1 - verifier_weight) * q."""

    fuzzy_divergence: FuzzyDivergence | None = None
    """Divergence gate for relaxed_target_method='rfsd' or 'fsd'. 'kl'
    means KL(p || q), where p is target/verifier and q is draft. The field name
    is retained for legacy vLLM overlay compatibility."""

    fuzzy_threshold: float | None = None
    """Divergence threshold for relaxed_target_method='rfsd' or 'fsd'.
    The field name is retained for legacy vLLM overlay compatibility."""

    spec_cascade_alpha: float | None = None
    """SpecCascades [OPT]/Tok3 confidence/slack parameter.

    Required when relaxed_target_method is 'spec_cascade_opt' or
    'spec_cascade_tok3'. OPT has no sign restriction; negative alpha makes the
    q-vs-p gate more target-favoring.
    Tok3 requires 0 <= spec_cascade_alpha <= 1."""

    spec_cascade_opt_gate: SpecCascadeOptGate = "processed"
    """Gate probabilities used by relaxed_target_method='spec_cascade_opt'.
    'processed' computes the whole gate from post-temperature/top-p/top-k
    probabilities. 'paper' uses pre-temperature/top-p/top-k max probabilities
    and processed probabilities for TV, acceptance, and residual sampling."""

    spec_cascade_tok3_top_set: SpecCascadeTok3TopSet = "paper"
    """Top-set logits used by relaxed_target_method='spec_cascade_tok3'.
    'paper' computes T_alpha before temperature/top-p/top-k. 'processed'
    computes T_alpha from the actual target sampling logits."""

    lossy_alpha: float | None = None
    """Lossy SD acceptance slack. Required when
    relaxed_target_method='lossy_spec_decode_beta1'; must satisfy
    0 <= lossy_alpha < 1."""

    scd_beta: float | None = None
    """SCD improved contrastive-decoding beta. Required when
    relaxed_target_method is 'scd_expert_toppk_gated' or 'scd_alpha';
    must satisfy scd_beta >= 0."""

    scd_temperature: float | None = None
    """Optional SCD contrastive softmax temperature. When unset, SCD uses the
    per-request decode temperature; explicit values must be > 0."""

    scd_alpha: float | None = None
    """SCD plausibility-gate alpha. Required only when
    relaxed_target_method='scd_alpha'; must satisfy 0 < scd_alpha <= 1."""

    relaxed_bonus_token_policy: RelaxedBonusTokenPolicy = "target_p"
    """Bonus-token distribution after all draft tokens are accepted. 'target_p'
    preserves vLLM's target-only bonus token. 'relaxed_T_qp' runs one
    conditional post-accept drafter pass and samples from the relaxed T(q, p)
    bonus distribution."""

    synthetic_acceptance_rates: list[float] | None = None
    """Per-position *unconditional* acceptance rates for synthetic rejection
    sampling. Position i's entry is the marginal probability that the first
    i+1 draft tokens are all accepted; the list must have length
    num_speculative_tokens, each entry in [0, 1], and be monotonically
    non-increasing. Only valid when rejection_sample_method is 'synthetic'.
    Mutually exclusive with synthetic_acceptance_length."""

    synthetic_acceptance_length: float | None = None
    """Target mean acceptance length for synthetic rejection sampling, in
    [1, num_speculative_tokens + 1]. Resolved internally to
    synthetic_acceptance_rates. Only valid when rejection_sample_method is 'synthetic'.
    Mutually exclusive with synthetic_acceptance_rates."""

    @staticmethod
    def _acceptance_length_to_rates(length: float, n: int) -> list[float]:
        """Mean acceptance length to unconditional per-position rates, using
        the minimum-variance schedule."""
        num_drafts = length - 1  # expected number of accepted draft tokens
        num_full = int(num_drafts)
        return (
            [1.0] * num_full + [num_drafts - num_full] + [0.0] * (n - num_full - 1)
        )[:n]

    @staticmethod
    def _resolve_synthetic_acceptance_rates(
        n: int,
        rates: list[float] | None,
        length: float | None,
    ) -> list[float]:
        """Return per-position unconditional acceptance rates from exactly one
        of `rates` or `length` (validates range, length, and monotonicity)."""
        if (rates is None) == (length is None):
            raise ValueError(
                "rejection_sample_method='synthetic' requires exactly one of "
                "synthetic_acceptance_rates or synthetic_acceptance_length."
            )
        if rates is not None:
            if len(rates) != n:
                raise ValueError(
                    f"synthetic_acceptance_rates must have length {n}, got {rates}."
                )
            if not all(0.0 <= r <= 1.0 for r in rates):
                raise ValueError(
                    f"synthetic_acceptance_rates entries must be in [0, 1], "
                    f"got {rates}."
                )
            if any(rates[i] > rates[i - 1] for i in range(1, n)):
                raise ValueError(
                    f"synthetic_acceptance_rates must be non-increasing, got {rates}."
                )
            return list(rates)
        assert length is not None
        if not 1.0 <= length <= float(n + 1):
            raise ValueError(
                f"synthetic_acceptance_length must be in [1, {n + 1}], got {length}."
            )
        return SpeculativeConfig._acceptance_length_to_rates(length, n)

    def compute_hash(self) -> str:
        """
        WARNING: Whenever a new field is added to this config,
        ensure that it is included in the factors list if
        it affects the computation graph.

        Provide a hash that uniquely identifies all the configs
        that affect the structure of the computation
        graph from input ids/embeddings to the final hidden states,
        excluding anything before input ids/embeddings and after
        the final hidden states.
        """
        factors: list[Any] = []
        # Eagle3 and extract_hidden_states affect the computation graph because
        # they return intermediate hidden states in addition to the final hidden state.
        uses_aux_hidden_states = self.method in (
            "eagle3",
            "extract_hidden_states",
            "dflash",
        )
        factors.append(uses_aux_hidden_states)

        # The specific layers used also affect the computation graph
        if uses_aux_hidden_states and self.draft_model_config is not None:
            layer_ids = getattr(
                self.draft_model_config.hf_config,
                "eagle_aux_hidden_state_layer_ids",
                None,
            )
            if layer_ids is not None:
                # Convert to tuple to make it hashable
                factors.append(tuple(layer_ids))

        hash_str = safe_hash(str(factors).encode(), usedforsecurity=False).hexdigest()
        return hash_str

    @staticmethod
    def hf_config_override(hf_config: PretrainedConfig) -> PretrainedConfig:
        initial_architecture = hf_config.architectures[0]
        if hf_config.model_type in (
            "deepseek_v3",
            "deepseek_v32",
            "glm_moe_dsa",
        ):
            hf_config.model_type = "deepseek_mtp"
        if hf_config.model_type == "deepseek_mtp":
            n_predict = getattr(hf_config, "num_nextn_predict_layers", None)
            hf_config.update(
                {"n_predict": n_predict, "architectures": ["DeepSeekMTPModel"]}
            )
        if hf_config.model_type == "deepseek_v4":
            hf_config.model_type = "deepseek_mtp"
            n_predict = getattr(hf_config, "num_nextn_predict_layers", None)
            hf_config.update(
                {"n_predict": n_predict, "architectures": ["DeepSeekV4MTPModel"]}
            )
        if hf_config.model_type in ("pangu_ultra_moe"):
            hf_config.model_type = "pangu_ultra_moe_mtp"
        if hf_config.model_type == "pangu_ultra_moe_mtp":
            n_predict = getattr(hf_config, "num_nextn_predict_layers", None)
            hf_config.update(
                {"n_predict": n_predict, "architectures": ["OpenPanguMTPModel"]}
            )

        if hf_config.architectures[0] == "MiMoForCausalLM":
            hf_config.model_type = "mimo_mtp"
            n_predict = getattr(hf_config, "num_nextn_predict_layers", None)
            hf_config.update(
                {
                    "num_hidden_layers": 0,
                    "n_predict": n_predict,
                    "architectures": ["MiMoMTPModel"],
                }
            )

        if hf_config.architectures[0] == "Glm4MoeForCausalLM":
            hf_config.model_type = "glm4_moe_mtp"
            n_predict = getattr(hf_config, "num_nextn_predict_layers", None)
            hf_config.update(
                {
                    "n_predict": n_predict,
                    "architectures": ["Glm4MoeMTPModel"],
                }
            )

        if hf_config.architectures[0] == "Glm4MoeLiteForCausalLM":
            hf_config.model_type = "glm4_moe_lite_mtp"
            n_predict = getattr(hf_config, "num_nextn_predict_layers", None)
            hf_config.update(
                {
                    "num_hidden_layers": 0,
                    "n_predict": n_predict,
                    "architectures": ["Glm4MoeLiteMTPModel"],
                }
            )

        if hf_config.architectures[0] == "GlmOcrForConditionalGeneration":
            hf_config.model_type = "glm_ocr_mtp"
            n_predict = getattr(hf_config, "num_nextn_predict_layers", None)
            hf_config.update(
                {
                    "num_hidden_layers": 0,
                    "n_predict": n_predict,
                    "architectures": ["GlmOcrMTPModel"],
                }
            )

        if hf_config.model_type == "ernie4_5_moe":
            hf_config.model_type = "ernie_mtp"
        if hf_config.model_type == "ernie_mtp":
            n_predict = getattr(hf_config, "num_nextn_predict_layers", None)
            hf_config.update(
                {"n_predict": n_predict, "architectures": ["ErnieMTPModel"]}
            )

        if hf_config.architectures[0] == "NemotronH_Super_Omni_Reasoning_V3":
            # Promote VLM's text_config so MTP detection below fires correctly
            hf_config = hf_config.text_config

        if (
            hf_config.model_type in {"nemotron_h", "nemotron_h_puzzle"}
            and hasattr(hf_config, "num_nextn_predict_layers")
            and hf_config.num_nextn_predict_layers > 0
        ):
            # Check if this is an MTP variant
            hf_config.model_type = "nemotron_h_mtp"
        if hf_config.model_type == "nemotron_h_mtp":
            n_predict = getattr(hf_config, "num_nextn_predict_layers", 1)
            hf_config.update(
                {"n_predict": n_predict, "architectures": ["NemotronHMTPModel"]}
            )

        if hf_config.model_type == "qwen3_next":
            hf_config.model_type = "qwen3_next_mtp"
        if hf_config.model_type == "qwen3_next_mtp":
            n_predict = getattr(hf_config, "num_nextn_predict_layers", None)
            hf_config.update(
                {"n_predict": n_predict, "architectures": ["Qwen3NextMTP"]}
            )

        if hf_config.model_type == "exaone_moe":
            hf_config.model_type = "exaone_moe_mtp"
        if hf_config.model_type == "exaone_moe_mtp":
            n_predict = getattr(hf_config, "num_nextn_predict_layers", None)
            hf_config.update(
                {"n_predict": n_predict, "architectures": ["ExaoneMoeMTP"]}
            )
        if "exaone4_5" in hf_config.model_type:
            hf_config.model_type = "exaone4_5_mtp"
        if hf_config.model_type == "exaone4_5_mtp":
            n_predict = getattr(hf_config, "num_nextn_predict_layers", None)
            hf_config.update(
                {"n_predict": n_predict, "architectures": ["Exaone4_5_MTP"]}
            )
        if hf_config.model_type in ("qwen3_5", "qwen3_5_moe"):
            is_moe = hf_config.model_type == "qwen3_5_moe"
            hf_config.model_type = "qwen3_5_mtp"
            n_predict = getattr(hf_config, "mtp_num_hidden_layers", None)
            hf_config.update(
                {
                    "n_predict": n_predict,
                    "architectures": ["Qwen3_5MoeMTP" if is_moe else "Qwen3_5MTP"],
                }
            )
        if hf_config.model_type == "longcat_flash":
            hf_config.model_type = "longcat_flash_mtp"
            n_predict = getattr(hf_config, "num_nextn_predict_layers", 1)
            hf_config.update(
                {"n_predict": n_predict, "architectures": ["LongCatFlashMTPModel"]}
            )

        if hf_config.model_type == "step3p5":
            hf_config.model_type = "step3p5_mtp"
            n_predict = getattr(hf_config, "num_nextn_predict_layers", 1)
            hf_config.update({"n_predict": n_predict, "architectures": ["Step3p5MTP"]})

        if initial_architecture == "MistralLarge3ForCausalLM":
            hf_config.update({"architectures": ["EagleMistralLarge3ForCausalLM"]})

        if hf_config.model_type == "hy_v3":
            hf_config.model_type = "hy_v3_mtp"
            n_predict = getattr(hf_config, "num_nextn_predict_layers", None)
            hf_config.update(
                {"n_predict": n_predict, "architectures": ["HYV3MTPModel"]}
            )

        return hf_config

    def __post_init__(self):
        # Note: "method" is a new parameter that helps to extend the
        # configuration of non-model-based proposers, and the "model" parameter
        # will be used to set the draft model, eagle head, or additional weight
        # when needed. If users do not specify "method", the speculative method
        # will be detected automatically if possible. If the speculative method
        # can not be detected, it will be considered as the "draft_model" by
        # default.

        # infer method from user args
        if self.method is None:
            if self.model in ("ngram", "[ngram]"):
                self.method = "ngram"
            else:
                self.method = "draft_model"
        if self.relaxed_target_method == "fuzzy":
            self.relaxed_target_method = "rfsd"

        if self.method in get_args(MTPModelTypes) and self.method != "mtp":
            logger.warning(
                "method `%s` is deprecated and replaced with mtp.", self.method
            )
            self.method = "mtp"

        if self.model is None and self.num_speculative_tokens is not None:
            if self.method == "mtp":
                if self.target_model_config is None:
                    raise ValueError("target_model_config must be present for mtp")
                if self.target_model_config.hf_text_config.model_type == "deepseek_v32":
                    # FIXME(luccafong): cudagraph with v32 MTP is not supported,
                    # remove this when the issue is fixed.
                    self.enforce_eager = True
                # use the draft model from the same model:
                self.model = self.target_model_config.model
                # Align the quantization of draft model for cases such as
                # --quantization fp8 with a bf16 checkpoint.
                if not self.quantization:
                    self.quantization = self.target_model_config.quantization
            elif self.method in ("ngram", "[ngram]"):
                self.model = "ngram"
            elif self.method == "ngram_gpu":
                self.model = "ngram_gpu"
            elif self.method == "suffix":
                self.model = "suffix"
            elif self.method == "extract_hidden_states":
                self.model = "extract_hidden_states"
            else:
                raise ValueError(
                    "num_speculative_tokens was provided but without speculative model."
                )

        if self.method in ("ngram", "[ngram]"):
            self.method = "ngram"

        if self.method in ("ngram", "ngram_gpu"):
            # Set default values if not provided
            if self.prompt_lookup_min is None and self.prompt_lookup_max is None:
                # TODO(woosuk): Tune these values. They are arbitrarily chosen.
                self.prompt_lookup_min = 5
                self.prompt_lookup_max = 5
            elif self.prompt_lookup_min is None:
                if self.prompt_lookup_max is None:
                    raise ValueError(
                        "Either prompt_lookup_max or prompt_lookup_min must be "
                        "provided when using the ngram method."
                    )
                self.prompt_lookup_min = self.prompt_lookup_max
            elif self.prompt_lookup_max is None:
                if self.prompt_lookup_min is None:
                    raise ValueError(
                        "Either prompt_lookup_max or prompt_lookup_min must be "
                        "provided when using the ngram method."
                    )
                self.prompt_lookup_max = self.prompt_lookup_min

            # Validate values
            if self.prompt_lookup_min > self.prompt_lookup_max:
                raise ValueError(
                    f"prompt_lookup_min={self.prompt_lookup_min} must "
                    f"be <= prompt_lookup_max={self.prompt_lookup_max}"
                )

            # TODO: current we still need extract vocab_size from target model
            # config, in future, we may try refactor it out, and set
            # draft related config as None here.
            self.draft_model_config = self.target_model_config
            self.draft_parallel_config = self.target_parallel_config
        elif self.method == "suffix":
            self._validate_suffix_decoding()
        elif self.method == "extract_hidden_states":
            from vllm.transformers_utils.configs.extract_hidden_states import (
                ExtractHiddenStatesConfig,
            )

            # ExtractHiddenStatesModel is instantiated manually in load_model()
            # We just need to store the target model config for KV cache shape info
            self.model = "extract_hidden_states"
            self.prompt_lookup_max = 0
            self.prompt_lookup_min = 0

            if hasattr(self.draft_model_config, "hf_config"):
                hf_config = self.draft_model_config.hf_config.to_dict()
            elif (
                isinstance(self.draft_model_config, dict)
                and "hf_config" in self.draft_model_config
            ):
                hf_config = self.draft_model_config["hf_config"]
            else:
                hf_config = {}

            self.draft_model_config = copy.copy(self.target_model_config)
            self.draft_model_config.hf_config = ExtractHiddenStatesConfig(
                self.draft_model_config.hf_config, **hf_config
            )
            self.update_arch_()
            self.draft_parallel_config = self.target_parallel_config

        else:
            self.prompt_lookup_max = 0
            self.prompt_lookup_min = 0

            if self.model is not None:
                self.draft_model_config = ModelConfig(
                    model=self.model,
                    runner="draft",
                    tokenizer=self.target_model_config.tokenizer,
                    tokenizer_mode=self.target_model_config.tokenizer_mode,
                    trust_remote_code=self.target_model_config.trust_remote_code,
                    allowed_local_media_path=self.target_model_config.allowed_local_media_path,
                    allowed_media_domains=self.target_model_config.allowed_media_domains,
                    dtype=self.target_model_config.dtype,
                    seed=self.target_model_config.seed,
                    revision=self.revision,
                    code_revision=self.code_revision,
                    tokenizer_revision=self.target_model_config.tokenizer_revision,
                    spec_target_max_model_len=self.target_model_config.max_model_len,
                    quantization=self.quantization,
                    enforce_eager=self.target_model_config.enforce_eager,
                    max_logprobs=self.target_model_config.max_logprobs,
                    hf_overrides=SpeculativeConfig.hf_config_override,
                    config_format=self.target_model_config.config_format,
                )

                # Automatically detect the method
                if self.method in ("eagle", "eagle3", "dflash"):
                    pass
                # examples:
                # yuhuili/EAGLE-LLaMA3-Instruct-8B
                # yuhuili/EAGLE3-LLaMA3.1-Instruct-8B
                # AngelSlim/Qwen3-8B_eagle3
                elif "eagle-" in self.draft_model_config.model.lower():
                    self.method = "eagle"
                elif "eagle3" in self.draft_model_config.model.lower():
                    self.method = "eagle3"
                elif "dflash" in self.draft_model_config.model.lower():
                    self.method = "dflash"
                elif self.draft_model_config.hf_config.model_type == "medusa":
                    self.method = "medusa"
                elif self.draft_model_config.hf_config.model_type == "mlp_speculator":
                    self.method = "mlp_speculator"
                elif self.draft_model_config.hf_config.model_type in get_args(
                    MTPModelTypes
                ):
                    self.method = "mtp"
                    if self.num_speculative_tokens > 1:
                        logger.warning(
                            "Enabling num_speculative_tokens > 1 will run "
                            "multiple times of forward on same MTP layer"
                            ",which may result in lower acceptance rate"
                        )
                elif self.draft_model_config.hf_config.model_type in (
                    "longcat_flash_mtp"
                ):
                    self.method = "longcat_flash_mtp"
                    if self.num_speculative_tokens > 1:
                        logger.warning(
                            "LongCat MTP models only have "
                            "one layer. Might need some code changes "
                            "to support multiple layers."
                        )
                elif self.method == "draft_model":
                    pass
                else:
                    raise NotImplementedError(
                        f"Unsupported speculative method: '{self.method}'"
                    )

                # Replace hf_config for EAGLE draft_model
                if self.method in ("eagle", "eagle3", "dflash"):
                    from vllm.transformers_utils.configs.eagle import EAGLEConfig
                    from vllm.transformers_utils.configs.speculators import (
                        SpeculatorsConfig,
                    )

                    if isinstance(
                        self.draft_model_config.hf_config,
                        (EAGLEConfig, SpeculatorsConfig),
                    ):
                        pass
                    else:
                        eagle_config = EAGLEConfig(
                            self.draft_model_config.hf_config,
                            method=self.method,
                            model_type="eagle",
                        )
                        self.draft_model_config.hf_config = eagle_config
                        self.update_arch_()

                if self.method == "dflash":
                    self.parallel_drafting = True

                if self.num_speculative_tokens is not None and hasattr(
                    self.draft_model_config.hf_config, "num_lookahead_tokens"
                ):
                    self.draft_model_config.hf_config.num_lookahead_tokens = (
                        self.num_speculative_tokens
                    )

                n_predict = getattr(
                    self.draft_model_config.hf_config, "n_predict", None
                )
                if n_predict is not None:
                    if self.num_speculative_tokens is None:
                        # Default to max value defined in draft model config.
                        self.num_speculative_tokens = n_predict
                    elif (
                        self.num_speculative_tokens > n_predict
                        and self.num_speculative_tokens % n_predict != 0
                    ):
                        # Ensure divisibility for MTP module reuse.
                        raise ValueError(
                            f"num_speculative_tokens:{self.num_speculative_tokens}"
                            f" must be divisible by {n_predict=}"
                        )

                if self.speculative_token_tree is None:
                    if self.num_speculative_tokens is None:
                        raise ValueError(
                            "A speculative model was provided, but neither "
                            "`speculative_token_tree` nor `num_speculative_tokens` "
                            "was provided"
                        )

                    # Generate chain of tokens.
                    self.speculative_token_tree = str(
                        [(i + 1) * (0,) for i in range(self.num_speculative_tokens)]
                    )
                else:
                    # Sort the token tree breadth-first.
                    tree_choices = ast.literal_eval(self.speculative_token_tree)
                    self.speculative_token_tree = str(
                        sorted(tree_choices, key=lambda t: (len(t), t))
                    )

                self.draft_tensor_parallel_size = (
                    SpeculativeConfig._verify_and_get_draft_tp(
                        self.target_parallel_config,
                        self.draft_tensor_parallel_size,
                        self.draft_model_config.hf_config,
                    )
                )

                self.draft_model_config.max_model_len = (
                    SpeculativeConfig._maybe_override_draft_max_model_len(
                        self.max_model_len,
                        self.draft_model_config.max_model_len,
                        self.target_model_config.max_model_len,
                    )
                )

                self.draft_parallel_config = (
                    SpeculativeConfig.create_draft_parallel_config(
                        self.target_parallel_config, self.draft_tensor_parallel_size
                    )
                )
        return self

    def _validate_suffix_decoding(self):
        if not has_arctic_inference():
            raise ImportError(
                "Arctic Inference is required for suffix decoding. "
                "Install via `pip install arctic-inference==0.1.1`."
            )
        if self.num_speculative_tokens is None:
            # Suffix decoding decides the actual number of speculative tokens
            # dynamically and treats num_speculative_tokens as a maximum limit.
            self.num_speculative_tokens = self.suffix_decoding_max_tree_depth
            logger.warning(
                "Defaulted num_speculative_tokens to %s for suffix decoding.",
                self.num_speculative_tokens,
            )
        # Validate values
        if self.suffix_decoding_max_tree_depth < 1:
            raise ValueError(
                f"suffix_decoding_max_tree_depth="
                f"{self.suffix_decoding_max_tree_depth} must be >= 1"
            )
        if self.suffix_decoding_max_cached_requests < 0:
            raise ValueError(
                f"suffix_decoding_max_cached_requests="
                f"{self.suffix_decoding_max_cached_requests} must be >= 0"
            )
        if self.suffix_decoding_max_spec_factor < 0:
            raise ValueError(
                f"suffix_decoding_max_spec_factor="
                f"{self.suffix_decoding_max_spec_factor} must be >= 0"
            )
        if not 0 <= self.suffix_decoding_min_token_prob <= 1:
            raise ValueError(
                f"suffix_decoding_min_token_prob="
                f"{self.suffix_decoding_min_token_prob} must be in [0, 1]"
            )

    @staticmethod
    def _maybe_override_draft_max_model_len(
        speculative_max_model_len: int | None,
        draft_max_model_len: int,
        target_max_model_len: int,
    ) -> int:
        """Determine the max sequence len for the draft model. This is usually
        the draft_max_model_len, but may be the target_max_model_len if it is
        less than the draft_max_model_len, or may be speculative_max_model_len
        if it is specified.

        This is necessary so that sequences do not exceed the capacity of the
        draft model or the target model.

        speculative_max_model_len is mainly used for testing that sequences can
        skip speculation.
        """

        if speculative_max_model_len is not None:
            if speculative_max_model_len > draft_max_model_len:
                raise ValueError(
                    f"{speculative_max_model_len=} cannot be "
                    f"larger than {draft_max_model_len=}"
                )

            if speculative_max_model_len > target_max_model_len:
                raise ValueError(
                    f"{speculative_max_model_len=} cannot be "
                    f"larger than {target_max_model_len=}"
                )

            return speculative_max_model_len

        return min(
            draft_max_model_len,
            target_max_model_len,
        )

    @staticmethod
    def _verify_and_get_draft_tp(
        target_parallel_config: ParallelConfig,
        speculative_draft_tensor_parallel_size: int | None,
        draft_hf_config: PretrainedConfig,
    ) -> int:
        """
        Verifies and adjusts the tensor parallel size for a draft model
        specified using speculative_draft_tensor_parallel_size.
        """
        # If speculative_draft_tensor_parallel_size is unset then set it
        # appropriately else verify that it is set correctly.
        if speculative_draft_tensor_parallel_size is None:
            if draft_hf_config.model_type == "mlp_speculator":
                speculative_draft_tensor_parallel_size = 1
                if target_parallel_config.tensor_parallel_size > 1:
                    logger.warning(
                        "%s cannot currently be run with tp>1; "
                        "setting speculative_draft_tensor_parallel_size=1",
                        draft_hf_config.model_type,
                    )
            else:
                speculative_draft_tensor_parallel_size = (
                    target_parallel_config.tensor_parallel_size
                )
        elif speculative_draft_tensor_parallel_size not in (
            1,
            target_parallel_config.tensor_parallel_size,
        ):
            raise ValueError(
                f"{speculative_draft_tensor_parallel_size=} cannot be "
                f"other value than 1 or target model tensor_parallel_size"
            )
        return speculative_draft_tensor_parallel_size

    def update_arch_(self):
        """
        EagleConfig and ExtractHiddenStatesConfig update architectures, so update all
        architectures-related fields in self.draft_model_config
        """
        self.draft_model_config.hf_text_config = get_hf_text_config(
            self.draft_model_config.hf_config
        )
        self.draft_model_config.model_arch_config = (
            self.draft_model_config.get_model_arch_config()
        )
        model_info, arch = self.draft_model_config.registry.inspect_model_cls(
            self.draft_model_config.architectures,
            self.draft_model_config,
        )
        self.draft_model_config._model_info = model_info
        self.draft_model_config._architecture = arch

    @staticmethod
    def create_draft_parallel_config(
        target_parallel_config: ParallelConfig,
        speculative_draft_tensor_parallel_size: int,
    ) -> ParallelConfig:
        """Create a parallel config for use by the draft worker.

        This is mostly a copy of the target parallel config, except the tp_size.
        """
        draft_parallel_config = ParallelConfig(
            pipeline_parallel_size=target_parallel_config.pipeline_parallel_size,
            tensor_parallel_size=speculative_draft_tensor_parallel_size,
            distributed_executor_backend=target_parallel_config.distributed_executor_backend,
            max_parallel_loading_workers=target_parallel_config.max_parallel_loading_workers,
            disable_custom_all_reduce=target_parallel_config.disable_custom_all_reduce,
            ray_workers_use_nsight=target_parallel_config.ray_workers_use_nsight,
            placement_group=target_parallel_config.placement_group,
        )

        return draft_parallel_config

    @model_validator(mode="after")
    def _verify_args(self) -> Self:
        if self.tensor_parallel_size is not None:
            raise ValueError(
                "'tensor_parallel_size' is not a valid argument in the "
                "speculative_config. Please pass 'draft_tensor_parallel_size' instead."
            )

        if self.num_speculative_tokens is None:
            raise ValueError(
                "num_speculative_tokens must be provided with "
                "speculative model unless the draft model config contains an "
                "n_predict parameter."
            )

        if self.num_speculative_tokens <= 0:
            raise ValueError(
                "Expected num_speculative_tokens to be greater "
                f"than zero ({self.num_speculative_tokens})."
            )
        if self.entropy_monitoring and not self.entropy_monitoring_path:
            raise ValueError("entropy_monitoring requires entropy_monitoring_path.")

        if self.rejection_sample_method == "synthetic":
            # Consolidate to per-position rates
            self.synthetic_acceptance_rates = self._resolve_synthetic_acceptance_rates(
                self.num_speculative_tokens,
                self.synthetic_acceptance_rates,
                self.synthetic_acceptance_length,
            )
            self.synthetic_acceptance_length = None
        elif (
            self.synthetic_acceptance_rates is not None
            or self.synthetic_acceptance_length is not None
        ):
            raise ValueError(
                "synthetic_acceptance_rates / synthetic_acceptance_length "
                "are only valid with rejection_sample_method='synthetic'."
            )

        if (
            self.relaxed_target_method != "spec_cascade_tok3"
            and self.spec_cascade_tok3_top_set != "paper"
        ):
            raise ValueError(
                "spec_cascade_tok3_top_set is only valid with "
                "relaxed_target_method='spec_cascade_tok3'."
            )
        if (
            self.relaxed_target_method != "spec_cascade_opt"
            and self.spec_cascade_opt_gate != "processed"
        ):
            raise ValueError(
                "spec_cascade_opt_gate is only valid with "
                "relaxed_target_method='spec_cascade_opt'."
            )

        scd_methods = ("scd_expert_toppk_gated", "scd_alpha")
        relaxed_target_methods = ("draft_model", "mtp")
        if self.relaxed_target_method not in scd_methods and (
            self.scd_beta is not None
            or self.scd_temperature is not None
            or self.scd_alpha is not None
        ):
            raise ValueError(
                "scd_beta / scd_temperature / scd_alpha are only valid with "
                "relaxed_target_method='scd_expert_toppk_gated' or 'scd_alpha'."
            )
        if self.relaxed_target_method == "cactus":
            if self.method not in relaxed_target_methods:
                raise ValueError(
                    "relaxed_target_method='cactus' is currently supported only "
                    "with method='draft_model' or method='mtp'."
                )
            if self.cactus_delta is None or self.cactus_delta < 0:
                raise ValueError(
                    "relaxed_target_method='cactus' requires cactus_delta >= 0."
                )
            if self.draft_sampling_method != "stochastic":
                raise ValueError(
                    "relaxed_target_method='cactus' requires "
                    "draft_sampling_method='stochastic'."
                )
            if self.rejection_sample_method != "probabilistic":
                raise ValueError(
                    "relaxed_target_method='cactus' requires "
                    "rejection_sample_method='probabilistic'."
                )
            if self.verifier_weight is not None:
                raise ValueError(
                    "verifier_weight is only valid with "
                    "relaxed_target_method='ensemble'."
                )
            if self.fuzzy_divergence is not None or self.fuzzy_threshold is not None:
                raise ValueError(
                    "fuzzy_divergence / fuzzy_threshold are only valid with "
                    "relaxed_target_method='rfsd' or 'fsd'."
                )
            if self.spec_cascade_alpha is not None:
                raise ValueError(
                    "spec_cascade_alpha is only valid with "
                    "relaxed_target_method='spec_cascade_opt' or "
                    "'spec_cascade_tok3'."
                )
            if self.lossy_alpha is not None:
                raise ValueError(
                    "lossy_alpha is only valid with "
                    "relaxed_target_method='lossy_spec_decode_beta1'."
                )
        elif self.relaxed_target_method == "ensemble":
            if self.method not in relaxed_target_methods:
                raise ValueError(
                    "relaxed_target_method='ensemble' is currently supported only "
                    "with method='draft_model' or method='mtp'."
                )
            if (
                self.verifier_weight is None
                or not 0.0 <= self.verifier_weight <= 1.0
            ):
                raise ValueError(
                    "relaxed_target_method='ensemble' requires "
                    "verifier_weight in [0, 1]."
                )
            if self.draft_sampling_method != "stochastic":
                raise ValueError(
                    "relaxed_target_method='ensemble' requires "
                    "draft_sampling_method='stochastic'."
                )
            if self.rejection_sample_method != "probabilistic":
                raise ValueError(
                    "relaxed_target_method='ensemble' requires "
                    "rejection_sample_method='probabilistic'."
                )
            if self.cactus_delta is not None:
                raise ValueError(
                    "cactus_delta is only valid with relaxed_target_method='cactus'."
                )
            if self.fuzzy_divergence is not None or self.fuzzy_threshold is not None:
                raise ValueError(
                    "fuzzy_divergence / fuzzy_threshold are only valid with "
                    "relaxed_target_method='rfsd' or 'fsd'."
                )
            if self.spec_cascade_alpha is not None:
                raise ValueError(
                    "spec_cascade_alpha is only valid with "
                    "relaxed_target_method='spec_cascade_opt' or "
                    "'spec_cascade_tok3'."
                )
            if self.lossy_alpha is not None:
                raise ValueError(
                    "lossy_alpha is only valid with "
                    "relaxed_target_method='lossy_spec_decode_beta1'."
                )
        elif self.relaxed_target_method in ("rfsd", "fsd"):
            if self.method not in relaxed_target_methods:
                raise ValueError(
                    f"relaxed_target_method={self.relaxed_target_method!r} is "
                    "currently supported only with method='draft_model' or "
                    "method='mtp'."
                )
            if self.fuzzy_divergence is None:
                raise ValueError(
                    f"relaxed_target_method={self.relaxed_target_method!r} requires "
                    "fuzzy_divergence in {'kl', 'js'}."
                )
            if self.fuzzy_threshold is None or self.fuzzy_threshold < 0:
                raise ValueError(
                    f"relaxed_target_method={self.relaxed_target_method!r} requires "
                    "fuzzy_threshold >= 0."
                )
            if self.draft_sampling_method != "stochastic":
                raise ValueError(
                    f"relaxed_target_method={self.relaxed_target_method!r} requires "
                    "draft_sampling_method='stochastic'."
                )
            if self.rejection_sample_method != "probabilistic":
                raise ValueError(
                    f"relaxed_target_method={self.relaxed_target_method!r} requires "
                    "rejection_sample_method='probabilistic'."
                )
            if self.cactus_delta is not None:
                raise ValueError(
                    "cactus_delta is only valid with relaxed_target_method='cactus'."
                )
            if self.verifier_weight is not None:
                raise ValueError(
                    "verifier_weight is only valid with "
                    "relaxed_target_method='ensemble'."
                )
            if self.spec_cascade_alpha is not None:
                raise ValueError(
                    "spec_cascade_alpha is only valid with "
                    "relaxed_target_method='spec_cascade_opt' or "
                    "'spec_cascade_tok3'."
                )
            if self.lossy_alpha is not None:
                raise ValueError(
                    "lossy_alpha is only valid with "
                    "relaxed_target_method='lossy_spec_decode_beta1'."
                )
        elif self.relaxed_target_method == "spec_cascade_opt":
            if self.method not in relaxed_target_methods:
                raise ValueError(
                    "relaxed_target_method='spec_cascade_opt' is currently "
                    "supported only with method='draft_model' or method='mtp'."
                )
            if self.spec_cascade_alpha is None:
                raise ValueError(
                    "relaxed_target_method='spec_cascade_opt' requires "
                    "spec_cascade_alpha."
                )
            if self.spec_cascade_opt_gate not in ("processed", "paper"):
                raise ValueError(
                    "relaxed_target_method='spec_cascade_opt' requires "
                    "spec_cascade_opt_gate in {'processed', 'paper'}."
                )
            if self.draft_sampling_method != "stochastic":
                raise ValueError(
                    "relaxed_target_method='spec_cascade_opt' requires "
                    "draft_sampling_method='stochastic'."
                )
            if self.rejection_sample_method != "probabilistic":
                raise ValueError(
                    "relaxed_target_method='spec_cascade_opt' requires "
                    "rejection_sample_method='probabilistic'."
                )
            if self.cactus_delta is not None:
                raise ValueError(
                    "cactus_delta is only valid with relaxed_target_method='cactus'."
                )
            if self.verifier_weight is not None:
                raise ValueError(
                    "verifier_weight is only valid with "
                    "relaxed_target_method='ensemble'."
                )
            if self.fuzzy_divergence is not None or self.fuzzy_threshold is not None:
                raise ValueError(
                    "fuzzy_divergence / fuzzy_threshold are only valid with "
                    "relaxed_target_method='rfsd' or 'fsd'."
                )
            if self.lossy_alpha is not None:
                raise ValueError(
                    "lossy_alpha is only valid with "
                    "relaxed_target_method='lossy_spec_decode_beta1'."
                )
            if self.spec_cascade_tok3_top_set != "paper":
                raise ValueError(
                    "spec_cascade_tok3_top_set is only valid with "
                    "relaxed_target_method='spec_cascade_tok3'."
                )
        elif self.relaxed_target_method == "spec_cascade_tok3":
            if self.method not in relaxed_target_methods:
                raise ValueError(
                    "relaxed_target_method='spec_cascade_tok3' is currently "
                    "supported only with method='draft_model' or method='mtp'."
                )
            if (
                self.spec_cascade_alpha is None
                or not 0.0 <= self.spec_cascade_alpha <= 1.0
            ):
                raise ValueError(
                    "relaxed_target_method='spec_cascade_tok3' requires "
                    "0 <= spec_cascade_alpha <= 1."
                )
            if self.spec_cascade_tok3_top_set not in ("paper", "processed"):
                raise ValueError(
                    "relaxed_target_method='spec_cascade_tok3' requires "
                    "spec_cascade_tok3_top_set in {'paper', 'processed'}."
                )
            if self.draft_sampling_method != "stochastic":
                raise ValueError(
                    "relaxed_target_method='spec_cascade_tok3' requires "
                    "draft_sampling_method='stochastic'."
                )
            if self.rejection_sample_method != "probabilistic":
                raise ValueError(
                    "relaxed_target_method='spec_cascade_tok3' requires "
                    "rejection_sample_method='probabilistic'."
                )
            if self.cactus_delta is not None:
                raise ValueError(
                    "cactus_delta is only valid with relaxed_target_method='cactus'."
                )
            if self.verifier_weight is not None:
                raise ValueError(
                    "verifier_weight is only valid with "
                    "relaxed_target_method='ensemble'."
                )
            if self.fuzzy_divergence is not None or self.fuzzy_threshold is not None:
                raise ValueError(
                    "fuzzy_divergence / fuzzy_threshold are only valid with "
                    "relaxed_target_method='rfsd' or 'fsd'."
                )
            if self.lossy_alpha is not None:
                raise ValueError(
                    "lossy_alpha is only valid with "
                    "relaxed_target_method='lossy_spec_decode_beta1'."
                )
        elif self.relaxed_target_method in scd_methods:
            if self.method not in relaxed_target_methods:
                raise ValueError(
                    f"relaxed_target_method={self.relaxed_target_method!r} is "
                    "currently supported only with method='draft_model' or "
                    "method='mtp'."
                )
            if (
                self.scd_beta is None
                or not math.isfinite(self.scd_beta)
                or self.scd_beta < 0
            ):
                raise ValueError(
                    f"relaxed_target_method={self.relaxed_target_method!r} "
                    "requires scd_beta >= 0."
                )
            if self.scd_temperature is not None and (
                not math.isfinite(self.scd_temperature)
                or self.scd_temperature <= 0
            ):
                raise ValueError(
                    f"relaxed_target_method={self.relaxed_target_method!r} "
                    "requires scd_temperature > 0 when set."
                )
            if self.relaxed_target_method == "scd_alpha":
                if (
                    self.scd_alpha is None
                    or not math.isfinite(self.scd_alpha)
                    or not 0.0 < self.scd_alpha <= 1.0
                ):
                    raise ValueError(
                        "relaxed_target_method='scd_alpha' requires "
                        "0 < scd_alpha <= 1."
                    )
            elif self.scd_alpha is not None:
                raise ValueError(
                    "scd_alpha is only valid with relaxed_target_method='scd_alpha'."
                )
            if self.draft_sampling_method != "stochastic":
                raise ValueError(
                    f"relaxed_target_method={self.relaxed_target_method!r} requires "
                    "draft_sampling_method='stochastic'."
                )
            if self.rejection_sample_method != "probabilistic":
                raise ValueError(
                    f"relaxed_target_method={self.relaxed_target_method!r} requires "
                    "rejection_sample_method='probabilistic'."
                )
            if self.cactus_delta is not None:
                raise ValueError(
                    "cactus_delta is only valid with relaxed_target_method='cactus'."
                )
            if self.verifier_weight is not None:
                raise ValueError(
                    "verifier_weight is only valid with "
                    "relaxed_target_method='ensemble'."
                )
            if self.fuzzy_divergence is not None or self.fuzzy_threshold is not None:
                raise ValueError(
                    "fuzzy_divergence / fuzzy_threshold are only valid with "
                    "relaxed_target_method='rfsd' or 'fsd'."
                )
            if self.spec_cascade_alpha is not None:
                raise ValueError(
                    "spec_cascade_alpha is only valid with "
                    "relaxed_target_method='spec_cascade_opt' or "
                    "'spec_cascade_tok3'."
                )
            if self.lossy_alpha is not None:
                raise ValueError(
                    "lossy_alpha is only valid with "
                    "relaxed_target_method='lossy_spec_decode_beta1'."
                )
        elif self.relaxed_target_method == "lossy_spec_decode_beta1":
            if self.method not in relaxed_target_methods:
                raise ValueError(
                    "relaxed_target_method='lossy_spec_decode_beta1' is currently "
                    "supported only with method='draft_model' or method='mtp'."
                )
            if self.lossy_alpha is None or not 0.0 <= self.lossy_alpha < 1.0:
                raise ValueError(
                    "relaxed_target_method='lossy_spec_decode_beta1' requires "
                    "0 <= lossy_alpha < 1."
                )
            if self.draft_sampling_method != "stochastic":
                raise ValueError(
                    "relaxed_target_method='lossy_spec_decode_beta1' requires "
                    "draft_sampling_method='stochastic'."
                )
            if self.rejection_sample_method != "probabilistic":
                raise ValueError(
                    "relaxed_target_method='lossy_spec_decode_beta1' requires "
                    "rejection_sample_method='probabilistic'."
                )
            if self.cactus_delta is not None:
                raise ValueError(
                    "cactus_delta is only valid with relaxed_target_method='cactus'."
                )
            if self.verifier_weight is not None:
                raise ValueError(
                    "verifier_weight is only valid with "
                    "relaxed_target_method='ensemble'."
                )
            if self.fuzzy_divergence is not None or self.fuzzy_threshold is not None:
                raise ValueError(
                    "fuzzy_divergence / fuzzy_threshold are only valid with "
                    "relaxed_target_method='rfsd' or 'fsd'."
                )
            if self.spec_cascade_alpha is not None:
                raise ValueError(
                    "spec_cascade_alpha is only valid with "
                    "relaxed_target_method='spec_cascade_opt' or "
                    "'spec_cascade_tok3'."
                )
        elif self.cactus_delta is not None:
            raise ValueError(
                "cactus_delta is only valid with relaxed_target_method='cactus'."
            )
        elif self.verifier_weight is not None:
            raise ValueError(
                "verifier_weight is only valid with "
                "relaxed_target_method='ensemble'."
            )
        elif self.fuzzy_divergence is not None or self.fuzzy_threshold is not None:
            raise ValueError(
                "fuzzy_divergence / fuzzy_threshold are only valid with "
                "relaxed_target_method='rfsd' or 'fsd'."
            )
        elif self.spec_cascade_alpha is not None:
            raise ValueError(
                "spec_cascade_alpha is only valid with "
                "relaxed_target_method='spec_cascade_opt' or "
                "'spec_cascade_tok3'."
            )
        elif self.lossy_alpha is not None:
            raise ValueError(
                "lossy_alpha is only valid with "
                "relaxed_target_method='lossy_spec_decode_beta1'."
            )

        if (
            self.rejection_sample_method == "probabilistic"
            and self.method in relaxed_target_methods
            and self.draft_sampling_method != "stochastic"
        ):
            raise ValueError(
                "probabilistic draft-model/MTP rejection sampling requires "
                "draft_sampling_method='stochastic'."
            )

        if self.draft_model_config:
            self.draft_model_config.verify_with_parallel_config(
                self.draft_parallel_config
            )

        aux_hidden_states_supported = [
            "llama",
            "qwen",
            "minicpm",
            "gpt_oss",
            "hunyuan_vl",
            "hunyuan_v1_dense",
            "afmoe",
            "nemotron_h",
            "deepseek_v2",
            "deepseek_v3",
            "kimi_k2",
            "kimi_k25",
            "minimax_m2",
            "gemma4",
        ]
        if (
            self.method in ("eagle3", "extract_hidden_states", "dflash")
            and self.target_model_config
            and not any(
                supported_model in self.target_model_config.hf_text_config.model_type
                for supported_model in aux_hidden_states_supported
            )
        ):
            raise ValueError(
                f"{self.method} is only supported for {aux_hidden_states_supported}"
                f" models. Got {self.target_model_config.hf_text_config.model_type=}"
            )
        self.verify_equal_vocab_size_if_draft_model()
        return self

    def verify_equal_vocab_size_if_draft_model(self):
        if (
            self.method == "draft_model"
            and self.target_model_config is not None
            and self.draft_model_config is not None
        ):
            target_vocab_size = self.target_model_config.get_vocab_size()
            draft_vocab_size = self.draft_model_config.get_vocab_size()
            if target_vocab_size != draft_vocab_size:
                raise ValueError(
                    f"Target and draft model should have the same vocabulary size. "
                    f"Target model vocab_size={target_vocab_size}. "
                    f"Draft model vocab_size={draft_vocab_size}. "
                    f"Using models with different tokenizers can cause out-of-bounds "
                    f"errors during speculative decoding."
                )

    @property
    def max_num_new_slots_for_drafting(self) -> int:
        """
        Calculate the maximum number of new slots that might be added to the batch
        when drafting.
        """
        slots_per_req = 0  # for serial non-draft-model methods, no change needed
        if self.parallel_drafting:
            # For parallel drafting, we need one new slot per 'masked' token
            slots_per_req = self.num_speculative_tokens - 1
        if self.uses_draft_model():
            # For draft model-based speculation, we need one new slot per request
            # Since we do not slice the draft tokens
            slots_per_req += 1
        return slots_per_req

    def use_eagle(self) -> bool:
        return self.method in ("eagle", "eagle3", "mtp", "dflash")

    def use_dflash(self) -> bool:
        return self.method == "dflash"

    def uses_draft_model(self) -> bool:
        return self.method == "draft_model"

    def uses_extract_hidden_states(self) -> bool:
        return self.method == "extract_hidden_states"

    def use_ngram_gpu(self) -> bool:
        return self.method == "ngram_gpu"

    def __repr__(self) -> str:
        method = self.method
        model = (
            None
            if method in ("ngram", "suffix", "extract_hidden_states")
            else self.draft_model_config.model
        )
        num_spec_tokens = self.num_speculative_tokens
        return f"SpeculativeConfig({method=}, {model=}, {num_spec_tokens=})"
