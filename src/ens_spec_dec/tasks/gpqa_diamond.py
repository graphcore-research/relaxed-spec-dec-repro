"""GPQA Diamond adapter for the pinned lm-eval CoT generate-until protocol.

The prepared JSONL contains the gated `Idavidrein/gpqa` Diamond train split
after lm-eval's deterministic answer shuffle. Cactus seeds Python random with
0 and leaves `num_fewshot` at 0 for this task.
"""

from __future__ import annotations

import re
import random

from ens_spec_dec.contracts import (
    GenerationParams,
    JsonValue,
    SampleInput,
    SamplingConfig,
    TaskConfig,
    TaskDefaults,
)

from ._data import apply_task_sample_slice, load_benchmark_rows
from ._text import strip_qwen_thinking

_TASK_DIR = "gpqa_diamond"
_DATA_FILE = "train.jsonl"
_DATASET = "Idavidrein/gpqa"
_DATASET_NAME = "gpqa_diamond"
_SPLIT = "train"
_EXPECTED_ROWS = 198
_DESCRIPTION = (
    "Here are some example questions from experts. Answer the question by "
    "selecting one of the provided choices."
)
_LM_EVAL_DESCRIPTION = (
    "Here are some example questions from experts. Answer the final question "
    "yourself, following the format of the previous questions exactly."
)
_PROMPT_PROTOCOL = "gpqa_diamond_cot_final_choice_v1"
_DEFAULT_PROMPT_VARIANT = "local_final_choice"
_DEFAULT_ANSWER_ORDER_VARIANT = "prepared"
_PROMPT_VARIANTS = {
    _DEFAULT_PROMPT_VARIANT,
    "lm_eval_cot",
    "lighteval_instruct",
    "evalscope",
    "opencompass_simple_evals",
    "opencompass_legacy",
}
_ANSWER_ORDER_VARIANTS = {
    _DEFAULT_ANSWER_ORDER_VARIANT,
    "lighteval_seed0",
    "simple_evals_seed0",
    "opencompass_rotate",
}
_FINAL_CHOICE_INSTRUCTION = (
    "Let's think step by step. After reasoning, end with exactly:\n"
    "The answer is (X)."
)
_LEGACY_COT_SUFFIX = "Let's think step by step:"
_LM_EVAL_YAML = (
    "https://github.com/EleutherAI/lm-evaluation-harness/blob/"
    "v0.4.11/"
    "lm_eval/tasks/gpqa/cot_n_shot/gpqa_diamond_cot_n_shot.yaml"
)
_LM_EVAL_UTILS = (
    "https://github.com/EleutherAI/lm-evaluation-harness/blob/"
    "v0.4.11/"
    "lm_eval/tasks/gpqa/cot_n_shot/utils.py"
)
_CHOICE_PATTERN = re.compile(r"\(([A-D])\)", re.IGNORECASE)
_BOXED_CHOICE_PATTERN = re.compile(
    r"\\boxed\{\s*(?:\\(?:text|mathrm)\{\s*)?\(?\s*([A-D])"
    r"(?:\s*\)|[.)])?[^{}]*(?:\})?\s*\}",
    re.IGNORECASE,
)
_FINAL_CHOICE_PATTERN = re.compile(
    r"(?:final\s+answer|correct\s+answer|answer)\s*(?:is|:|=)"
    r"\s*(?:\*\*)?\(?\s*([A-D])\s*\)?",
    re.IGNORECASE,
)


def default_task_config() -> TaskConfig:
    # Source-grounded max token cap: Cactus drives lm-eval generate_until with
    # max_gen_toks=4096 for GPQA; configs set model-specific sampling knobs.
    sampling = SamplingConfig(max_new_tokens=4096, stop=[])
    return TaskConfig(
        name="gpqa_diamond",
        dataset=f"{_DATASET}/{_DATASET_NAME}",
        split=_SPLIT,
        max_samples=_EXPECTED_ROWS,
        defaults=TaskDefaults(generation_params=GenerationParams(sampling=sampling)),
        metadata={
            "protocol": {
                "source": _LM_EVAL_YAML,
                "process_docs_source": _LM_EVAL_UTILS,
                "dataset_path": _DATASET,
                "dataset_name": _DATASET_NAME,
                "split": _SPLIT,
                "num_fewshot": 0,
                "choice_shuffle_seed": 0,
                "metric": "exact_match",
                "filter": "boxed_or_final_choice_then_last_parenthesized_choice",
                "prompt_protocol": _PROMPT_PROTOCOL,
                "prompt_variant": _DEFAULT_PROMPT_VARIANT,
                "answer_order_variant": _DEFAULT_ANSWER_ORDER_VARIANT,
            }
        },
    )


def load_samples(task_config: TaskConfig | None = None) -> list[SampleInput]:
    task_config = default_task_config() if task_config is None else task_config
    prompt_variant = _prompt_variant(task_config)
    answer_order_variant = _answer_order_variant(task_config)
    rows = load_benchmark_rows(_TASK_DIR, _DATA_FILE)
    rows = apply_task_sample_slice(rows, task_config)
    order_rng = _answer_order_rng(answer_order_variant)

    samples: list[SampleInput] = []
    for index, row in enumerate(rows):
        row = _row_with_answer_order(row, answer_order_variant, index, order_rng)
        prompt, messages = _render_prompt_and_messages(row, prompt_variant)
        samples.append(
            SampleInput(
                sample_id=row["sample_id"],
                prompt=prompt,
                source={
                    "dataset": _DATASET,
                    "dataset_name": _DATASET_NAME,
                    "question": row["question"],
                    "choices": list(row["choices"]),
                    "correct_answer": row["correct_answer"],
                },
                reference=row["answer"],
                metadata={
                    "split": _SPLIT,
                    "prompt_protocol": _PROMPT_PROTOCOL,
                    "prompt_variant": prompt_variant,
                    "answer_order_variant": answer_order_variant,
                    "chat_messages": messages,
                },
            )
        )
    return samples


def build_prompt(sample: SampleInput) -> str:
    return sample.prompt


def parse_answer(text: str) -> str | None:
    text = strip_qwen_thinking(text)

    boxed_matches = _BOXED_CHOICE_PATTERN.findall(text)
    if boxed_matches:
        return f"({boxed_matches[-1].upper()})"

    final_matches = _FINAL_CHOICE_PATTERN.findall(text)
    if final_matches:
        return f"({final_matches[-1].upper()})"

    choice_matches = _CHOICE_PATTERN.findall(text)
    if not choice_matches:
        return None
    return f"({choice_matches[-1].upper()})"


def _with_final_choice_instruction(text: str) -> str:
    stripped = text.rstrip()
    if _FINAL_CHOICE_INSTRUCTION in stripped:
        return stripped
    if stripped.endswith(_LEGACY_COT_SUFFIX):
        stripped = stripped[: -len(_LEGACY_COT_SUFFIX)].rstrip()
    return f"{stripped}\n{_FINAL_CHOICE_INSTRUCTION}"


def _prompt_variant(task_config: TaskConfig) -> str:
    value = task_config.metadata.get("prompt_variant", _DEFAULT_PROMPT_VARIANT)
    variant = str(value)
    if variant not in _PROMPT_VARIANTS:
        allowed = ", ".join(sorted(_PROMPT_VARIANTS))
        raise ValueError(f"unknown GPQA prompt_variant {variant!r}; allowed: {allowed}")
    return variant


def _answer_order_variant(task_config: TaskConfig) -> str:
    value = task_config.metadata.get(
        "answer_order_variant",
        _DEFAULT_ANSWER_ORDER_VARIANT,
    )
    variant = str(value)
    if variant not in _ANSWER_ORDER_VARIANTS:
        allowed = ", ".join(sorted(_ANSWER_ORDER_VARIANTS))
        raise ValueError(
            f"unknown GPQA answer_order_variant {variant!r}; allowed: {allowed}"
        )
    return variant


def _answer_order_rng(variant: str) -> random.Random | None:
    if variant in {"lighteval_seed0", "simple_evals_seed0"}:
        return random.Random(0)
    return None


def _row_with_answer_order(
    row: dict[str, JsonValue],
    variant: str,
    index: int,
    rng: random.Random | None,
) -> dict[str, JsonValue]:
    if variant == _DEFAULT_ANSWER_ORDER_VARIANT:
        return row

    correct, incorrect = _split_correct_and_incorrect(row)
    if variant == "lighteval_seed0":
        if rng is None:
            raise AssertionError("lighteval_seed0 requires a random generator")
        gold_index = rng.randint(0, 3)
        choices = list(incorrect)
        choices.insert(gold_index, correct)
    elif variant == "simple_evals_seed0":
        if rng is None:
            raise AssertionError("simple_evals_seed0 requires a random generator")
        permutation = rng.sample(range(4), 4)
        base_choices = [correct, *incorrect]
        choices = [base_choices[position] for position in permutation]
        gold_index = permutation.index(0)
    elif variant == "opencompass_rotate":
        # OpenCompass' legacy GPQA loader starts counting at one before
        # selecting from this four-pattern cycle.
        base_choices = [correct, *incorrect]
        pattern = ("ABCD", "BCDA", "CDAB", "DABC")[(index + 1) % 4]
        choices = [base_choices[ord(letter) - ord("A")] for letter in pattern]
        gold_index = choices.index(correct)
    else:  # defensive; _answer_order_variant validates the value.
        raise AssertionError(f"unhandled answer_order_variant: {variant}")

    reordered = dict(row)
    reordered["choices"] = choices
    reordered["answer"] = f"({'ABCD'[gold_index]})"
    reordered["answer_order_variant"] = variant
    return reordered


def _split_correct_and_incorrect(row: dict[str, JsonValue]) -> tuple[str, list[str]]:
    correct = str(row["correct_answer"]).strip()
    choices = [str(choice).strip() for choice in row["choices"]]
    incorrect = [choice for choice in choices if choice != correct]
    if correct not in choices or len(incorrect) != 3:
        raise ValueError(
            "GPQA public answer-order variants require exactly one correct "
            "choice and three distinct incorrect choices"
        )
    return correct, incorrect


def _render_prompt_and_messages(
    row: dict[str, JsonValue],
    prompt_variant: str,
) -> tuple[str, list[dict[str, str]]]:
    question = str(row["question"]).strip()
    choices = [str(choice).strip() for choice in row["choices"]]

    if prompt_variant == _DEFAULT_PROMPT_VARIANT:
        prompt = _with_final_choice_instruction(str(row["prompt"]))
        user_prompt = _with_final_choice_instruction(str(row.get("user_prompt", prompt)))
        return prompt, [
            {"role": "system", "content": _DESCRIPTION},
            {"role": "user", "content": user_prompt},
        ]

    if prompt_variant == "lm_eval_cot":
        prompt = _render_lm_eval_cot_prompt(question, choices)
        return prompt, [
            {"role": "system", "content": _LM_EVAL_DESCRIPTION},
            {"role": "user", "content": prompt},
        ]

    if prompt_variant == "lighteval_instruct":
        prompt = _render_lighteval_prompt(question, choices)
    elif prompt_variant == "evalscope":
        prompt = _render_evalscope_prompt(question, choices)
    elif prompt_variant == "opencompass_simple_evals":
        prompt = _render_opencompass_simple_evals_prompt(question, choices)
    elif prompt_variant == "opencompass_legacy":
        prompt = _render_opencompass_legacy_prompt(question, choices)
    else:  # defensive; _prompt_variant validates the value.
        raise AssertionError(f"unhandled GPQA prompt_variant: {prompt_variant}")
    return prompt, [{"role": "user", "content": prompt}]


def _choice_lines_parenthesized(choices: list[str]) -> str:
    return "\n".join(
        f"({letter}) {choice}" for letter, choice in zip("ABCD", choices, strict=True)
    )


def _choice_lines_public(choices: list[str]) -> str:
    return "\n".join(
        f"{letter}) {choice}" for letter, choice in zip("ABCD", choices, strict=True)
    )


def _render_lm_eval_cot_prompt(question: str, choices: list[str]) -> str:
    return (
        f"Question: {question}\n"
        f"Choices:\n{_choice_lines_parenthesized(choices)}\n"
        "Let's think step by step:"
    )


def _render_lighteval_prompt(question: str, choices: list[str]) -> str:
    return (
        "Answer the following multiple choice question.\n"
        "The last line of your response should be of the following format: "
        "'Answer: $LETTER' (without quotes) where LETTER is one of ABCD.\n"
        "Think step by step before answering.\n\n"
        f"{question}\n\n"
        f"{_choice_lines_public(choices)}"
    )


def _render_evalscope_prompt(question: str, choices: list[str]) -> str:
    return (
        "Answer the following multiple choice question. The last line of your "
        "response should be of the following format: 'ANSWER: [LETTER]' "
        "(without quotes) where [LETTER] is one of A,B,C,D. Think step by step "
        "before answering.\n\n"
        f"{question}\n\n"
        f"{_choice_lines_public(choices)}"
    )


def _render_opencompass_simple_evals_prompt(
    question: str,
    choices: list[str],
) -> str:
    return (
        "Answer the following multiple choice question. The last line of your "
        "response should be of the following format: 'ANSWER: $LETTER' "
        "(without quotes) where LETTER is one of ABCD.\n"
        "Think step by step before answering.\n"
        f"{question}\n"
        f"{_choice_lines_public(choices)}"
    )


def _render_opencompass_legacy_prompt(question: str, choices: list[str]) -> str:
    choice_lines = "\n".join(
        f"({letter}){choice}" for letter, choice in zip("ABCD", choices, strict=True)
    )
    return (
        f"What is the correct answer to this question: {question}\n"
        f"Choices:\n{choice_lines}\n"
        'Format your response as follows: "The correct answer is (insert answer here)"'
    )


def score_answer(reference: JsonValue, parsed_answer: str | None) -> dict[str, JsonValue]:
    normalized_reference = _normalize_choice(reference)
    normalized_prediction = _normalize_choice(parsed_answer)
    correct = (
        normalized_reference is not None
        and normalized_prediction is not None
        and normalized_reference == normalized_prediction
    )
    return {
        "correct": correct,
        "exact_match": 1.0 if correct else 0.0,
        "reference_answer": normalized_reference,
        "parsed_answer": normalized_prediction,
        "scoring_policy": "lm_eval_gpqa_parenthesized_choice_exact_match",
    }


def _normalize_choice(value: JsonValue) -> str | None:
    if value is None:
        return None
    match = _CHOICE_PATTERN.fullmatch(str(value).strip())
    if match is None:
        return None
    return f"({match.group(1).upper()})"


__all__ = [
    "build_prompt",
    "default_task_config",
    "load_samples",
    "parse_answer",
    "score_answer",
]
