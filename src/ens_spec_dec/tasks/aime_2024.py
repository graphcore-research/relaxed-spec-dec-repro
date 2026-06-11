"""AIME 2024 task adapter for the full local 30-question benchmark slice.

This module keeps the task-side behavior in one place: local sample loading,
prompt construction, answer parsing, scoring, and task-owned decode defaults.
The local fixture is grounded in the public `HuggingFaceH4/aime_2024` wrapper
and carries one checked-in row per 2024 AIME problem.
"""

from __future__ import annotations

import re

from ens_spec_dec.contracts import (
    GenerationParams,
    JsonValue,
    SampleInput,
    SamplingConfig,
    TaskConfig,
    TaskDefaults,
)

from ._data import load_fixture_rows
from ._text import strip_qwen_thinking, text_after_qwen_thinking

_FIXTURE_FILE = "aime_2024_full.jsonl"
_DATASET = "HuggingFaceH4/aime_2024"
_SPLIT = "train"
_PROMPT_TEMPLATE_SOURCE = (
    "https://huggingface.co/datasets/HuggingFaceH4/aime_2024/blob/"
    "e6cf0cd64082ada1c025717826bd40e155b1ec81/eval.yaml"
)
_BOXED_PATTERN = re.compile(r"\\boxed\{([^{}]+)\}")
_MARKER_PATTERNS = [
    re.compile(r"Therefore, the final answer is:\s*(.+)", re.IGNORECASE),
    re.compile(r"Final answer:\s*(.+)", re.IGNORECASE),
    re.compile(r"Answer:\s*(.+)", re.IGNORECASE),
]
_INTEGER_PATTERN = re.compile(r"[-+]?\d+")
_MAX_AIME_DIGITS = 3


def default_task_config() -> TaskConfig:
    # Source-grounded: Qwen3 recommends thinking-mode sampling with
    # temperature=0.6, top_p=0.95, and top_k=20 on the official model card.
    # https://huggingface.co/Qwen/Qwen3-8B
    sampling = SamplingConfig(
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        # Source-grounded: Qwen3 recommends up to 38,912 output tokens for
        # complex benchmarking tasks such as math and programming.
        max_new_tokens=38912,
        stop=[],
    )
    return TaskConfig(
        name="aime_2024",
        dataset=_DATASET,
        split=_SPLIT,
        # Source-grounded: AIME 2024 has 30 problems in the H4 wrapper, and
        # this local fixture mirrors the full public slice.
        max_samples=30,
        defaults=TaskDefaults(generation_params=GenerationParams(sampling=sampling)),
        metadata={
            "decode_defaults": {
                "thinking_mode": True,
                # Source-grounded: current contract surface does not yet carry
                # all Qwen3 sampling knobs such as `min_p` or penalties.
                "qwen3_sampling_source": "https://huggingface.co/Qwen/Qwen3-8B",
                "qwen3_recommended_extras": {
                    "min_p": 0.0,
                    "presence_penalty": 1.5,
                },
                "parser": {
                    "kind": "boxed_or_final_integer_answer",
                    "marker_priority": [
                        "\\boxed{...}",
                        "Therefore, the final answer is:",
                        "Final answer:",
                        "Answer:",
                    ],
                },
                # Source-grounded: match the public prompt template used in the
                # H4 wrapper eval.yaml rather than inventing repo-local wording.
                "prompt_template_source": _PROMPT_TEMPLATE_SOURCE,
                # Repo policy: use the simplest and most widely adopted AIME
                # convention for the mainline benchmark path in this repo:
                # objective final-answer exact match rather than an LLM judge.
                "evaluation_convention": "final_answer_exact_match",
            },
            # Source-grounded: the public wrapper uses `train` with 3-digit
            # answer strings. https://huggingface.co/datasets/HuggingFaceH4/aime_2024
            "dataset_card_url": "https://huggingface.co/datasets/HuggingFaceH4/aime_2024",
        },
    )


def load_samples(task_config: TaskConfig | None = None) -> list[SampleInput]:
    task_config = default_task_config() if task_config is None else task_config
    rows = load_fixture_rows(_FIXTURE_FILE, task_config.max_samples)

    samples: list[SampleInput] = []
    for row in rows:
        problem = row["problem"]
        solution = row.get("solution")
        source = {
            "dataset": _DATASET,
            "problem": problem,
            "answer": row["answer"],
            "url": row["url"],
        }
        if isinstance(solution, str):
            source["solution"] = solution
        if "year" in row:
            source["year"] = row["year"]

        samples.append(
            SampleInput(
                sample_id=row["sample_id"],
                prompt=_render_prompt(problem),
                source=source,
                reference=str(row["answer"]),
                metadata={"split": _SPLIT},
            )
        )
    return samples


def build_prompt(sample: SampleInput) -> str:
    problem = sample.source.get("problem")
    if isinstance(problem, str):
        return _render_prompt(problem)
    return sample.prompt


def parse_answer(text: str) -> str | None:
    text = strip_qwen_thinking(text)
    boxed_matches = _BOXED_PATTERN.findall(text)
    for candidate in reversed(boxed_matches):
        answer = _extract_last_integer(candidate)
        if answer is not None:
            return answer

    for pattern in _MARKER_PATTERNS:
        match = pattern.search(text)
        if match is None:
            continue
        answer = _extract_last_integer(match.group(1))
        if answer is not None:
            return answer
    return _extract_last_integer(text)


def score_answer(reference: JsonValue, parsed_answer: str | None) -> dict[str, JsonValue]:
    # Repo benchmark contract: use the H4 AIME 2024 wrapper's `train` split,
    # prompt for a boxed final answer, parse boxed/final integers, and score
    # by exact match against the public integer answer. This keeps AIME an
    # objective final-answer benchmark rather than an LLM-judge task.
    # Sources:
    # - https://huggingface.co/datasets/HuggingFaceH4/aime_2024
    # - https://huggingface.co/datasets/HuggingFaceH4/aime_2024/blob/e6cf0cd64082ada1c025717826bd40e155b1ec81/eval.yaml
    normalized_reference = _normalize_integer(reference)
    normalized_prediction = _normalize_integer(parsed_answer)
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
        "scoring_policy": "aime_final_answer_exact_match",
    }


def score_generation(
    reference: JsonValue,
    generated_text: str,
    parsed_answer: str | None,
) -> dict[str, JsonValue]:
    payload = score_answer(reference, parsed_answer)
    post_think_text = text_after_qwen_thinking(generated_text)
    post_think_answer = (
        None if post_think_text is None else parse_answer(post_think_text)
    )
    post_think_payload = score_answer(reference, post_think_answer)
    payload.update(
        {
            "post_think_correct": post_think_payload["correct"],
            "post_think_exact_match": post_think_payload["exact_match"],
            "post_think_parsed_answer": post_think_payload["parsed_answer"],
            "has_qwen_think_end": post_think_text is not None,
            "post_think_text_present": bool(post_think_text),
        }
    )
    return payload


def _render_prompt(problem: str) -> str:
    # Source-grounded: this wording intentionally follows the public
    # `HuggingFaceH4/aime_2024` eval template closely, including the boxed
    # final-answer line and "Think step by step" instruction.
    # Source: `_PROMPT_TEMPLATE_SOURCE`.
    return (
        "Solve the following math problem efficiently and clearly. "
        "The last line of your response should be of the following format: "
        "'Therefore, the final answer is: $\\boxed{ANSWER}$. I hope it is "
        "correct' (without quotes) where ANSWER is just the final number or "
        "expression that solves the problem. Think step by step before "
        "answering.\n\n"
        f"{problem}"
    )


def _extract_last_integer(text: str) -> str | None:
    matches = _INTEGER_PATTERN.findall(text)
    if not matches:
        return None
    return _normalize_integer(matches[-1])


def _normalize_integer(value: JsonValue) -> str | None:
    if value is None:
        return None

    text = str(value).strip().replace(",", "")
    if not text or not re.fullmatch(r"[-+]?\d+", text):
        return None

    sign = "-" if text.startswith("-") else ""
    digits = text[1:] if text[:1] in ("-", "+") else text
    digits = digits.lstrip("0") or "0"
    if len(digits) > _MAX_AIME_DIGITS:
        return None
    if sign and digits != "0":
        return f"-{digits}"
    return digits


__all__ = [
    "build_prompt",
    "default_task_config",
    "load_samples",
    "parse_answer",
    "score_answer",
    "score_generation",
]
