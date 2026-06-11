"""LiveCodeBench code-generation-lite v6 adapter.

This adapter prepares prompts and references for LCB-lite code generation.
Scoring is intentionally deferred to `scripts/evaluate_lcb_codegen.py`,
because correctness requires executing generated Python against bundled tests.
"""

from __future__ import annotations

from ens_spec_dec.contracts import (
    GenerationParams,
    JsonValue,
    SampleInput,
    SamplingConfig,
    TaskConfig,
    TaskDefaults,
)

from ._code import extract_python_code
from ._data import apply_task_sample_slice, load_benchmark_rows

_TASK_DIR = "livecodebench_lite_v6"
_DATA_FILE = "test.jsonl"
_DATASET = "livecodebench/code_generation_lite"
_SPLIT = "test"
_VERSION_TAG = "v6"
_EXPECTED_ROWS = 175
_LCB_REPO = "https://github.com/LiveCodeBench/LiveCodeBench"
_LCB_DATASET = "https://huggingface.co/datasets/livecodebench/code_generation_lite"
_SYSTEM_MESSAGE = (
    "You are an expert Python programmer. You will be given a question "
    "(problem specification) and will generate a correct Python program that "
    "matches the specification and passes all tests."
)
_FORMAT_WITH_STARTER_CODE = (
    "You will use the following starter code to write the solution to the "
    "problem and enclose your code within delimiters."
)
_FORMAT_WITHOUT_STARTER_CODE = (
    "Read the inputs from stdin solve the problem and write the answer to "
    "stdout (do not directly test on the sample inputs). Enclose your code "
    "within delimiters as follows. Ensure that when the python program runs, "
    "it reads the inputs, runs the algorithm and writes output to STDOUT."
)


def default_task_config() -> TaskConfig:
    sampling = SamplingConfig(
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        max_new_tokens=4096,
        stop=[],
    )
    return TaskConfig(
        name="livecodebench_lite_v6",
        dataset=_DATASET,
        split=_SPLIT,
        max_samples=_EXPECTED_ROWS,
        defaults=TaskDefaults(generation_params=GenerationParams(sampling=sampling)),
        metadata={
            "protocol": {
                "source": _LCB_REPO,
                "dataset_card": _LCB_DATASET,
                "dataset_path": _DATASET,
                "split": _SPLIT,
                "version_tag": _VERSION_TAG,
                "metric": "pass@1",
                "slice_policy": "non_cumulative_v6_lite_test_split",
                "scoring": "external_local_code_execution",
            }
        },
    )


def load_samples(task_config: TaskConfig | None = None) -> list[SampleInput]:
    task_config = default_task_config() if task_config is None else task_config
    rows = load_benchmark_rows(_TASK_DIR, _DATA_FILE)
    rows = apply_task_sample_slice(rows, task_config)

    samples: list[SampleInput] = []
    for row in rows:
        prompt = _render_user_prompt(row)
        reference = {
            "input_output": row["input_output"],
            "evaluation_kind": "livecodebench_codegen_lite",
            "version_tag": _VERSION_TAG,
        }
        source = {
            "dataset": _DATASET,
            "version_tag": _VERSION_TAG,
            "question_title": row["question_title"],
            "question_id": row["question_id"],
            "contest_id": row["contest_id"],
            "contest_date": row["contest_date"],
            "platform": row["platform"],
            "difficulty": row["difficulty"],
            "starter_code": row["starter_code"],
            "metadata": dict(row.get("metadata") or {}),
        }
        samples.append(
            SampleInput(
                sample_id=row["sample_id"],
                prompt=prompt,
                source=source,
                reference=reference,
                metadata={
                    "split": _SPLIT,
                    "prompt_protocol": "livecodebench_generic_codegen_chat",
                    "chat_messages": [
                        {"role": "system", "content": _SYSTEM_MESSAGE},
                        {"role": "user", "content": prompt},
                    ],
                },
            )
        )
    return samples


def build_prompt(sample: SampleInput) -> str:
    return sample.prompt


def parse_answer(text: str) -> str:
    return extract_python_code(text)


def score_answer(reference: JsonValue, parsed_answer: str | None) -> dict[str, JsonValue]:
    if not isinstance(reference, dict) or "input_output" not in reference:
        raise TypeError("LCB reference must contain an input_output payload")
    return {
        "scoring_policy": "livecodebench_codegen_external_pending",
        "requires_external_evaluator": True,
        "parsed_code_present": bool(parsed_answer and parsed_answer.strip()),
    }


def _render_user_prompt(row: dict) -> str:
    prompt = f"### Question:\n{row['question_content']}\n\n"
    starter_code = row.get("starter_code") or ""
    if starter_code:
        prompt += f"### Format: {_FORMAT_WITH_STARTER_CODE}\n"
        prompt += f"```python\n{starter_code}\n```\n\n"
    else:
        prompt += f"### Format: {_FORMAT_WITHOUT_STARTER_CODE}\n"
        prompt += "```python\n# YOUR CODE HERE\n```\n\n"
    prompt += "### Answer: (use the provided format with backticks)\n\n"
    return prompt


__all__ = [
    "build_prompt",
    "default_task_config",
    "load_samples",
    "parse_answer",
    "score_answer",
]
