#!/usr/bin/env python3
"""Prepare only the benchmarks used by the paper.

AIME24 is shipped as a 30-row fixture in the package. This script materializes
GPQA Diamond and LiveCodeBench Code Generation Lite v6 under data/benchmarks/.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import pickle
import random
import re
import urllib.request
import zlib
from pathlib import Path

TASKS = ("gpqa", "lcb")
EXPECTED_ROWS = {
    "gpqa": ("gpqa_diamond/train.jsonl", 198),
    "lcb": ("livecodebench_lite_v6/test.jsonl", 175),
}
LCB_LITE_V6_SOURCE_URL = (
    "https://huggingface.co/datasets/livecodebench/code_generation_lite/"
    "resolve/main/test6.jsonl"
)
LCB_LITE_V6_SOURCE_DESCRIPTION = (
    "LiveCodeBench Code Generation Lite v6, non-cumulative 175-problem test split"
)
LETTERS = ("A", "B", "C", "D")
GPQA_DESCRIPTION = (
    "Here are some example questions from experts. Answer the question by "
    "selecting one of the provided choices."
)
GPQA_FINAL_CHOICE_INSTRUCTION = (
    "Let's think step by step. After reasoning, end with exactly:\n"
    "The answer is (X)."
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default=os.environ.get("SPEC_DEC_BENCHMARK_DATA_ROOT", "data/benchmarks"),
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=("all", *TASKS),
        default=["all"],
    )
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser()
    tasks = TASKS if "all" in args.tasks else tuple(args.tasks)
    if args.check_only:
        _check_expected_files(output_dir, tasks)
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    if "gpqa" in tasks:
        _prepare_gpqa(_load_dataset_fn(), output_dir, args.hf_token)
    if "lcb" in tasks:
        _prepare_lcb_lite_v6(output_dir)
    _check_expected_files(output_dir, tasks)


def _load_dataset_fn():
    try:
        from datasets import load_dataset
    except ModuleNotFoundError as exc:
        raise SystemExit("Missing dependency `datasets`; run `uv sync`.") from exc
    return load_dataset


def _prepare_gpqa(load_dataset, output_dir: Path, hf_token: str | None) -> None:
    kwargs = {"token": hf_token} if hf_token else {}
    dataset = load_dataset("Idavidrein/gpqa", "gpqa_diamond", **kwargs)
    train_rows = list(dataset["train"])
    rng = random.Random(0)
    rows = []
    for index, row in enumerate(train_rows):
        correct_answer = row["Correct Answer"]
        choice_items = [
            (row["Incorrect Answer 1"], False),
            (row["Incorrect Answer 2"], False),
            (row["Incorrect Answer 3"], False),
            (correct_answer, True),
        ]
        rng.shuffle(choice_items)
        choices = [choice for choice, _is_correct in choice_items]
        answer_index = next(
            idx for idx, (_choice, is_correct) in enumerate(choice_items) if is_correct
        )
        user_prompt = _render_gpqa_user_prompt(row["Question"], choices)
        rows.append(
            {
                "sample_id": str(
                    row.get("Record ID") or row.get("record_id") or f"gpqa-{index:03d}"
                ),
                "question": row["Question"],
                "choices": choices,
                "correct_answer": correct_answer,
                "answer": f"({LETTERS[answer_index]})",
                "prompt": f"{GPQA_DESCRIPTION}\n\n{user_prompt}",
                "user_prompt": user_prompt,
                "choice_shuffle_seed": 0,
            }
        )
    _write_jsonl(output_dir / "gpqa_diamond" / "train.jsonl", rows)


def _prepare_lcb_lite_v6(output_dir: Path) -> None:
    print(f"preparing {LCB_LITE_V6_SOURCE_DESCRIPTION}")
    rows = []
    for index, row in enumerate(_iter_lcb_lite_v6_rows()):
        metadata = _json_object(row["metadata"])
        public_tests = _test_cases(row["public_test_cases"])
        private_tests = _test_cases(row["private_test_cases"])
        all_tests = public_tests + private_tests
        input_output = {
            "inputs": [test["input"] for test in all_tests],
            "outputs": [test["output"] for test in all_tests],
            "fn_name": metadata.get("func_name"),
        }
        question_id = str(row["question_id"])
        rows.append(
            {
                "sample_id": f"lcb-lite-v6-{index:04d}-{_safe_id(question_id)}",
                "question_title": row["question_title"],
                "question_content": row["question_content"],
                "platform": row["platform"],
                "question_id": question_id,
                "contest_id": str(row["contest_id"]),
                "contest_date": str(row["contest_date"]),
                "starter_code": row["starter_code"] or "",
                "difficulty": row["difficulty"],
                "metadata": metadata,
                "input_output": json.dumps(input_output, ensure_ascii=False),
                "public_test_count": len(public_tests),
                "private_test_count": len(private_tests),
            }
        )
    _write_jsonl(output_dir / "livecodebench_lite_v6" / "test.jsonl", rows)


def _iter_lcb_lite_v6_rows():
    headers = {}
    token = os.environ.get("HF_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(LCB_LITE_V6_SOURCE_URL, headers=headers)
    with urllib.request.urlopen(request, timeout=60) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8")
            if line.strip():
                yield json.loads(line)


def _render_gpqa_user_prompt(question: str, choices: list[str]) -> str:
    choice_lines = "\n".join(
        f"({letter}) {choice}" for letter, choice in zip(LETTERS, choices, strict=True)
    )
    return f"Question: {question}\nChoices:\n{choice_lines}\n{GPQA_FINAL_CHOICE_INSTRUCTION}"


def _json_object(value: object) -> dict:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        return json.loads(value)
    raise TypeError(f"expected JSON object, saw {type(value).__name__}")


def _test_cases(value: object) -> list[dict]:
    if isinstance(value, list):
        return [dict(item) for item in value]
    if not isinstance(value, str):
        raise TypeError(f"expected JSON test case string, saw {type(value).__name__}")
    try:
        cases = json.loads(value)
    except json.JSONDecodeError:
        raw = base64.b64decode(value.encode("utf-8"))
        cases = json.loads(pickle.loads(zlib.decompress(raw)))
    return [dict(item) for item in cases]


def _safe_id(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip())
    return text.strip("-") or "problem"


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {path} ({len(rows)} rows)")


def _check_expected_files(output_dir: Path, tasks: tuple[str, ...]) -> None:
    for task in tasks:
        relative, expected_rows = EXPECTED_ROWS[task]
        path = output_dir / relative
        if not path.is_file():
            raise SystemExit(f"missing {path}")
        row_count = _count_jsonl_rows(path)
        if row_count != expected_rows:
            raise SystemExit(f"{path}: expected {expected_rows} rows, saw {row_count}")
        print(f"ok {path} ({row_count} rows)")


def _count_jsonl_rows(path: Path) -> int:
    with path.open(encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


if __name__ == "__main__":
    main()
