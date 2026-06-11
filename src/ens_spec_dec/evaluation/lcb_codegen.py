"""Local LiveCodeBench code-generation scoring over saved run artifacts.

Generation stays in the normal offline runner. This module starts from a
completed LCB run directory, extracts code from each model output, executes it
against the bundled LCB-lite tests, updates `samples.jsonl`, and regenerates
the stable summary.
"""

from __future__ import annotations

import builtins
import contextlib
import io
import json
import multiprocessing as mp
import queue
import signal
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from ens_spec_dec.artifacts import (
    RUN_ARTIFACT_PATHS,
    read_manifest,
    read_scored_samples,
    write_debug_json,
)
from ens_spec_dec.contracts import JsonDict, ScoredSample, write_scored_samples_jsonl
from ens_spec_dec.evaluation.summary import regenerate_run_summary
from ens_spec_dec.tasks._code import extract_python_code

TASK_NAME = "livecodebench_lite_v6"
EXTRACTION_POLICY = "livecodebench_official_last_fenced_block"
SCORING_POLICY = "livecodebench_lite_v6_official_extract_pass_at_1"
MAX_JOB_TIMEOUT_SECONDS = 60
COMMON_IMPORTS = """
from string import *
from re import *
from datetime import *
from collections import *
from heapq import *
from bisect import *
from copy import *
from math import *
from random import *
from statistics import *
from itertools import *
from functools import *
from operator import *
from io import *
from sys import *
from json import *
from typing import *
import builtins
import string
import re
import datetime
import collections
import heapq
import bisect
import copy
import math
import random
import statistics
import itertools
import functools
import operator
import io
import sys
import json
pow = builtins.pow
sys.setrecursionlimit(50000)
"""


class TimeoutException(Exception):
    pass


@dataclass(slots=True)
class CodeEvalConfig:
    num_process_evaluate: int = 4
    timeout: int = 6
    debug: bool = False

    def to_dict(self) -> JsonDict:
        return {
            "num_process_evaluate": self.num_process_evaluate,
            "timeout": self.timeout,
            "debug": self.debug,
        }


def evaluate_lcb_codegen_run(
    run_dir: str | Path,
    config: CodeEvalConfig | None = None,
) -> JsonDict:
    """Score one completed LCB generation run and regenerate its summary."""

    config = CodeEvalConfig() if config is None else config
    run_path = Path(run_dir)
    manifest = read_manifest(run_path)
    if manifest.config.task.name != TASK_NAME:
        raise ValueError(f"expected task {TASK_NAME}, saw {manifest.config.task.name}")

    samples = read_scored_samples(run_path)
    jobs = [_job_from_sample(index, sample, config) for index, sample in enumerate(samples)]
    results = _run_jobs(jobs, config)
    evaluated_samples = [
        _sample_with_score(sample, results[index])
        for index, sample in enumerate(samples)
    ]
    write_scored_samples_jsonl(
        run_path / RUN_ARTIFACT_PATHS["samples"],
        evaluated_samples,
    )

    debug_payload = _debug_payload(manifest.run_id, config, evaluated_samples, results)
    write_debug_json(run_path, "lcb_code_eval.json", debug_payload)
    regenerate_run_summary(run_path)
    return debug_payload


def _job_from_sample(
    index: int,
    sample: ScoredSample,
    config: CodeEvalConfig,
) -> JsonDict:
    reference = sample.sample.reference
    if not isinstance(reference, dict) or not isinstance(reference.get("input_output"), str):
        raise ValueError(f"sample {sample.sample.sample_id} lacks LCB input_output")
    code = extract_python_code(sample.generation.generated_text)
    return {
        "index": index,
        "sample_id": sample.sample.sample_id,
        "input_output": reference["input_output"],
        "code": code,
        "timeout": config.timeout,
    }


def _run_jobs(jobs: list[JsonDict], config: CodeEvalConfig) -> dict[int, JsonDict]:
    if not jobs:
        return {}

    results: dict[int, JsonDict] = {}
    workers = max(1, config.num_process_evaluate)
    if config.debug or workers == 1:
        for job in jobs:
            results[int(job["index"])] = _evaluate_job(job)
        return results

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_evaluate_job, job): int(job["index"]) for job in jobs}
        for future in as_completed(futures):
            index = futures[future]
            results[index] = future.result()
    return results


def _evaluate_job(job: JsonDict) -> JsonDict:
    input_output = json.loads(str(job["input_output"]))
    code = str(job["code"])
    timeout = int(job["timeout"])
    started_at = time.perf_counter()
    try:
        results, metadata = _run_lcb_tests_with_process_timeout(
            input_output,
            code,
            timeout,
        )
    except Exception as exc:
        results = [-4]
        metadata = {"error": repr(exc), "error_code": -4, "error_message": "EvalError"}

    elapsed = time.perf_counter() - started_at
    passed = bool(results) and all(value is True for value in results)
    passed_count = sum(value is True for value in results)
    return {
        "sample_id": job["sample_id"],
        "code": code,
        "code_extracted": bool(code.strip()),
        "results": list(results),
        "metadata": dict(metadata),
        "correct": passed,
        "passed_test_count": passed_count,
        "test_count": len(input_output.get("inputs") or []),
        "eval_wall_time_s": elapsed,
    }


def _run_lcb_tests(
    input_output: dict[str, Any],
    code: str,
    timeout: int,
) -> tuple[list[bool | int], JsonDict]:
    inputs = list(input_output.get("inputs") or [])
    outputs = list(input_output.get("outputs") or [])
    if len(inputs) != len(outputs):
        return [-5], {"error_code": -5, "error_message": "MalformedTests"}

    fn_name = input_output.get("fn_name")
    if fn_name:
        return _run_functional_tests(code, inputs, outputs, str(fn_name), timeout)
    return _run_stdio_tests(code, inputs, outputs, timeout)


def _run_lcb_tests_with_process_timeout(
    input_output: dict[str, Any],
    code: str,
    timeout: int,
) -> tuple[list[bool | int], JsonDict]:
    ctx = mp.get_context("fork")
    result_queue = ctx.Queue(maxsize=1)
    process = ctx.Process(
        target=_run_lcb_tests_child,
        args=(input_output, code, timeout, result_queue),
    )
    process.start()
    process.join(_job_timeout_seconds(input_output, timeout))
    if process.is_alive():
        process.terminate()
        process.join(1)
        if process.is_alive():
            process.kill()
            process.join()
        return [-3], {
            "error_code": -3,
            "error_message": "Time Limit Exceeded",
            "timeout_s": _job_timeout_seconds(input_output, timeout),
        }

    try:
        payload = result_queue.get_nowait()
    except queue.Empty:
        return [-4], {
            "error_code": -4,
            "error_message": "EvalError",
            "exitcode": process.exitcode,
        }

    if payload["ok"]:
        return payload["results"], payload["metadata"]
    return [-4], payload["metadata"]


def _run_lcb_tests_child(
    input_output: dict[str, Any],
    code: str,
    timeout: int,
    result_queue,
) -> None:
    try:
        results, metadata = _run_lcb_tests(input_output, code, timeout)
        result_queue.put({"ok": True, "results": results, "metadata": metadata})
    except BaseException as exc:
        result_queue.put(
            {
                "ok": False,
                "metadata": {
                    "error": repr(exc),
                    "error_code": -4,
                    "error_message": "EvalError",
                },
            }
        )


def _job_timeout_seconds(input_output: dict[str, Any], timeout: int) -> int:
    n_tests = len(input_output.get("inputs") or [])
    return min(MAX_JOB_TIMEOUT_SECONDS, max(1, timeout) * (n_tests + 1))


def _run_functional_tests(
    code: str,
    inputs: list[str],
    outputs: list[str],
    fn_name: str,
    timeout: int,
) -> tuple[list[bool | int], JsonDict]:
    namespace: dict[str, Any] = {"__name__": "__lcb_solution__"}
    try:
        _with_alarm(lambda: exec(f"{COMMON_IMPORTS}\n{code}", namespace), timeout)
        target = namespace.get(fn_name)
        if target is None and isinstance(namespace.get("Solution"), type):
            target = getattr(namespace["Solution"](), fn_name, None)
        if target is None:
            return [-4], {"error_code": -4, "error_message": "MissingFunction"}
    except Exception as exc:
        return [-4], {
            "error": repr(exc),
            "error_code": -4,
            "error_message": "CompileError",
        }

    results: list[bool | int] = []
    started_at = time.perf_counter()
    for raw_inputs, raw_output in zip(inputs, outputs, strict=True):
        try:
            args = [json.loads(line) for line in raw_inputs.split("\n") if line != ""]
            expected = json.loads(raw_output)
            prediction = _with_alarm(lambda: target(*args), timeout)
        except TimeoutException as exc:
            results.append(-3)
            return results, _runtime_error("Time Limit Exceeded", exc, raw_inputs, raw_output)
        except Exception as exc:
            results.append(-4)
            return results, _runtime_error("Runtime Error", exc, raw_inputs, raw_output)

        prediction = _json_like(prediction)
        expected = _json_like(expected)
        if prediction != expected:
            results.append(-2)
            return results, {
                "error_code": -2,
                "error_message": "Wrong Answer",
                "inputs": _truncate(raw_inputs),
                "expected": _truncate(expected),
                "output": _truncate(prediction),
            }
        results.append(True)

    return results, {"execution_time": time.perf_counter() - started_at}


def _run_stdio_tests(
    code: str,
    inputs: list[str],
    outputs: list[str],
    timeout: int,
) -> tuple[list[bool | int], JsonDict]:
    results: list[bool | int] = []
    started_at = time.perf_counter()
    for raw_inputs, raw_output in zip(inputs, outputs, strict=True):
        try:
            prediction = _with_alarm(lambda: _exec_stdio(code, raw_inputs), timeout)
        except TimeoutException as exc:
            results.append(-3)
            return results, _runtime_error("Time Limit Exceeded", exc, raw_inputs, raw_output)
        except Exception as exc:
            results.append(-4)
            return results, _runtime_error("Runtime Error", exc, raw_inputs, raw_output)

        ok, message = _stdio_outputs_match(prediction, raw_output)
        if not ok:
            results.append(-2)
            return results, {
                "error_code": -2,
                "error_message": message,
                "inputs": _truncate(raw_inputs),
                "expected": _truncate(raw_output),
                "output": _truncate(prediction),
            }
        results.append(True)

    return results, {"execution_time": time.perf_counter() - started_at}


def _exec_stdio(code: str, stdin_text: str) -> str:
    namespace: dict[str, Any] = {"__name__": "__main__"}
    stdout = io.StringIO()
    stderr = io.StringIO()
    stdin = _MockStdin(stdin_text)
    real_open = builtins.open

    def patched_open(file, *args, **kwargs):
        if file == 0:
            return io.StringIO(stdin_text)
        return real_open(file, *args, **kwargs)

    old_stdin = sys.stdin
    old_open = builtins.open
    existing_threads = set(threading.enumerate())
    current_thread = threading.current_thread()
    try:
        sys.stdin = stdin
        builtins.open = patched_open
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            try:
                exec(f"{COMMON_IMPORTS}\n{code}", namespace)
            except SystemExit:
                pass
            _join_spawned_threads(existing_threads, current_thread)
    finally:
        builtins.open = old_open
        sys.stdin = old_stdin
    return stdout.getvalue()


def _join_spawned_threads(
    existing_threads: set[threading.Thread],
    current_thread: threading.Thread,
) -> None:
    """Wait for submitted-code worker threads while the judge captures stdout."""

    while True:
        spawned = [
            thread
            for thread in threading.enumerate()
            if thread is not current_thread
            and thread not in existing_threads
            and not thread.daemon
        ]
        if not spawned:
            return
        for thread in spawned:
            thread.join()


class _MockStdin:
    def __init__(self, text: str) -> None:
        self._text = text
        self._stream = io.StringIO(text)
        self.buffer = io.BytesIO(text.encode("utf-8"))

    def read(self, *args):
        return self._stream.read(*args)

    def readline(self, *args):
        return self._stream.readline(*args)

    def readlines(self, *args):
        return self._stream.readlines(*args)

    def __iter__(self):
        return iter(self._stream)


def _with_alarm(fn, timeout: int):
    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(timeout)
    try:
        return fn()
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def _timeout_handler(signum, frame) -> None:
    raise TimeoutException("timeout")


def _stdio_outputs_match(prediction: str, expected: str) -> tuple[bool, str]:
    predicted_lines = _stripped_lines(prediction)
    expected_lines = _stripped_lines(expected)
    if len(predicted_lines) != len(expected_lines):
        return False, "Wrong answer: mismatched output length"

    for index, (predicted, target) in enumerate(zip(predicted_lines, expected_lines)):
        if predicted == target:
            continue
        predicted_numbers = _decimal_tokens(predicted)
        target_numbers = _decimal_tokens(target)
        if predicted_numbers is not None and predicted_numbers == target_numbers:
            continue
        return False, f"Wrong answer at output_line_idx={index}"
    return True, "OK"


def _stripped_lines(text: str) -> list[str]:
    return [line.strip() for line in text.strip().split("\n")]


def _decimal_tokens(text: str) -> list[Decimal] | None:
    try:
        return [Decimal(item) for item in text.split()]
    except InvalidOperation:
        return None


def _runtime_error(message: str, exc: Exception, inputs: object, expected: object) -> JsonDict:
    return {
        "error": repr(exc),
        "error_code": -3 if message == "Time Limit Exceeded" else -4,
        "error_message": message,
        "inputs": _truncate(inputs),
        "expected": _truncate(expected),
    }


def _json_like(value: object) -> object:
    if isinstance(value, tuple):
        return [_json_like(item) for item in value]
    if isinstance(value, list):
        return [_json_like(item) for item in value]
    if isinstance(value, dict):
        return {key: _json_like(item) for key, item in value.items()}
    return value


def _truncate(value: object, length: int = 300) -> str:
    text = value if isinstance(value, str) else repr(value)
    if len(text) <= length:
        return text
    half = length // 2
    return f"{text[:half]}...(truncated)...{text[-half:]}"


def _sample_with_score(sample: ScoredSample, result: JsonDict) -> ScoredSample:
    score_payload = {
        "correct": bool(result["correct"]),
        "exact_match": 1.0 if result["correct"] else 0.0,
        "pass_at_1": 1.0 if result["correct"] else 0.0,
        "passed_test_count": int(result["passed_test_count"]),
        "test_count": int(result["test_count"]),
        "code_extracted": bool(result["code_extracted"]),
        "extraction_policy": EXTRACTION_POLICY,
        "scoring_policy": SCORING_POLICY,
    }
    sample.score_payload = score_payload
    sample.generation.parsed_answer = result["code"]
    return sample


def _debug_payload(
    run_id: str,
    config: CodeEvalConfig,
    samples: list[ScoredSample],
    results: dict[int, JsonDict],
) -> JsonDict:
    ordered = [results[index] for index in sorted(results)]
    n_correct = sum(1 for result in ordered if result["correct"])
    return {
        "run_id": run_id,
        "task_name": TASK_NAME,
        "scoring_policy": SCORING_POLICY,
        "extraction_policy": EXTRACTION_POLICY,
        "evaluator_config": config.to_dict(),
        "n_samples": len(samples),
        "n_correct": n_correct,
        "pass_at_1": (n_correct / len(samples)) if samples else None,
        "source_note": (
            "Local offline execution against bundled LiveCodeBench-lite tests. "
            "This is not an online judge call and CPU evaluation time is kept "
            "separate from vLLM generation wall time."
        ),
        "samples": [
            {
                "sample_id": result["sample_id"],
                "correct": result["correct"],
                "passed_test_count": result["passed_test_count"],
                "test_count": result["test_count"],
                "code_extracted": result["code_extracted"],
                "eval_wall_time_s": result["eval_wall_time_s"],
                "results": result["results"],
                "metadata": result["metadata"],
            }
            for result in ordered
        ],
    }


__all__ = ["CodeEvalConfig", "evaluate_lcb_codegen_run"]
