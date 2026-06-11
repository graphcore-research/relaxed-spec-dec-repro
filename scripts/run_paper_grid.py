#!/usr/bin/env python3
"""Generate and optionally run paper reproduction configs."""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ens_spec_dec.evaluation.lcb_codegen import CodeEvalConfig, evaluate_lcb_codegen_run
from ens_spec_dec.offline_run import run_offline_from_yaml

DRAFT_LENGTHS = (3, 5, 10, 20)
BASE_SEED = 7


@dataclass(frozen=True)
class TaskSpec:
    key: str
    task_name: str
    dataset: str
    split: str
    full_samples: int


@dataclass(frozen=True)
class ModelSpec:
    key: str
    verifier: str
    draft_model: str | None
    native_mtp: bool = False


@dataclass(frozen=True)
class MethodSpec:
    key: str
    paper_label: str
    backend_name: str
    param_name: str | None
    values: tuple[float, ...]
    include_draft_lengths: bool = True
    baseline: bool = False


TASKS = {
    "aime24": TaskSpec(
        key="aime24",
        task_name="aime_2024",
        dataset="HuggingFaceH4/aime_2024",
        split="train",
        full_samples=30,
    ),
    "gpqa": TaskSpec(
        key="gpqa",
        task_name="gpqa_diamond",
        dataset="Idavidrein/gpqa/gpqa_diamond",
        split="train",
        full_samples=198,
    ),
    "lcb": TaskSpec(
        key="lcb",
        task_name="livecodebench_lite_v6",
        dataset="livecodebench/code_generation_lite",
        split="test",
        full_samples=175,
    ),
}

MODELS = {
    "qwen3_0p6b_32b": ModelSpec(
        key="qwen3_0p6b_32b",
        verifier="Qwen/Qwen3-32B",
        draft_model="Qwen/Qwen3-0.6B",
    ),
    "qwen3_8b_32b": ModelSpec(
        key="qwen3_8b_32b",
        verifier="Qwen/Qwen3-32B",
        draft_model="Qwen/Qwen3-8B",
    ),
    "qwen35_mtp_27b": ModelSpec(
        key="qwen35_mtp_27b",
        verifier="Qwen/Qwen3.5-27B",
        draft_model=None,
        native_mtp=True,
    ),
}

METHODS = {
    "ar": MethodSpec("ar", "AR", "ar_ref", None, (), False, True),
    "strict": MethodSpec("strict", "strict spec-dec", "strict", None, (), True, True),
    "always_accept": MethodSpec(
        "always_accept",
        "always-accept",
        "ensemble",
        "verifier_weight",
        (0.0,),
    ),
    "cactus": MethodSpec("cactus", "CACTUS", "cactus", "delta", (0.10, 0.25, 1.0, 10.0)),
    "ens": MethodSpec("ens", "ens", "ensemble", "verifier_weight", (0.95, 0.90, 0.80)),
    "spec_casc_opt": MethodSpec(
        "spec_casc_opt",
        "spec-casc-opt",
        "spec_cascade_opt",
        "spec_cascade_alpha",
        (-0.10, 0.0, 0.05),
    ),
    "spec_casc_tok": MethodSpec(
        "spec_casc_tok",
        "spec-casc-tok",
        "spec_cascade_tok3",
        "spec_cascade_alpha",
        (0.0, 0.35, 0.80),
    ),
    "mentored_dec": MethodSpec(
        "mentored_dec",
        "mentored-dec",
        "lossy_spec_decode_beta1",
        "lossy_alpha",
        (0.95, 0.70, 0.30),
    ),
    "spec_cont_dec": MethodSpec(
        "spec_cont_dec",
        "spec-cont-dec",
        "scd_expert_toppk_gated",
        "scd_beta",
        (0.10, 0.50, 1.0),
    ),
    "r_fuzzy": MethodSpec(
        "r_fuzzy",
        "r-fuzzy",
        "rfsd",
        "divergence_threshold",
        (0.05, 0.20, 0.30),
    ),
}


def main() -> None:
    args = parse_args()
    rows = list(iter_grid(args))
    if args.command == "list":
        for row in rows:
            print(row["run_id"])
        print(f"configs={len(rows)}")
        return

    output_dir = Path(args.output_dir)
    written = write_configs(rows, output_dir, dry_run=args.dry_run)
    if args.command == "write-configs":
        print(f"configs={len(written)} output_dir={output_dir}")
        return

    if args.command != "run":
        raise AssertionError(args.command)
    if args.dry_run:
        for path in written:
            print(path)
        print(f"dry-run configs={len(written)}")
        return

    for path, row in zip(written, rows, strict=True):
        run_dir = run_offline_from_yaml(path, runs_root=args.runs_root)
        print(run_dir)
        if row["task"].task_name == "livecodebench_lite_v6" and args.evaluate_lcb:
            evaluate_lcb_codegen_run(
                run_dir,
                CodeEvalConfig(
                    num_process_evaluate=args.lcb_processes,
                    timeout=args.lcb_timeout,
                ),
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("list", "write-configs", "run"))
    parser.add_argument("--preset", choices=("debug", "full"), default="debug")
    parser.add_argument("--tasks", nargs="+", choices=TASKS.keys(), default=list(TASKS))
    parser.add_argument(
        "--model-settings",
        nargs="+",
        choices=MODELS.keys(),
        default=list(MODELS),
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=METHODS.keys(),
        default=list(METHODS),
    )
    parser.add_argument("--draft-lengths", nargs="+", type=int, default=list(DRAFT_LENGTHS))
    parser.add_argument("--max-configs", type=int, default=None)
    parser.add_argument(
        "--first-value-only",
        action="store_true",
        help="For smoke tests, keep only the first relaxation value per method.",
    )
    parser.add_argument("--output-dir", default="configs/generated")
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--tensor-parallel-size", type=int, default=None)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=None)
    parser.add_argument(
        "--override-verifier-model",
        default=None,
        help="Use a smaller verifier model for mechanical smoke tests only.",
    )
    parser.add_argument(
        "--override-draft-model",
        default=None,
        help="Use a smaller draft model for mechanical smoke tests only.",
    )
    parser.add_argument("--seed", type=int, default=BASE_SEED)
    parser.add_argument("--evaluate-lcb", action="store_true")
    parser.add_argument("--lcb-processes", type=int, default=1)
    parser.add_argument("--lcb-timeout", type=int, default=6)
    return parser.parse_args()


def iter_grid(args: argparse.Namespace):
    count = 0
    for task_key in args.tasks:
        task = TASKS[task_key]
        for model_key in args.model_settings:
            model = MODELS[model_key]
            for method_key in args.methods:
                method = METHODS[method_key]
                draft_lengths = (
                    [None]
                    if not method.include_draft_lengths
                    else [d for d in args.draft_lengths if d in DRAFT_LENGTHS]
                )
                values: tuple[float | None, ...] = (
                    (None,) if not method.values else method.values
                )
                if args.first_value_only and values != (None,):
                    values = values[:1]
                for draft_length in draft_lengths:
                    for value in values:
                        row = {
                            "task": task,
                            "model": model,
                            "method": method,
                            "draft_length": draft_length,
                            "value": value,
                            "preset": args.preset,
                            "seed": args.seed,
                            "tensor_parallel_size": args.tensor_parallel_size,
                            "max_model_len": args.max_model_len,
                            "gpu_memory_utilization": args.gpu_memory_utilization,
                            "override_verifier_model": args.override_verifier_model,
                            "override_draft_model": args.override_draft_model,
                        }
                        row["run_id"] = run_id(row)
                        yield row
                        count += 1
                        if args.max_configs is not None and count >= args.max_configs:
                            return


def write_configs(rows: list[dict[str, Any]], output_dir: Path, *, dry_run: bool) -> list[Path]:
    paths = []
    for row in rows:
        path = output_dir / f"{row['run_id']}.yaml"
        paths.append(path)
        if dry_run:
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(build_config(row), sort_keys=False), encoding="utf-8")
    return paths


def build_config(row: dict[str, Any]) -> dict[str, Any]:
    task: TaskSpec = row["task"]
    model: ModelSpec = row["model"]
    method: MethodSpec = row["method"]
    preset = row["preset"]
    debug = preset == "debug"
    max_samples = 1 if debug else task.full_samples
    thinking_budget, max_new_tokens = token_caps(task, model, debug=debug)
    decode_mode, backend_method_name = decode_mode_and_method(model, method)
    draft_length = row["draft_length"]
    backend_options = {
        "prompt_format": "chat_model_template",
        "chat_template_kwargs": {"enable_thinking": True},
        "gdn_prefill_backend": "triton",
        "language_model_only": True,
    }
    if model.native_mtp:
        backend_options["trust_remote_code"] = True
        backend_options.pop("language_model_only", None)
    if debug:
        backend_options["max_num_seqs"] = 1
        backend_options["max_num_batched_tokens"] = row["max_model_len"] or 4096
        backend_options["max_model_len"] = row["max_model_len"] or 4096
    elif row["max_model_len"] is not None:
        backend_options["max_model_len"] = row["max_model_len"]
    if row["gpu_memory_utilization"] is not None:
        backend_options["gpu_memory_utilization"] = row["gpu_memory_utilization"]
    verifier_model = row["override_verifier_model"] or model.verifier
    draft_model = (
        row["override_draft_model"]
        if row["override_draft_model"] is not None
        else model.draft_model
    )

    method_params = method_params_for(method, row["value"])
    return {
        "metadata": {
            "run_id": row["run_id"],
            "measurement_scope": "offline_eval",
            "decode_mode": decode_mode,
            "tags": ["paper_repro", preset, task.key, model.key, method.key],
        },
        "task": {
            "name": task.task_name,
            "dataset": task.dataset,
            "split": task.split,
            "max_samples": max_samples,
            "defaults": {
                "generation_params": {
                    "sampling": {
                        "temperature": 0.6,
                        "top_p": 0.95,
                        "top_k": 20,
                        "max_new_tokens": max_new_tokens,
                        "thinking_budget_tokens": thinking_budget,
                        "stop": [],
                        "seed": None,
                    },
                    "method": {
                        "name": None,
                        "variant": None,
                        "method_version": None,
                        "draft_length": None,
                        "params": {},
                    },
                }
            },
            "metadata": task_metadata(task),
        },
        "backend": {
            "name": "vllm",
            "model": verifier_model,
            "draft_model": None if method.key == "ar" else draft_model,
            "revision": None,
            "dtype": "bfloat16",
            "options": backend_options,
        },
        "execution": {
            "batch_size": 1,
            "tensor_parallel_size": row["tensor_parallel_size"],
            "data_parallel_size": None,
            "warmup_enabled": False if debug else True,
            "warmup_sample_count": 1,
            "warmup_max_new_tokens": 64 if debug else 256,
            "record_warmup_debug": False,
            "repeat_count": 1 if debug else repeat_count(task),
            "repeat_batch_size": 1,
            "repeat_seed_offset": 0,
            "seed_list": [] if debug else seed_list(task),
        },
        "generation_params": {
            "sampling": {
                "temperature": 0.6,
                "top_p": 0.95,
                "top_k": 20,
                "max_new_tokens": max_new_tokens,
                "thinking_budget_tokens": thinking_budget,
                "stop": [],
                "seed": row["seed"],
            },
            "method": {
                "name": backend_method_name,
                "variant": paper_variant(row),
                "method_version": "paper-repro-v1",
                "draft_length": draft_length,
                "params": method_params,
            },
        },
        "debug_refs": {},
    }


def token_caps(task: TaskSpec, model: ModelSpec, *, debug: bool) -> tuple[int | None, int]:
    if debug:
        return 96, 128
    if model.native_mtp and task.key in {"aime24", "lcb"}:
        return 65536, 69632
    return 32768, 36864


def repeat_count(task: TaskSpec) -> int:
    return 48 if task.key == "aime24" else 6


def seed_list(task: TaskSpec) -> list[int]:
    return [BASE_SEED + i for i in range(repeat_count(task))]


def decode_mode_and_method(model: ModelSpec, method: MethodSpec) -> tuple[str, str]:
    if method.key == "ar":
        return "ar_ref", "ar_ref"
    if method.key == "strict":
        return ("sd_mtp", "sd_mtp") if model.native_mtp else ("sd_vanilla", "sd_vanilla")
    return ("sd_mtp_relaxed", method.backend_name) if model.native_mtp else ("sd_relaxed", method.backend_name)


def method_params_for(method: MethodSpec, value: float | None) -> dict[str, Any]:
    if method.key in {"ar", "strict"}:
        return {}
    params: dict[str, Any] = {
        "relaxed_target_method": method.backend_name,
        "bonus_token_policy": "target_p",
    }
    if method.key == "always_accept":
        params["relaxed_target_method"] = "ensemble"
        params["verifier_weight"] = 0.0
        return params
    if method.key == "r_fuzzy":
        params["divergence_metric"] = "js"
    if method.key == "spec_cont_dec":
        params["bonus_token_policy"] = "relaxed_T_qp"
    if value is not None and method.param_name is not None:
        params[method.param_name] = float(value)
    return params


def task_metadata(task: TaskSpec) -> dict[str, Any]:
    if task.key == "aime24":
        return {
            "protocol": {
                "metric": "boxed_or_final_integer_exact_match",
                "paper_slice": "30 H4 AIME 2024 problems",
            }
        }
    if task.key == "gpqa":
        return {
            "protocol": {
                "metric": "final_choice_exact_match",
                "paper_slice": "198 GPQA Diamond questions",
                "answer_choice_shuffle_seed": 0,
            }
        }
    if task.key == "lcb":
        return {
            "protocol": {
                "metric": "pass@1",
                "paper_slice": "175-problem non-cumulative v6 Code Generation Lite split",
            }
        }
    raise AssertionError(task.key)


def paper_variant(row: dict[str, Any]) -> str:
    method: MethodSpec = row["method"]
    value = row["value"]
    if value is None:
        return method.paper_label
    return f"{method.paper_label}:{method.param_name}={value:g}"


def run_id(row: dict[str, Any]) -> str:
    method: MethodSpec = row["method"]
    parts = [
        row["preset"],
        row["task"].key,
        row["model"].key,
        method.key,
    ]
    if row["draft_length"] is not None:
        parts.append(f"d{row['draft_length']}")
    if row["value"] is not None and method.param_name is not None:
        parts.append(f"{method.param_name}-{format_value(row['value'])}")
    parts.append(f"s{row['seed']}")
    return "run-" + "-".join(slug(part) for part in parts)


def format_value(value: float) -> str:
    text = f"{value:g}".replace("-", "m").replace(".", "p")
    return text


def slug(value: object) -> str:
    text = str(value).lower().replace("_", "-")
    return re.sub(r"[^a-z0-9.-]+", "-", text).strip("-")


if __name__ == "__main__":
    main()
