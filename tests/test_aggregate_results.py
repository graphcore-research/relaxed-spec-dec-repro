from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from scripts.aggregate_results import rows_from_runs

from ens_spec_dec.contracts import (
    BackendConfig,
    DecodeMode,
    ExecutionConfig,
    GenerationParams,
    MeasurementScope,
    MethodConfig,
    RunConfig,
    RunMetadata,
    SamplingConfig,
)
from ens_spec_dec.engines import BackendSampleOutput, GenerateBatchResult
from ens_spec_dec.offline_run import run_offline
from ens_spec_dec.tasks import aime_2024


class FakeBackend:
    def __init__(self, config):
        self.config = config

    def generate_batch(self, prompts, generation_params):
        return GenerateBatchResult(
            outputs=[
                BackendSampleOutput(
                    generated_text=r"The final answer is \boxed{204}.",
                    generated_token_ids=[1, 2, 3],
                    n_prompt_tokens=16,
                    n_out_tokens=3,
                )
            ],
            generate_call_id="fake-1",
            generate_wall_time_s=0.01,
        )

    def get_run_debug_payload(self):
        return {"backend_name": "fake", "metrics": []}


class AggregateResultsTest(unittest.TestCase):
    def test_rows_from_runs_finds_nested_run_directories(self) -> None:
        task = aime_2024.default_task_config()
        task.max_samples = 1
        config = RunConfig(
            metadata=RunMetadata(
                run_id="run-nested-aggregate-test",
                measurement_scope=MeasurementScope.OFFLINE_EVAL,
                decode_mode=DecodeMode.SD_RELAXED,
            ),
            task=task,
            backend=BackendConfig(
                name="vllm",
                model="Qwen/Qwen3-32B",
                draft_model="Qwen/Qwen3-0.6B",
                options={"prompt_format": "raw"},
            ),
            generation_params=GenerationParams(
                sampling=SamplingConfig(max_new_tokens=8, seed=7),
                method=MethodConfig(
                    name="cactus",
                    variant="cactus",
                    draft_length=3,
                    params={"delta": 0.1},
                ),
            ),
            execution=ExecutionConfig(batch_size=1, warmup_enabled=False),
        )
        with tempfile.TemporaryDirectory() as tmp:
            run_offline(
                config,
                runs_root=Path(tmp) / "aime-cactus",
                backend_factory=FakeBackend,
            )
            rows = rows_from_runs(Path(tmp))

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["run_id"], "run-nested-aggregate-test")
        self.assertEqual(rows[0]["accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
