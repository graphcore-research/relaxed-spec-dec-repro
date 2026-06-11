from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ens_spec_dec.artifacts import read_scored_samples, read_summary
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
        self.calls = 0

    def generate_batch(self, prompts, generation_params):
        self.calls += 1
        return GenerateBatchResult(
            outputs=[
                BackendSampleOutput(
                    generated_text=r"Therefore, the final answer is: \boxed{204}.",
                    generated_token_ids=[1, 2, 3],
                    n_prompt_tokens=16,
                    n_out_tokens=3,
                )
            ],
            generate_call_id=f"fake-{self.calls}",
            generate_wall_time_s=0.01,
        )

    def get_run_debug_payload(self):
        return {"backend_name": "fake", "metrics": []}


class ReleaseRunnerTest(unittest.TestCase):
    def test_aime_run_writes_scored_artifacts(self) -> None:
        task = aime_2024.default_task_config()
        task.max_samples = 1
        config = RunConfig(
            metadata=RunMetadata(
                run_id="run-release-runner-test",
                measurement_scope=MeasurementScope.OFFLINE_EVAL,
                decode_mode=DecodeMode.AR_REF,
            ),
            task=task,
            backend=BackendConfig(
                name="vllm",
                model="Qwen/Qwen3-32B",
                options={"prompt_format": "raw"},
            ),
            generation_params=GenerationParams(
                sampling=SamplingConfig(max_new_tokens=8, seed=7),
                method=MethodConfig(name="ar_ref"),
            ),
            execution=ExecutionConfig(batch_size=1, warmup_enabled=False),
        )
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = run_offline(config, runs_root=tmp, backend_factory=FakeBackend)
            samples = read_scored_samples(run_dir)
            summary = read_summary(run_dir)

        self.assertEqual(len(samples), 1)
        self.assertTrue(samples[0].score_payload["correct"])
        self.assertEqual(summary.score_summary["accuracy"], 1.0)
        self.assertEqual(summary.total_output_tokens, 3)


if __name__ == "__main__":
    unittest.main()
