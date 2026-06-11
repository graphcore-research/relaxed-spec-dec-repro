from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[1]
GRID_PATH = REPO_ROOT / "scripts" / "run_paper_grid.py"


def load_grid_module():
    spec = importlib.util.spec_from_file_location("run_paper_grid", GRID_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class ReleaseGridTest(unittest.TestCase):
    def setUp(self) -> None:
        self.grid = load_grid_module()

    def test_full_grid_count_includes_all_tasks_models_methods_and_values(self) -> None:
        args = SimpleNamespace(
            preset="full",
            tasks=list(self.grid.TASKS),
            model_settings=list(self.grid.MODELS),
            methods=list(self.grid.METHODS),
            draft_lengths=list(self.grid.DRAFT_LENGTHS),
            max_configs=None,
            first_value_only=False,
            seed=7,
            tensor_parallel_size=None,
            max_model_len=None,
            gpu_memory_utilization=None,
            override_verifier_model=None,
            override_draft_model=None,
        )
        rows = list(self.grid.iter_grid(args))
        self.assertEqual(len(rows), 873)

    def test_r_fuzzy_maps_to_js_rfsd_backend(self) -> None:
        row = self._single_row(method="r_fuzzy")
        config = self.grid.build_config(row)
        method = config["generation_params"]["method"]
        self.assertEqual(config["metadata"]["decode_mode"], "sd_relaxed")
        self.assertEqual(method["name"], "rfsd")
        self.assertEqual(method["params"]["divergence_metric"], "js")
        self.assertEqual(method["params"]["divergence_threshold"], 0.05)

    def test_native_mtp_relaxed_uses_no_draft_model(self) -> None:
        row = self._single_row(method="cactus", model_setting="qwen35_mtp_27b")
        config = self.grid.build_config(row)
        self.assertEqual(config["metadata"]["decode_mode"], "sd_mtp_relaxed")
        self.assertIsNone(config["backend"]["draft_model"])
        self.assertEqual(config["generation_params"]["method"]["draft_length"], 3)

    def test_debug_caps_are_clipped(self) -> None:
        row = self._single_row(method="cactus")
        config = self.grid.build_config(row)
        sampling = config["generation_params"]["sampling"]
        self.assertEqual(config["task"]["max_samples"], 1)
        self.assertEqual(sampling["thinking_budget_tokens"], 96)
        self.assertEqual(sampling["max_new_tokens"], 128)

    def _single_row(self, *, method: str, model_setting: str = "qwen3_0p6b_32b"):
        args = SimpleNamespace(
            preset="debug",
            tasks=["aime24"],
            model_settings=[model_setting],
            methods=[method],
            draft_lengths=[3],
            max_configs=1,
            first_value_only=False,
            seed=7,
            tensor_parallel_size=None,
            max_model_len=None,
            gpu_memory_utilization=None,
            override_verifier_model=None,
            override_draft_model=None,
        )
        return next(self.grid.iter_grid(args))


if __name__ == "__main__":
    unittest.main()
