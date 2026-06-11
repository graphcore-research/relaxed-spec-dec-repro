from __future__ import annotations

import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class ReleaseOverlayTest(unittest.TestCase):
    def test_overlay_contains_only_source_files(self) -> None:
        overlay = REPO_ROOT / "vllm_overlay" / "files"
        pycache_dirs = list(overlay.rglob("__pycache__"))
        pyc_files = list(overlay.rglob("*.pyc"))
        self.assertEqual(pycache_dirs, [])
        self.assertEqual(pyc_files, [])

    def test_expected_overlay_files_exist(self) -> None:
        files = {
            "files/config/speculative.py",
            "files/v1/sample/rejection_sampler.py",
            "files/v1/spec_decode/llm_base_proposer.py",
            "files/v1/spec_decode/metrics.py",
            "files/v1/worker/gpu_model_runner.py",
        }
        for relative in files:
            self.assertTrue((REPO_ROOT / "vllm_overlay" / relative).is_file())


if __name__ == "__main__":
    unittest.main()
