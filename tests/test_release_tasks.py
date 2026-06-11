from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ens_spec_dec.contracts import TaskConfig
from ens_spec_dec.tasks import aime_2024, gpqa_diamond, livecodebench_lite_v6


class ReleaseTasksTest(unittest.TestCase):
    def test_aime_fixture_loads_and_scores_boxed_answer(self) -> None:
        config = TaskConfig.from_dict({"name": "aime_2024", "max_samples": 1})
        sample = aime_2024.load_samples(config)[0]
        parsed = aime_2024.parse_answer(r"We get \boxed{%s}." % sample.reference)
        score = aime_2024.score_answer(sample.reference, parsed)
        self.assertTrue(score["correct"])

    def test_gpqa_final_choice_parser(self) -> None:
        parsed = gpqa_diamond.parse_answer("After reasoning, the answer is (C).")
        self.assertEqual(parsed, "(C)")
        self.assertTrue(gpqa_diamond.score_answer("(C)", parsed)["correct"])

    def test_lcb_extracts_final_fenced_python_block(self) -> None:
        text = "```python\nprint('wrong')\n```\nfinal\n```python\nprint('ok')\n```"
        parsed = livecodebench_lite_v6.parse_answer(text)
        self.assertEqual(parsed.strip(), "print('ok')")


if __name__ == "__main__":
    unittest.main()
