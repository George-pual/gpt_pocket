"""Optional integration tests: skip in a clean checkout without local datasets."""
import json
from pathlib import Path
import unittest

from r1_lab.evidence import build_report

ROOT = Path(__file__).resolve().parents[1]
COMBINED = ROOT / "datasets/math/test_l2_l3_remaining_v1/combined_analysis_v32_final"


@unittest.skipUnless((COMBINED / "summary.json").exists(), "Local experiment artifacts not distributed")
class LocalArtifactTests(unittest.TestCase):
    def test_recomputed_metrics_match_frozen_reports(self):
        registry = json.loads((ROOT / "configs/evidence_registry.json").read_text(encoding="utf-8"))
        result = build_report(ROOT, registry)
        old = json.loads((COMBINED / "summary.json").read_text(encoding="utf-8"))
        for dataset, old_cohort in (("math", "math_all"), ("gsm8k", "gsm8k")):
            for model, expected in old[old_cohort].items():
                actual = result["datasets"][dataset]["metrics"]["all"][model]
                for key in ("total", "correct", "unconfirmed", "eos", "truncated", "repeated", "mean_tokens", "median_tokens"):
                    self.assertAlmostEqual(actual[key], expected[key], msg=f"{dataset}/{model}/{key}")
        v4 = result["datasets"]["math"]["metrics"]["all"]["sft_v4"]
        self.assertEqual(v4["correct"], 1386)
        for model, expected in old["math_new1824"].items():
            actual = result["datasets"]["math"]["metrics"]["remaining1824"][model]
            self.assertEqual(actual["correct"], expected["correct"])


if __name__ == "__main__":
    unittest.main()
