import json
from pathlib import Path
import tempfile
import unittest

from r1_lab.cli import main, render_html
from r1_lab.evidence import audit_splits, build_report, metrics, paired, sha256, validate_predictions, wilson


def prediction(rid="q1", correct=True):
    return {"row_id": rid, "question": "What is 1 + 1?", "gold": "2", "completion": "2",
            "tokens": 1, "eos": True, "truncated": False, "repeated_20_word_span_3_times": False,
            "audited_score_v3": {"schema": "audited-final-answer-v3.2",
                                  "correct": correct, "status": "correct" if correct else "incorrect"}}


class EvidenceTests(unittest.TestCase):
    def test_unknown_verdict_remains_in_denominator(self):
        unknown = prediction("q2", False)
        unknown["audited_score_v3"]["status"] = "prediction_unsupported"
        result = metrics([prediction(), unknown])
        self.assertEqual(result["accuracy"], 0.5)
        self.assertEqual(result["unconfirmed"], 1)

    def test_pair_by_id_not_position(self):
        a = validate_predictions([prediction("a"), prediction("b", False)])
        b = validate_predictions([prediction("b"), prediction("a", False)])
        result = paired(a, b)
        self.assertEqual((result["gain"], result["loss"], result["net"]), (1, 1, 0))

    def test_mismatch_cannot_be_silently_paired(self):
        a = validate_predictions([prediction()])
        changed = prediction()
        changed["gold"] = "3"
        with self.assertRaisesRegex(ValueError, "mismatch"):
            paired(a, validate_predictions([changed]))
        with self.assertRaisesRegex(ValueError, "identical"):
            paired(a, validate_predictions([prediction("other")]))

    def test_duplicate_and_invalid_fields_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_predictions([prediction(), prediction()])
        for key, value in (("tokens", True), ("tokens", -1), ("eos", "false"), ("truncated", True)):
            row = prediction()
            row[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                validate_predictions([row])
        row = prediction()
        row["audited_score_v3"]["schema"] = "old"
        with self.assertRaisesRegex(ValueError, "scorer"):
            validate_predictions([row])

    def test_inconsistent_score_rejected(self):
        row = prediction()
        row["audited_score_v3"]["correct"] = False
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            validate_predictions([row])

    def test_wilson_bounds(self):
        self.assertAlmostEqual(wilson(50, 100)[0], 0.40383153, places=7)
        self.assertAlmostEqual(wilson(0, 100)[0], 0)
        self.assertAlmostEqual(wilson(100, 100)[1], 1)

    def test_split_audit_normalization_and_messages(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "train.jsonl").write_text(json.dumps({"problem": "Ａ + B?"}) + "\n" + json.dumps({"question": "a + b"}), encoding="utf-8")
            (root / "eval.jsonl").write_text(json.dumps({"messages": [{"role": "user", "content": "a + b"}]}), encoding="utf-8")
            result = audit_splits({"train": root / "train.jsonl", "eval": root / "eval.jsonl"})
            self.assertFalse(result["ok"])
            self.assertEqual(result["splits"]["train"]["duplicates"], 1)
            self.assertEqual(result["overlaps"][0]["count"], 1)
            self.assertEqual(main(["audit", "--split", f"train={root / 'train.jsonl'}", "--split", f"eval={root / 'eval.jsonl'}"]), 1)

    def test_registry_hash_and_html_escaping(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / "predictions.jsonl"
            path.write_text(json.dumps(prediction()) + "\n", encoding="utf-8")
            registry = {"datasets": {"<script>": {"expected_rows": 1, "models": {
                "base": {"path": path.name, "sha256": sha256(path)}}}}}
            report = build_report(root, registry)
            rendered = render_html(report)
            self.assertNotIn("<script>", rendered)
            self.assertIn("&lt;script&gt;", rendered)
            path.write_text(json.dumps(prediction(correct=False)), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Hash mismatch"):
                build_report(root, registry)

    def test_cli_failure_exit_code(self):
        self.assertEqual(main(["audit", "--split", "bad"]), 2)


if __name__ == "__main__":
    unittest.main()
