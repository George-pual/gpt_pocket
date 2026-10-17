"""Audit the three exported GRPO pools against their archived G4 screening data.

This is an offline audit only. It does not alter pool files or start training.
"""
from __future__ import annotations

import collections
import hashlib
import importlib.util
from functools import lru_cache
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "datasets/math/base_self600_20261015"
RESULTS = ROOT / "grpo_rescreen_v4_results"
REMOTE = RESULTS / "remote"
GROUPS = REMOTE / "screen_current_grpo/grpo_final/groups"
sys.path.insert(0, str(Path(__file__).resolve().parent))
from math_verification import extract_final_answer, verify_final_answer  # noqa: E402


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@lru_cache(maxsize=1)
def load_recovery_region():
    path = ROOT / "remote_results/tools/recover_math_self600_format.py"
    spec = importlib.util.spec_from_file_location("recover_math_self600_format", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module.region


def recover_explicit_final(text: str):
    """Take only an explicit final SOLUTION region; never search reasoning."""
    region = load_recovery_region()(text)
    if not region:
        return None
    _, final = region
    candidate, _, error = extract_final_answer(final)
    if not error:
        return candidate.strip()
    # This matches the prior audit's narrow fallback: a sole math span in the
    # declared final region, with no other numeric/equation material around it.
    spans = list(re.finditer(r"\\\((.+?)\\\)|\$(.+?)\$|\\\[(.+?)\\\]", final, re.S))
    if len(spans) == 1:
        span = spans[0]
        outside = final[:span.start()] + final[span.end():]
        if not re.search(r"[\d=+*/<>]", outside):
            return next(value for value in span.groups() if value is not None).strip()
    return final.strip()


def main():
    source_rows = read_jsonl(REMOTE / "data/candidates1200.jsonl")
    source = {row["row_id"]: row for row in source_rows}
    annotations = {row["row_id"]: row for row in read_jsonl(RESULTS / "annotated_scores.jsonl")}
    groups = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(GROUPS.glob("*.json"))]
    group_by_id = {group["row_id"]: group for group in groups}
    assert len(source) == len(annotations) == len(group_by_id) == 1200

    pools = {
        name: read_jsonl(RESULTS / f"train_{name}.jsonl")
        for name in ("strict_math", "mixed_math_protocol", "protocol_only")
    }
    report = {
        "status": "audited",
        "scope": "Offline quality audit of candidate prompts and archived G4 outcomes; no training started.",
        "source_questions": len(source),
        "pools": {},
        "cross_pool_overlap": {},
        "recoverable_final_answer_candidates": {},
    }
    pool_ids = {}
    for name, rows in pools.items():
        ids = [row["row_id"] for row in rows]
        pool_ids[name] = set(ids)
        assert len(ids) == len(set(ids))
        assert all(annotations[rid]["category"] == name for rid in ids)
        assert all(len(group_by_id[rid]["samples"]) == 4 for rid in ids)
        statuses = collections.Counter()
        errors = collections.Counter()
        patterns = collections.Counter()
        subjects = collections.Counter()
        difficulties = collections.Counter()
        origins = collections.Counter()
        levels = collections.Counter()
        math_only_correct_counts = collections.Counter()
        for row in rows:
            rid = row["row_id"]
            group = group_by_id[rid]
            verdicts = [sample["verdict"] for sample in group["samples"]]
            counts = collections.Counter(v["status"] for v in verdicts)
            pattern = "+".join(f"{key}:{counts[key]}" for key in sorted(counts))
            patterns[pattern] += 1
            math_only_correct_counts[counts["correct"]] += 1
            subjects[row["type"]] += 1
            difficulties[row["difficulty"]] += 1
            origins[row["selection_role"]] += 1
            levels[row["level"]] += 1
            for verdict in verdicts:
                statuses[verdict["status"]] += 1
                if verdict.get("error"):
                    errors[verdict["error"]] += 1
            assert all(sample["eos"] and not sample["truncated"] for sample in group["samples"])
            assert annotations[rid]["eligible"]
        report["pools"][name] = {
            "questions": len(rows),
            "unique_row_ids": len(set(ids)),
            "unique_prompts": len({row["prompt"][1]["content"] for row in rows}),
            "status_per_answer": dict(statuses),
            "score_patterns_per_question": dict(patterns),
            "mathematically_correct_answers_per_question": dict(math_only_correct_counts),
            "extraction_errors": dict(errors),
            "difficulty_from_total_correct_k": dict(difficulties),
            "levels": dict(levels),
            "subjects": dict(subjects),
            "old_vs_new_source": dict(origins),
        }

    for left, right in (("strict_math", "mixed_math_protocol"),
                        ("strict_math", "protocol_only"),
                        ("mixed_math_protocol", "protocol_only")):
        report["cross_pool_overlap"][f"{left}:{right}"] = len(pool_ids[left] & pool_ids[right])

    recover_region = load_recovery_region()
    recovery = collections.defaultdict(collections.Counter)
    recovered_examples = []
    recovered_records = []
    recovered_group_ids = collections.defaultdict(set)
    repaired_statuses = {}
    for group in groups:
        rid = group["row_id"]
        category = annotations[rid]["category"]
        if category not in ("mixed_math_protocol", "protocol_only"):
            continue
        outcomes = [sample["verdict"]["status"] for sample in group["samples"]]
        for sample in group["samples"]:
            verdict = sample["verdict"]
            if verdict["status"] not in ("extraction_failed", "prediction_parse_failed"):
                continue
            if not sample["eos"] or sample["truncated"]:
                continue
            recovery[category]["raw_unconfirmed_or_extraction_failed_answers"] += 1
            candidate = recover_explicit_final(sample["completion"])
            if candidate is None:
                recovery[category]["no_safe_explicit_final_region"] += 1
                continue
            checked = verify_final_answer(
                "<SOLUTION>" + candidate + "</SOLUTION>",
                str(source[rid]["answer"]),
                question=source[rid]["problem"],
            )
            recovery[category]["explicit_final_regions_regraded"] += 1
            if checked["status"] == "correct":
                recovery[category]["candidate_now_verifies_correct"] += 1
                recovered_group_ids[(category, "correct")].add(rid)
                outcomes[sample["sample_index"]] = "correct"
                if len(recovered_examples) < 24:
                    recovered_examples.append({
                        "category": category,
                        "row_id": rid,
                        "sample_index": sample["sample_index"],
                        "original_error": verdict.get("error"),
                        "candidate": candidate[:300],
                    })
            elif checked["status"] == "incorrect":
                recovery[category]["candidate_now_verifies_incorrect"] += 1
                recovered_group_ids[(category, "incorrect")].add(rid)
                outcomes[sample["sample_index"]] = "incorrect"
            else:
                recovery[category][f"still_unconfirmed:{checked['status']}"] += 1
                if checked.get("error"):
                    recovery[category][f"regrade_error:{checked['error']}"] += 1
            if checked["status"] in ("correct", "incorrect"):
                recovered_records.append({
                    "category": category,
                    "row_id": rid,
                    "sample_index": sample["sample_index"],
                    "original_error": verdict.get("error"),
                    "problem": source[rid]["problem"],
                    "gold": source[rid]["answer"],
                    "candidate_final_answer": candidate,
                    "regraded_status": checked["status"],
                    "method": "explicit final SOLUTION region only; no reasoning search or gold substitution",
                })
        repaired_statuses[rid] = (category, outcomes)
    for category in ("mixed_math_protocol", "protocol_only"):
        recovery[category]["groups_with_recovered_correct_answer"] = len(recovered_group_ids[(category, "correct")])
        recovery[category]["groups_with_recovered_incorrect_answer"] = len(recovered_group_ids[(category, "incorrect")])
    protocol_new_math_contrast = 0
    protocol_fully_repaired = 0
    for rid in pool_ids["protocol_only"]:
        _, outcomes = repaired_statuses[rid]
        if "correct" in outcomes and "incorrect" in outcomes:
            protocol_new_math_contrast += 1
        if all(value in ("correct", "incorrect") for value in outcomes):
            protocol_fully_repaired += 1
    report["recovery_reclassification_candidates"] = {
        "protocol_only_groups_with_new_correct_incorrect_contrast": protocol_new_math_contrast,
        "protocol_only_groups_with_all_four_answers_confirmed_after_strict_recovery": protocol_fully_repaired,
        "warning": "These are deterministic candidates for human spot review, not edits to original scores or training pools.",
    }
    report["recoverable_final_answer_candidates"] = {key: dict(value) for key, value in recovery.items()}
    report["recovery_examples"] = recovered_examples
    report["input_checks"] = {
        "verified_archive_status": json.loads((RESULTS / "retrieval_verification.json").read_text(encoding="utf-8"))["status"],
        "strict_pool_sha256": sha256(RESULTS / "train_strict_math.jsonl"),
        "mixed_pool_sha256": sha256(RESULTS / "train_mixed_math_protocol.jsonl"),
        "protocol_pool_sha256": sha256(RESULTS / "train_protocol_only.jsonl"),
        "all_included_groups_four_samples_and_normal_eos": True,
        "training_started": False,
    }
    path = RESULTS / "DATA_QUALITY_AUDIT.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    recovered_path = RESULTS / "recovered_answer_candidates.jsonl"
    recovered_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in recovered_records),
        encoding="utf-8",
    )
    print(json.dumps({"report": str(path), "pools": report["pools"],
                      "recoverable_final_answer_candidates": report["recoverable_final_answer_candidates"],
                      "recovery_reclassification_candidates": report["recovery_reclassification_candidates"],
                      "recovered_records": len(recovered_records)},
                     ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
