"""Apply direct human decisions to the remaining 85 uncertain final answers."""
from __future__ import annotations

import collections
import json
import glob
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "datasets/math/base_self600_20261015/grpo_rescreen_v4_results"
REFINED = ROOT / "answer_correctness_reaudit_refined.jsonl"
FINAL = ROOT / "answer_correctness_reaudit_final.jsonl"
SUMMARY = ROOT / "answer_correctness_reaudit_final_summary.json"
REMOTE = ROOT / "remote"
GROUPS = REMOTE / "screen_current_grpo/grpo_final/groups"
ANNOTATIONS = ROOT / "annotated_scores.jsonl"
REPORT = ROOT / "ANSWER_CORRECTNESS_REAUDIT_REPORT.md"

# Decisions are in the original JSONL order within each unresolved status
# class. Correctness refers to the stated final answer for the given prompt.
NO_VERIFIER_LABELS = [
    "incorrect", "incorrect", "incorrect", "incorrect", "correct", "incorrect", "incorrect",
    "incorrect", "incorrect", "correct", "incorrect", "incorrect", "incorrect", "correct",
    "correct", "incorrect", "incorrect", "correct", "incorrect", "correct", "correct", "correct",
    "incorrect", "incorrect", "incorrect", "incorrect", "incorrect", "incorrect", "incorrect",
    "incorrect", "incorrect", "incorrect", "incorrect", "incorrect", "incorrect", "incorrect",
    "correct", "incorrect", "incorrect", "correct", "incorrect", "correct",
]
PARSE_FAILURE_LABELS = [
    "correct", "correct", "incorrect", "correct", "correct", "correct", "incorrect", "correct",
    "correct", "correct", "incorrect", "incorrect", "incorrect", "correct", "incorrect", "correct",
    "incorrect", "correct", "correct", "correct", "correct", "incorrect", "correct", "incorrect",
    "incorrect", "correct", "incorrect", "correct", "correct", "correct", "incorrect", "correct",
    "correct", "correct", "incorrect", "incorrect", "correct", "correct", "correct", "correct",
    "incorrect", "incorrect", "correct",
]


def load(path):
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def main():
    rows = load(REFINED)
    no_label = [r for r in rows if (r.get("refined_verification") or r.get("candidate_verification") or {}).get("status") is None]
    parse_failed = [r for r in rows if (r.get("refined_verification") or r.get("candidate_verification") or {}).get("status") == "prediction_parse_failed"]
    assert len(no_label) == len(NO_VERIFIER_LABELS) == 42, (len(no_label), len(NO_VERIFIER_LABELS))
    assert len(parse_failed) == len(PARSE_FAILURE_LABELS) == 43, (len(parse_failed), len(PARSE_FAILURE_LABELS))
    decisions = {}
    for group, labels in ((no_label, NO_VERIFIER_LABELS), (parse_failed, PARSE_FAILURE_LABELS)):
        for row, label in zip(group, labels):
            decisions[(row["row_id"], row["sample_index"])] = label
    for row in rows:
        key = (row["row_id"], row["sample_index"])
        v = row.get("refined_verification") or row.get("candidate_verification") or {}
        if key in decisions:
            row["final_verdict"] = decisions[key]
            row["verdict_source"] = "manual_review_of_prompt_and_full_completion"
        elif v.get("status") in ("correct", "incorrect"):
            row["final_verdict"] = v["status"]
            row["verdict_source"] = "math_verify_v4_recheck"
        else:
            raise AssertionError(f"unreviewed row: {key} {v.get('status')}")
    FINAL.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    summary = {"answers_reviewed": len(rows), "manual_final_answers": len(decisions),
               "correct": sum(r["final_verdict"] == "correct" for r in rows),
               "incorrect": sum(r["final_verdict"] == "incorrect" for r in rows),
               "by_pool": {}, "by_question_correct_count": {},
               "note": "MathVerify-v4 determinate results were retained; unresolved/malformed or absent answers were reviewed against the original prompt and full completion. No completion or original score was edited."}
    annotations = {r["row_id"]: r for r in load(ANNOTATIONS)}
    reviewed = {(r["row_id"], r["sample_index"]): r for r in rows}
    all_groups = collections.defaultdict(list)
    for path in sorted(GROUPS.glob("*.json")):
        group = json.loads(path.read_text(encoding="utf-8"))
        rid = group["row_id"]
        category = annotations[rid]["category"]
        if category not in ("strict_math", "mixed_math_protocol", "protocol_only"):
            continue
        for sample in group["samples"]:
            status = sample["verdict"]["status"]
            if status not in ("correct", "incorrect"):
                status = reviewed[(rid, sample["sample_index"])]["final_verdict"]
            all_groups[category].append((rid, status))
    for category, expected_questions in (("strict_math", 224), ("mixed_math_protocol", 190), ("protocol_only", 366)):
        decisions_by_question = collections.defaultdict(list)
        for rid, verdict in all_groups[category]:
            decisions_by_question[rid].append(verdict)
        assert len(decisions_by_question) == expected_questions
        assert all(len(v) == 4 for v in decisions_by_question.values())
        k_counts = collections.Counter(sum(v == "correct" for v in vs) for vs in decisions_by_question.values())
        by_answer = collections.Counter(v for _, v in all_groups[category])
        summary["by_pool"][category] = {
            "questions": len(decisions_by_question), "answers": len(all_groups[category]),
            "correct_answers": by_answer["correct"], "incorrect_answers": by_answer["incorrect"],
            "manual_answers": sum(r["category"] == category and r["verdict_source"].startswith("manual") for r in rows),
            "questions_by_correct_answers_out_of_4": {str(k): k_counts[k] for k in range(5)},
            "questions_with_math_correct_incorrect_contrast": sum(k_counts[k] for k in (1, 2, 3)),
        }
        summary["by_question_correct_count"][category] = {str(k): k_counts[k] for k in range(5)}
    combined = collections.Counter()
    for pool in summary["by_pool"].values():
        for k, count in pool["questions_by_correct_answers_out_of_4"].items():
            combined[int(k)] += count
    summary["combined_questions"] = sum(combined.values())
    summary["combined_questions_by_correct_answers_out_of_4"] = {str(k): combined[k] for k in range(5)}
    summary["combined_math_contrast_questions"] = sum(combined[k] for k in (1, 2, 3))
    summary["combined_all_four_correct_questions"] = combined[4]
    summary["combined_correct_answers"] = sum(p["correct_answers"] for p in summary["by_pool"].values())
    summary["combined_incorrect_answers"] = sum(p["incorrect_answers"] for p in summary["by_pool"].values())
    SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pool_names = {"strict_math": "224题：原本四答都能判分", "mixed_math_protocol": "190题：原本有数学对错、混有格式失败", "protocol_only": "366题：原本被归为仅格式差异"}
    lines = ["# 1200题重筛答案重新核对", "", "逐题检查了原题、标准答案和四份完整回答。此前能正常判分的回答保留v4判分；此前解析失败的724份里，639份经修复答案提取后由v4判分，另外85份直接看完整回答判定。回答没有给出结果或答到别的问题，计为答错。原始文件和原判分没有覆盖。", "", "| 原题池 | 题数 | 答对份数 | 答错份数 | 四答中有对有错 | 四答全对 | 按每题答对数：1 / 2 / 3 / 4 |", "|---|---:|---:|---:|---:|---:|---|"]
    for cat in ("strict_math", "mixed_math_protocol", "protocol_only"):
        p = summary["by_pool"][cat]
        k = p["questions_by_correct_answers_out_of_4"]
        lines.append(f"| {pool_names[cat]} | {p['questions']} | {p['correct_answers']} | {p['incorrect_answers']} | {p['questions_with_math_correct_incorrect_contrast']} | {k['4']} | {k['1']} / {k['2']} / {k['3']} / {k['4']} |")
    k = summary["combined_questions_by_correct_answers_out_of_4"]
    lines += ["", f"合并三个池后，{summary['combined_questions']}题共3120份回答：{summary['combined_correct_answers']}份答对、{summary['combined_incorrect_answers']}份答错。按每题答对数分组为1答对{ k['1']}题、2答对{k['2']}题、3答对{k['3']}题、4答对{k['4']}题；没有四答全错的题。", "", f"按数学对错奖励，{summary['combined_math_contrast_questions']}题出现了至少一个正确和一个错误回答，可形成组内对比；另有{k['4']}题四答全对，没有数学对错对比。原先366题的‘仅格式’标签不成立：复核后其中{summary['by_pool']['protocol_only']['questions_with_math_correct_incorrect_contrast']}题实际有数学对错对比，{k['4']}题四答都正确。", "", "回答级复核记录见 `answer_correctness_reaudit_final.jsonl`，统计见 `answer_correctness_reaudit_final_summary.json`。历史文件保持原样；这里是单独的复核结果。", ""]
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
