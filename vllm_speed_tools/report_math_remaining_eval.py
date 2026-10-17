"""Validate downloaded results and report new and reused MATH subsets separately."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics


ARMS = {
    "base": "Base", "sft_v3": "SFT v3", "grpo1024": "GRPO-1024",
    "grpo1280": "GRPO-1280", "grpo1536": "GRPO-1536",
}


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metrics(rows):
    counts = Counter(r["audited_score_v3"]["status"] for r in rows)
    n = len(rows)
    return {
        "total": n,
        "correct": sum(bool(r["audited_score_v3"]["correct"]) for r in rows),
        "unconfirmed": sum(v for k, v in counts.items() if k not in ("correct", "incorrect")),
        "eos": sum(bool(r["eos"]) for r in rows),
        "truncated": sum(bool(r["truncated"]) for r in rows),
        "repeated": sum(bool(r["repeated_20_word_span_3_times"]) for r in rows),
        "format_eos": sum(bool(r["format_eos"]) for r in rows),
        "mean_tokens": statistics.mean(r["tokens"] for r in rows),
        "median_tokens": statistics.median(r["tokens"] for r in rows),
        "status_counts": dict(counts),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--scores", type=Path, required=True)
    p.add_argument("--arms", default=",".join(ARMS))
    a = p.parse_args()
    arms = a.arms.split(",")
    assert len(arms) == len(set(arms)) and set(arms) <= set(ARMS)
    questions = read_rows(a.bundle / "questions_all.jsonl")
    remaining = read_rows(a.bundle / "questions_remaining.jsonl")
    assert len(questions) == 2024 and len(remaining) == 1824
    qmap = {r["row_id"]: r for r in questions}
    new_ids = {r["row_id"] for r in remaining}
    assert len(qmap) == 2024 and len(new_ids) == 1824
    manifest = json.loads((a.scores / "manifest.json").read_text(encoding="utf-8"))
    config = json.loads((a.results / "evaluation_config.json").read_text(encoding="utf-8"))
    expected_schema = "audited-final-answer-v3.1"
    assert manifest["schema"] == expected_schema
    assert manifest["scorer_sha256"] == sha(a.bundle / "scoring_v31/audited_answer_scoring.py")
    assert manifest["scorer_sha256"] == sha(a.scores / "audited_answer_scoring.py")
    assert manifest["runner_sha256"] == sha(a.scores / "math_remaining_eval.py")
    assert manifest["bundle_sha256"] == config["bundle_manifest_sha256"] == sha(a.bundle / "manifest.json")
    assert config["sampling"] == dict(n=1, temperature=0.0, top_p=1.0, top_k=-1,
                                     max_tokens=3894, eos_token_id=151643, ignore_eos=False)
    checks = json.loads((a.scores / "reference_self_checks.json").read_text(encoding="utf-8"))
    assert checks["total"] == 2024 and checks["failed"] == 0
    summary = json.loads((a.scores / "summary.json").read_text(encoding="utf-8"))
    grouped, checked_hashes = {}, {}
    all_scored = {}
    for arm in arms:
        rows = read_rows(a.scores / f"{arm}.jsonl")
        assert [r["row_id"] for r in rows] == [r["row_id"] for r in questions]
        assert len({r["row_id"] for r in rows}) == 2024
        old = read_rows(a.bundle / "reused" / f"{arm}.jsonl")
        old_by_id = {r["row_id"]: r for r in old}
        paths = sorted((a.results / arm).glob("chunk_*.jsonl"))
        raw = [r for path in paths for r in read_rows(path)]
        assert len(paths) == 29 and len(raw) == 1824
        assert [(r["row_id"], r["question"], r["gold"]) for r in raw] == [
            (q["row_id"], q["problem"], q["answer"]) for q in remaining]
        raw_by_id = {r["row_id"]: r for r in raw}
        for path in paths:
            candidates = [h for name, h in manifest["inputs"].items()
                          if name.replace("\\", "/").endswith(f"/{arm}/{path.name}")]
            assert len(candidates) == 1 and candidates[0] == sha(path), path
            checked_hashes[f"{arm}/{path.name}"] = candidates[0]
        for r in rows:
            q = qmap[r["row_id"]]
            assert (r["question"], r["gold"], r["level"]) == (
                q["problem"], q["answer"], q["level"])
            if "subject" in r:
                assert r["subject"] == q["type"]
            assert bool(r["reused"]) == (r["row_id"] not in new_ids)
            assert r["audited_score_v3"]["schema"] == expected_schema
            assert not r["reference_requires_review"]
            source = old_by_id[r["row_id"]] if r["reused"] else raw_by_id[r["row_id"]]
            for key in ("completion", "tokens", "eos", "truncated"):
                assert r[key] == source[key], (arm, r["row_id"], key)
            assert 0 <= r["tokens"] <= 3894
        grouped[arm] = {
            "all": metrics(rows),
            "remaining": metrics([r for r in rows if not r["reused"]]),
            "reused": metrics([r for r in rows if r["reused"]]),
            "remaining_by_level": {level: metrics([r for r in rows if not r["reused"] and r["level"] == level])
                                   for level in ("Level 2", "Level 3")},
        }
        for subset in ("all", "remaining", "reused"):
            recomputed = grouped[arm][subset]
            for key in ("total", "correct", "eos", "truncated", "repeated", "format_eos", "status_counts"):
                assert recomputed[key] == summary[arm][subset][key], (arm, subset, key)
            for key in ("mean_tokens", "median_tokens"):
                assert abs(recomputed[key] - summary[arm][subset][key]) < 1e-9
        all_scored[arm] = rows
    review = read_rows(a.scores / "review_queue.jsonl")
    for arm in arms:
        assert sum(r["model"] == arm for r in review) == grouped[arm]["all"]["unconfirmed"]
    report = ["# MATH Level 2/3 五模型对比", "",
        "本轮只评测已有权重，没有训练。原 200 题回答复用，每模型新增生成 1824 题。",
        "主要比较下表的新增 1824 题；原 200 题曾用于反馈，合并的 2024 题只作为补充汇总。",
        "官方 Level 2/3 test 原有 2025 题，排除 1 条与训练题规范化精确重叠后得到 2024 题。",
        "", "## 新增 1824 题（主要结果）", "",
        "|模型|自动判对|判对率|未确认|EOS 率|截断率|重复率|平均 tokens|格式且 EOS|",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for arm in arms:
        x = grouped[arm]["remaining"]; n = x["total"]
        report.append(f"|{ARMS[arm]}|{x['correct']}/{n}|{x['correct']/n:.2%}|{x['unconfirmed']}|"
                      f"{x['eos']/n:.2%}|{x['truncated']/n:.2%}|{x['repeated']/n:.2%}|"
                      f"{x['mean_tokens']:.1f}|{x['format_eos']/n:.2%}|")
    report += ["", "## 新增题的难度分组", "", "|模型|Level 2（794题）|Level 3（1030题）|",
               "|---|---:|---:|"]
    for arm in arms:
        l2, l3 = [grouped[arm]["remaining_by_level"][level] for level in ("Level 2", "Level 3")]
        report.append(f"|{ARMS[arm]}|{l2['correct']}/{l2['total']}（{l2['correct']/l2['total']:.2%}）|"
                      f"{l3['correct']}/{l3['total']}（{l3['correct']/l3['total']:.2%}）|")
    report += ["", "## 合并此前 200 题（补充汇总）", "",
               "|模型|原 200 题自动判对|全部 2024 题自动判对|全部判对率|全部未确认|",
               "|---|---:|---:|---:|---:|"]
    for arm in arms:
        x, old = grouped[arm]["all"], grouped[arm]["reused"]
        report.append(f"|{ARMS[arm]}|{old['correct']}/200|{x['correct']}/2024|"
                      f"{x['correct']/2024:.2%}|{x['unconfirmed']}|")
    report += ["", "## 口径与复现", "",
        "- 五模型使用相同题目、原自定义 prompt、greedy 解码、temperature=0、top_p=1、top_k=-1、max_tokens=3894、EOS=151643。配置详见 results/evaluation_config.json。",
        "- 统一使用冻结的 audited-final-answer-v3.1，math-verify=0.8.0；2024 份参考答案自检通过，原五组 200 题的判对结论未改变。版本和源码哈希详见 scored_v31/manifest.json。",
        "- 未确认包括无法提取、无法解析或不支持的答案，按未判对计入自动评分分母；它们不能全部解释为数学错误。详见 scored_v31/review_queue.jsonl。没有按模型单独放宽规则或人工改分。",
        "- EOS 是生成结束标记，格式率另要求工作区间与 SOLUTION 标签完整闭合。两者均不能单独证明没有过度推理；回答长度、截断和重复率一起参考。",
        "- 重复率指同一空白分词的连续 20 词片段出现至少三次，是启发式指标；它不会捕捉所有语义重复。",
        "- 已与已知 SFT/GRPO 训练题做规范化精确去重，没有语义去重，也无法排除 Base 预训练污染。不能将本子集结果外推为所有数学题能力或每一步推理的正确率。",
        "- 原始回答、评分记录和生成日志完整保留。下载后的 145 个分批文件哈希、五组题序/题面/答案、旧回答复用、评分器版本及汇总重算核验通过；见 comparison_validation.json。",
        "", "评分命令：", "", "```text",
        "python scoring_v31/math_remaining_eval.py score --bundle . --output results --score-output <新评分目录>",
        "python <工作区>/vllm_speed_tools/report_math_remaining_eval.py --bundle . --results results --scores scored_v31",
        "```", ""]
    if len(arms) != 5:
        report.insert(2, "此文件为已完成模型的中间结果，尚非五模型最终比较。")
        report = [line.replace("145 个分批文件", f"{29*len(arms)} 个分批文件") for line in report]
    (a.scores.parent / "COMPARISON.md").write_text("\n".join(report), encoding="utf-8")
    validation = dict(validated_at_utc=datetime.now(timezone.utc).isoformat(), arms=arms,
                      rows_per_arm=2024, new_per_arm=1824, reused_per_arm=200,
                      scorer_sha256=manifest["scorer_sha256"], input_hashes=checked_hashes,
                      checks="passed", metrics=grouped)
    (a.scores.parent / "comparison_validation.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({arm: grouped[arm]["remaining"] for arm in arms}, ensure_ascii=True))


if __name__ == "__main__":
    main()
