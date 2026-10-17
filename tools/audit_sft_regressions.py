"""Recount saved Base/SFT v3 pairs; inspect fixed cases without generating answers."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "datasets/math/test_l2_l3_remaining_v1/combined_analysis_v32_final/math_v32"
OUTPUT = ROOT / "docs/SFT_REGRESSION_REVIEW.json"
CASES = [
    ("algebra", 82, "位值建模错误", "将AMC10、AMC12的五位数相加误当数字求和。",
     "用10000A+1000M+100C表示高位，得到100A+10M+C=617，A+M+C=14。"),
    ("algebra", 141, "遗漏边界条件", "将正整数因子的乘积一概判为合数，漏掉一个因子为1；还把n=1/2时的乘积0写成1。",
     "n=3时(n-1)(n-2)=2；n>=4时两因子都大于1，共1个正整数解。"),
    ("counting_and_probability", 69, "擅自增加题目条件", "原题仅要求第一字母为元音且不重复，却限制后两字母为辅音。",
     "各位置选择数为5、25、24、4、9、8，乘积864000。"),
    ("counting_and_probability", 79, "公式正确但数值计算错误", "列出6!*5!*3!*3!，最终乘积却写成4147200。",
     "720*120*6*6=3110400；应区分建模正确与执行计算错误。"),
    ("number_theory", 25, "素因子指数补齐与回代缺失", "将720=2^4*3^2*5补成立方时少补一个2，且所报立方根未回代成立。",
     "最小乘数为2^2*3*5^2=300，720*300=216000=60^3。"),
    ("prealgebra", 146, "提取公因式错误", "将3*2^20+2^21错误化成(3+1)*2^21。",
     "应为(3+2)*2^20，除以2^17后为40。"),
    ("precalculus", 35, "公式适用条件错误", "将三角形ABM当作直角三角形，用AB/BM求cos角ABM。",
     "AM=BM=13，余弦定理给(10^2+13^2-13^2)/(2*10*13)=5/13。"),
    ("precalculus", 83, "列式正确但解方程错误", "已列a+2c=1、2a+c=-4，却给出(-2,3)，没有代回检验。",
     "联立得(a,c)=(-3,2)，满足两条方程；(-2,3)不满足。"),
]


def load(name: str) -> tuple[dict, str]:
    path = SOURCE / f"{name}.jsonl"
    raw = path.read_bytes()
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    result = {}
    for row in rows:
        key = row["row_id"]
        if key in result:
            raise ValueError(f"Duplicate ID in {name}: {key}")
        score = row["audited_score_v3"]
        if score["schema"] != "audited-final-answer-v3.2" or type(score["correct"]) is not bool:
            raise ValueError(f"Unexpected saved score schema: {name}/{key}")
        for flag in ("truncated", "format_eos"):
            if type(row[flag]) is not bool:
                raise ValueError(f"Expected boolean {flag}: {name}/{key}")
        result[key] = row
    return result, hashlib.sha256(raw).hexdigest()


def main() -> None:
    base, base_hash = load("base")
    sft, sft_hash = load("sft_v3")
    if base.keys() != sft.keys():
        raise ValueError("Base/SFT ID sets differ")
    for key in base:
        for field in ("question", "gold"):
            if field in base[key] and field in sft[key] and base[key][field] != sft[key][field]:
                raise ValueError(f"Mismatched {field}: {key}")
    correct = lambda row: row["audited_score_v3"]["correct"]
    regression = [sft[key] for key in base if correct(base[key]) and not correct(sft[key])]
    improved = sum(not correct(base[key]) and correct(sft[key]) for key in base)
    cases = []
    for subject, index, category, observation, check in CASES:
        key = f"math:{subject}/test-00000-of-00001.parquet:{index}"
        b, s = base[key], sft[key]
        if not (correct(b) and not correct(s) and s["format_eos"] and not s["truncated"]):
            raise ValueError(f"Review case no longer meets selection conditions: {key}")
        cases.append({
            "row_id": key, "category": category, "observation": observation,
            "mathematical_check": check, "gold": s.get("gold", b.get("gold")),
            "sft_prediction": s["audited_score_v3"]["prediction"],
            "base_correct": True, "sft_correct": False,
            "sft_format_eos": True, "sft_truncated": False,
        })
    payload = {
        "schema": "sft-regression-review-v1", "review_date": "2026-10-12",
        "method": "Offline saved-score pairing plus model-assisted inspection of eight deliberately selected cases; no new inference or rescoring.",
        "case_selection": "Illustrative formatted, nontruncated regressions; not a random sample or an estimate of semantic category frequencies.",
        "sources": {
            str((SOURCE / "base.jsonl").relative_to(ROOT)).replace("\\", "/"): base_hash,
            str((SOURCE / "sft_v3.jsonl").relative_to(ROOT)).replace("\\", "/"): sft_hash,
        },
        "counts": {
            "paired_questions": len(base), "base_correct": sum(map(correct, base.values())),
            "sft_correct": sum(map(correct, sft.values())), "improved": improved,
            "regressed": len(regression), "regressed_sft_truncated": sum(r["truncated"] for r in regression),
            "regressed_sft_format_eos": sum(r["format_eos"] for r in regression),
            "regressed_sft_score_status": dict(Counter(r["audited_score_v3"]["status"] for r in regression)),
        },
        "reviewed_cases": cases,
        "limitations": [
            "Automatic final-answer scores, not independent certification of all derivations.",
            "Observed errors do not establish which SFT examples or hyperparameters caused them.",
            "Known benchmark used for retrospective feedback; do not treat it as a fresh blind test.",
        ],
    }
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["counts"], ensure_ascii=True))
    print(f"Saved {OUTPUT.relative_to(ROOT)}; reviewed {len(cases)} fixed cases")


if __name__ == "__main__":
    main()
