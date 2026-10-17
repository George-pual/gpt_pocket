"""Recover clearly stated answers from the first-pass unresolved responses."""
from __future__ import annotations

import collections
import json
import multiprocessing
import re
from pathlib import Path

from math_verification import normalize_final_candidate

ROOT = Path(__file__).resolve().parents[1] / "datasets/math/base_self600_20261015/grpo_rescreen_v4_results"
SOURCE = ROOT / "answer_correctness_reaudit.jsonl"
OUT = ROOT / "answer_correctness_reaudit_refined.jsonl"
SUMMARY = ROOT / "answer_correctness_reaudit_refined_summary.json"


def _grade_candidate(candidate: str, gold: str, question: str):
    from math_verification import _grade_pair
    return _grade_pair(candidate, str(gold), None, question)


def _cut_at_task_drift(text: str) -> str:
    markers = [
        r"Create a problem based on the given working out and solution",
        r"<\s*start_problem\s*>",
        r"Given the equation below, solve for",
        r"Question\s*:\s*(?!.*original)",
    ]
    positions = [m.start() for pat in markers if (m := re.search(pat, text, re.I))]
    return text[:min(positions)] if positions else text


def _boxes(text: str):
    out = []
    for match in re.finditer(r"\\boxed\s*\{", text):
        start, depth = match.end(), 1
        for pos in range(start, len(text)):
            if text[pos] == "{":
                depth += 1
            elif text[pos] == "}":
                depth -= 1
                if depth == 0:
                    out.append(text[start:pos].strip())
                    break
    return out


def propose(row):
    text = _cut_at_task_drift(row["completion"])
    # Prefer an explicit solution region that precedes any later unrelated task.
    regions = list(re.finditer(r"<\s*SOLUTION\s*>(.*?)<\s*/\s*SOLUTION\s*>", text, re.I | re.S))
    candidate = regions[-1].group(1).strip() if regions else (row.get("candidate_final_answer") or text)
    # Also accept incomplete/misspelled wrappers such as <SOLUTION> 56/25
    # or <solution>7<\solution>; these commonly caused false parse failures.
    loose_regions = list(re.finditer(r"(?:<\s*SOLUTION\s*>|\bSOLUTION\s*>)(.*?)(?=<\s*/?\s*(?:SOLUTION|end_working_out|start_working_out)|$)", text, re.I | re.S))
    loose = [m.group(1).strip() for m in loose_regions if m.group(1).strip()]
    if loose:
        candidate = loose[-1]
    # Some generations have an empty, misspelled, or unclosed SOLUTION tag.
    # In those cases the original response's final lines carry the answer.
    if not candidate.strip() or len(candidate.strip()) < 2:
        candidate = text
    # A final boxed result is an explicit answer even if tags around it are broken.
    boxes = _boxes(text)
    if boxes:
        candidate = boxes[-1]
    # Strip control markup that can make the math parser stop at the wrong tag.
    candidate = re.sub(r"<\s*/?\s*(?:SOLUTION|start_working_out|end_working_out)\s*>", " ", candidate, flags=re.I)
    candidate = re.sub(r"<\|[^|]+\|>|\[(?:extract_tex|extract_itex)\]", " ", candidate, flags=re.I)
    candidate = candidate.replace("[/extract_tex]", " ").replace("[/extract_itex]", " ")
    candidate = re.sub(r"<[^>]{1,80}>", " ", candidate)
    candidate = re.sub(r"<mfrac><mn>\s*([^<]+)\s*</mn><mn>\s*([^<]+)\s*</mn></mfrac>", r"\\frac{\1}{\2}", candidate, flags=re.I)
    candidate = candidate.strip().strip("*` ")

    normalized, rule, error = normalize_final_candidate(candidate, row["problem"])
    if normalized is not None and normalized.strip():
        return normalized, "normalized_candidate:" + str(rule)

    # Recover a number only from a complete, answer-like sentence at the end.
    scalar = r"([+-]?(?:\\frac\s*\{[^{}]+\}\s*\{[^{}]+\}|\\sqrt\s*\{[^{}]+\}|\d+(?:\.\d+)?(?:\s*/\s*\d+)?))"
    patterns = [
        rf"(?:The answer is|Final answer is|The solution is|Therefore,|Thus,|Hence,|So,)\s*(?:[^\n.!?]*?\s+)?{scalar}(?:\s*(?:%|percent|degrees?|°|marbles?|ways|orders|students|units?))?\s*[.!]*\s*$",
        rf"(?:has|is|equals?|=)\s*{scalar}\s*(?:%|percent|degrees?|°|marbles?|ways|orders|students|units?)?\s*[.!]*\s*$",
        rf"^\s*{scalar}\s+(?:different\s+)?(?:ways|orders|marbles|students)\.?\s*$",
    ]
    for pattern in patterns:
        matches = list(re.finditer(pattern, candidate, re.I | re.S))
        if matches:
            return matches[-1].group(1).strip(), "terminal_answer_sentence"
    # A final equation is usable only when it is the last mathematical claim.
    expression = r"(\\frac\s*\{[^{}]+\}\s*\{[^{}]+\}|\\sqrt\s*\{[^{}]+\}|[+-]?\d+(?:\.\d+)?(?:\s*/\s*\d+)?)(?:\s*(?:\\circ|°|%))?"
    equations = list(re.finditer(r"(?:^|[\n\s])(?:[A-Za-z])\s*=\s*" + expression, text))
    if equations:
        tail = candidate[equations[-1].end():]
        if not re.search(r"\b(?:or|alternatively|but|however|maybe)\b", tail, re.I):
            result = equations[-1].group(1).strip().rstrip("$ ")
            return result, "last_terminal_equation"
    # Recover the final scalar in an answer-like concluding sentence.
    scalar_matches = list(re.finditer(scalar, text))
    if scalar_matches:
        last = scalar_matches[-1]
        tail = text[last.end():]
        preceding = text[max(0, last.start() - 100):last.start()]
        if len(tail) <= 120 and re.search(r"\b(?:orders|ways|marbles|students|percent|answer|ratio|value|degrees?|fraction|digits?|sum|angle|number|prime)\b|%|°", tail + preceding, re.I):
            return last.group(1).strip(), "terminal_scalar_phrase"
    return None, error or "no_safe_answer_recovery"


def main():
    rows = [json.loads(x) for x in SOURCE.read_text(encoding="utf-8").splitlines() if x.strip()]
    counts = collections.Counter()
    pool = None
    ctx = multiprocessing.get_context("spawn")
    try:
        with OUT.open("w", encoding="utf-8", newline="\n") as output:
            for row in rows:
                first = row.get("candidate_verification") or {}
                if first.get("status") in ("correct", "incorrect"):
                    row["refined_candidate"] = None
                    row["refined_verification"] = None
                    row["refinement_note"] = "first_pass_clear"
                else:
                    candidate, note = propose(row)
                    row["refined_candidate"] = candidate
                    row["refinement_note"] = note
                    if candidate is not None:
                        if pool is None:
                            pool = ctx.Pool(processes=1)
                        pending = pool.apply_async(_grade_candidate, (candidate, row["gold"], row["problem"]))
                        try:
                            row["refined_verification"] = pending.get(timeout=15)
                        except multiprocessing.TimeoutError:
                            row["refined_verification"] = {"status": "verification_timeout", "correct": False,
                                                           "error": "worker_timeout"}
                            pool.terminate(); pool.join(); pool = None
                    else:
                        row["refined_verification"] = None
                v = row.get("refined_verification") or row.get("candidate_verification") or {}
                status = v.get("status", "unresolved")
                counts[status] += 1
                if row.get("refined_candidate") is not None and status not in ("correct", "incorrect"):
                    counts["proposed_but_unresolved"] += 1
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
                output.flush()
                if sum(counts[k] for k in ("correct", "incorrect", "unresolved", "prediction_parse_failed", "verification_timeout")) % 50 == 0:
                    print(f"Refined {sum(counts.values())}/{len(rows)}", flush=True)
    finally:
        if pool is not None:
            pool.close(); pool.join()
    summary = {"rows": len(rows), "counts": dict(counts), "output": OUT.name,
               "note": "Automated second-pass cleanup only. Remaining unclear answers require manual judgment."}
    SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
