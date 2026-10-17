"""Recheck unparsed answers in the 190 + 366 GRPO screening pools.

Candidate selection is restricted to an explicit final region/declaration. The
script never searches reasoning for a value matching the gold answer and never
modifies the original screening verdicts or training pools.
"""
from __future__ import annotations

import collections
import json
import multiprocessing
import re
import sys
from pathlib import Path

from math_verification import verify_final_answer

ROOT = Path(__file__).resolve().parents[1] / "datasets/math/base_self600_20261015"
RESULTS = ROOT / "grpo_rescreen_v4_results"
REMOTE = RESULTS / "remote"
GROUPS = REMOTE / "screen_current_grpo/grpo_final/groups"
OUT = RESULTS / "answer_correctness_reaudit.jsonl"
SUMMARY = RESULTS / "answer_correctness_reaudit_summary.json"


def _grade_candidate(candidate: str, gold: str, question: str):
    # Reuse a dedicated worker process; starting a fresh Windows process for
    # every answer made this audit needlessly slow. The parent bounds each
    # request and replaces the worker if Math-Verify stalls.
    from math_verification import _grade_pair
    return _grade_pair(candidate, str(gold), None, question)


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def balanced_boxed(text: str):
    boxes = []
    for match in re.finditer(r"\\boxed\s*\{", text):
        start = match.end()
        depth = 1
        for pos in range(start, len(text)):
            if text[pos] == "{":
                depth += 1
            elif text[pos] == "}":
                depth -= 1
                if depth == 0:
                    boxes.append((match.start(), pos + 1, text[start:pos].strip()))
                    break
    return boxes


def solution_regions(text: str):
    pattern = re.compile(r"<\s*SOLUTION\s*>(.*?)<\s*/\s*SOLUTION\s*>", re.I | re.S)
    return [(match.start(), match.end(), match.group(1).strip()) for match in pattern.finditer(text)]


def choose_explicit_candidate(text: str):
    """Return a gold-blind candidate and why it was chosen; ambiguous cases stay open."""
    solutions = solution_regions(text)
    nonempty_solutions = [(start, end, value) for start, end, value in solutions if value]
    if nonempty_solutions:
        distinct = {re.sub(r"\s+", "", value) for _, _, value in nonempty_solutions}
        last_start, last_end, last_value = nonempty_solutions[-1]
        tail = text[last_end:].strip()
        # If generation switched to another question after an earlier answer,
        # do not treat the final unrelated response as an answer to this prompt.
        if re.search(r"(?:\n\s*(?:Question\s*:|Given the equation|Solve for the variable|What is the value of|What percent of|Please be sure to|Answer Choices))", tail, re.I):
            before = nonempty_solutions[0]
            first_tail = text[before[1]:]
            if re.search(r"(?:\n\s*(?:Question\s*:|Given the equation|Solve for the variable|What is the value of|What percent of|Please be sure to|Answer Choices))", first_tail, re.I):
                return {"candidate": None, "source": "solution_regions", "reason": "answer_followed_by_unrelated_question"}
        if len(distinct) > 1 and len(nonempty_solutions) == 1:
            return {"candidate": None, "source": "solution_regions", "reason": "conflicting_solution_regions"}
        return {"candidate": last_value, "source": "last_explicit_solution_region",
                "reason": "multiple_solution_regions" if len(nonempty_solutions) > 1 else None}

    boxes = balanced_boxed(text)
    if boxes:
        values = [value for _, _, value in boxes]
        last_start, last_end, last_value = boxes[-1]
        tail = text[last_end:].strip()
        if re.search(r"(?:\n\s*(?:Question\s*:|Given the equation|Solve for the variable|What is the value of|What percent of|Please be sure to|Answer Choices))", tail, re.I):
            return {"candidate": None, "source": "boxed_answer", "reason": "boxed_answer_followed_by_unrelated_question"}
        if len({re.sub(r"\s+", "", value) for value in values}) > 1:
            # Keep conflicts unresolved unless there is an explicit terminal
            # answer declaration after the final box.
            return {"candidate": None, "source": "boxed_answers", "reason": "conflicting_boxed_answers"}
        return {"candidate": last_value, "source": "explicit_boxed_answer", "reason": None}

    # Use only the final explicit answer declaration, never an arbitrary number
    # from reasoning. Its short tail must be a single answer statement.
    matches = list(re.finditer(
        r"(?:Final Answer|Final answer|Answer|The answer is|Therefore|Thus|Hence|So the answer is|The value is|The value of [^\n:]{1,90} is)\s*[:：]?\s*([^\n]{1,180})",
        text, re.I))
    if matches:
        match = matches[-1]
        candidate = match.group(1).strip().strip("*` ")
        candidate = re.sub(r"(?:<\s*/?\s*(?:end_working_out|start_working_out)\s*>|<\|[^|]+\|>)\s*$", "", candidate, flags=re.I).strip()
        if candidate and not re.search(r"\b(?:or|alternatively|maybe|not sure|cannot determine)\b", candidate, re.I):
            return {"candidate": candidate, "source": "terminal_answer_declaration", "reason": None}
    return {"candidate": None, "source": None, "reason": "no_explicit_final_answer_region"}


def main():
    source = {row["row_id"]: row for row in read_jsonl(REMOTE / "data/candidates1200.jsonl")}
    annotations = {row["row_id"]: row for row in read_jsonl(RESULTS / "annotated_scores.jsonl")}
    groups = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(GROUPS.glob("*.json"))]
    rows = []
    counts = collections.Counter()
    by_category = collections.defaultdict(collections.Counter)
    context = multiprocessing.get_context("spawn")
    pool = None
    try:
        with OUT.open("w", encoding="utf-8", newline="\n") as output:
            for group in groups:
                row_id = group["row_id"]
                category = annotations[row_id]["category"]
                if category not in ("mixed_math_protocol", "protocol_only"):
                    continue
                for sample in group["samples"]:
                    original = sample["verdict"]
                    if original["status"] != "extraction_failed":
                        continue
                    proposal = choose_explicit_candidate(sample["completion"])
                    record = {
                        "row_id": row_id,
                        "category": category,
                        "sample_index": sample["sample_index"],
                        "problem": source[row_id]["problem"],
                        "gold": source[row_id]["answer"],
                        "original_error": original.get("error"),
                        "completion": sample["completion"],
                        "candidate_source": proposal["source"],
                        "candidate_reason": proposal["reason"],
                        "candidate_final_answer": proposal["candidate"],
                        "manual_verdict": None,
                        "manual_note": None,
                    }
                    if proposal["candidate"] is not None:
                        if pool is None:
                            pool = context.Pool(processes=1)
                        pending = pool.apply_async(
                            _grade_candidate,
                            (proposal["candidate"], source[row_id]["answer"], source[row_id]["problem"]),
                        )
                        try:
                            grade = pending.get(timeout=15)
                            record["candidate_verification"] = {
                                "schema": "math-verify-final-answer-v4",
                                "answer_source": proposal["source"],
                                **grade,
                            }
                        except multiprocessing.TimeoutError:
                            record["candidate_verification"] = {
                                "status": "verification_timeout", "correct": False,
                                "error": "worker_timeout",
                            }
                            pool.terminate()
                            pool.join()
                            pool = None
                    else:
                        record["candidate_verification"] = None
                    rows.append(record)
                    output.write(json.dumps(record, ensure_ascii=False) + "\n")
                    output.flush()
                    counts["total_unparsed_answers"] += 1
                    counts["candidate_proposed"] += proposal["candidate"] is not None
                    counts["no_candidate"] += proposal["candidate"] is None
                    if record["candidate_verification"]:
                        status = record["candidate_verification"]["status"]
                        counts["candidate_" + status] += 1
                        by_category[category]["candidate_" + status] += 1
                    by_category[category]["total"] += 1
                    if counts["total_unparsed_answers"] % 50 == 0:
                        print(f"Audited {counts['total_unparsed_answers']}/724", flush=True)
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    assert len(rows) == counts["total_unparsed_answers"] == 724
    OUT.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    summary = {
        "status": "candidate_reaudit_complete_needs_final_adjudication",
        "counts": dict(counts),
        "by_category": {key: dict(value) for key, value in by_category.items()},
        "output": OUT.name,
        "training_started": False,
        "note": "Explicit final-answer candidates are independently regraded, but conflicting/off-task/incomplete responses still require direct review. Original scores are unchanged.",
    }
    SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
