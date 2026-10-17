"""Validate saved verdicts, never silently re-grade historical answers."""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import unicodedata

SCHEMA = "audited-final-answer-v3.2"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path):
    rows = []
    with Path(path).open(encoding="utf-8-sig") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{number}: invalid JSON") from error
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{number}: expected an object")
            rows.append(row)
    if not rows:
        raise ValueError(f"{path}: empty dataset")
    return rows


def canonical(text):
    # Same rule as historical prepare_bc.py; exact normalization, not semantic dedup.
    text = unicodedata.normalize("NFKC", text).casefold()
    text = re.sub(r"\\(?:left|right|big|Big|,|;|!|\(|\)|\[|\])", "", text)
    text = text.replace(r"\ldots", "...").replace(r"\cdots", "...")
    return re.sub(r"[^\w\\+*/=<>.^-]", "", text)


def question(row):
    for key in ("question", "problem"):
        if isinstance(row.get(key), str) and row[key].strip():
            return row[key]
    for key in ("messages", "prompt"):
        messages = row.get(key)
        if isinstance(messages, list):
            for message in messages:
                if isinstance(message, dict) and message.get("role") == "user":
                    content = message.get("content")
                    if isinstance(content, str) and content.strip():
                        return content
    raise ValueError("Missing nonempty question/problem/user message")


def audit_splits(paths):
    """Collect every duplicate and train/evaluation exact overlap; no content export."""
    seen, summary, issues = {}, {}, []
    for name, path in paths.items():
        rows = read_jsonl(path)
        keys = []
        for number, row in enumerate(rows, 1):
            key = canonical(question(row))
            if not key:
                raise ValueError(f"{name}:{number}: empty normalized question")
            keys.append(key)
        counts = Counter(keys)
        duplicates = sum(count - 1 for count in counts.values())
        if duplicates:
            issues.append(f"{name}: {duplicates} duplicate rows")
        seen[name] = set(keys)
        summary[name] = {"rows": len(rows), "unique": len(counts),
                         "duplicates": duplicates, "sha256": sha256(path)}
    overlaps = []
    names = list(seen)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            count = len(seen[left] & seen[right])
            overlaps.append({"left": left, "right": right, "count": count})
            if count:
                issues.append(f"{left}/{right}: {count} overlapping questions")
    return {"ok": not issues, "splits": summary, "overlaps": overlaps,
            "issues": issues, "scope": "Normalized exact matches only; no semantic or pretraining leakage certification"}


def validate_predictions(rows):
    indexed = {}
    for row in rows:
        rid = row.get("row_id")
        if not isinstance(rid, str) or not rid or rid in indexed:
            raise ValueError(f"Missing or duplicate row_id: {rid!r}")
        question(row)
        if not isinstance(row.get("gold"), str) or not row["gold"].strip():
            raise ValueError(f"{rid}: missing gold")
        if not isinstance(row.get("completion"), str):
            raise ValueError(f"{rid}: missing completion")
        score = row.get("audited_score_v3", {})
        if score.get("schema") != SCHEMA:
            raise ValueError(f"{rid}: expected scorer {SCHEMA}")
        if type(score.get("correct")) is not bool or not isinstance(score.get("status"), str):
            raise ValueError(f"{rid}: invalid verdict")
        if score["correct"] != (score["status"] == "correct"):
            raise ValueError(f"{rid}: inconsistent status/correct")
        for key in ("eos", "truncated", "repeated_20_word_span_3_times"):
            if type(row.get(key)) is not bool:
                raise ValueError(f"{rid}: {key} must be boolean")
        if row["eos"] and row["truncated"]:
            raise ValueError(f"{rid}: EOS and truncation conflict")
        if type(row.get("tokens")) is not int or row["tokens"] < 0:
            raise ValueError(f"{rid}: invalid token count")
        indexed[rid] = row
    if not indexed:
        raise ValueError("Empty prediction set")
    return indexed


def wilson(successes, total):
    """95% Wilson interval for descriptive binomial uncertainty."""
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [max(0, center - half), min(1, center + half)]


def metrics(rows):
    rows = list(rows)
    if not rows:
        raise ValueError("Cannot summarize an empty cohort")
    statuses = Counter(row["audited_score_v3"]["status"] for row in rows)
    correct = statuses["correct"]
    return {"total": len(rows), "correct": correct, "accuracy": correct / len(rows),
            "accuracy_ci95_wilson": wilson(correct, len(rows)),
            "unconfirmed": sum(n for s, n in statuses.items() if s not in ("correct", "incorrect")),
            "status_counts": dict(statuses),
            "eos": sum(row["eos"] for row in rows),
            "truncated": sum(row["truncated"] for row in rows),
            "repeated": sum(row["repeated_20_word_span_3_times"] for row in rows),
            "mean_tokens": statistics.mean(row["tokens"] for row in rows),
            "median_tokens": statistics.median(row["tokens"] for row in rows)}


def paired(left, right):
    if left.keys() != right.keys():
        raise ValueError("Paired comparison requires identical question IDs")
    gain = loss = both = 0
    for rid, a in left.items():
        b = right[rid]
        if (question(a), a["gold"]) != (question(b), b["gold"]):
            raise ValueError(f"{rid}: question/reference mismatch")
        ac, bc = a["audited_score_v3"]["correct"], b["audited_score_v3"]["correct"]
        gain += not ac and bc
        loss += ac and not bc
        both += ac and bc
    return {"total": len(left), "gain": gain, "loss": loss, "net": gain - loss,
            "delta_percentage_points": 100 * (gain - loss) / len(left), "both_correct": both}


def build_report(root, registry):
    datasets, sources = {}, []
    for name, spec in registry["datasets"].items():
        indexed = {}
        for model, source in spec["models"].items():
            path = root / source["path"]
            actual = sha256(path)
            if actual != source["sha256"]:
                raise ValueError(f"Hash mismatch: {source['path']}")
            rows = read_jsonl(path)
            if len(rows) != spec["expected_rows"]:
                raise ValueError(f"{name}/{model}: unexpected row count")
            indexed[model] = validate_predictions(rows)
            sources.append({"dataset": name, "model": model, **source})
        base = indexed["base"]
        # Validate all models even when no pair-specific result is requested.
        comparisons = {model: paired(base, rows) for model, rows in indexed.items() if model != "base"}
        cohorts = {"all": set(base)}
        if "remaining_ids" in spec:
            source = spec["remaining_ids"]
            path = root / source["path"]
            if sha256(path) != source["sha256"]:
                raise ValueError("Remaining subset hash mismatch")
            ids = [row["row_id"] for row in read_jsonl(path)]
            if len(ids) != 1824 or len(set(ids)) != 1824 or not set(ids) <= base.keys():
                raise ValueError("Invalid MATH remaining1824 membership")
            cohorts.update(remaining1824=set(ids), previous200=set(base) - set(ids))
            if len(cohorts["previous200"]) != 200:
                raise ValueError("Invalid previous200 cohort")
            sources.append({"dataset": name, "model": "cohort_membership", **source})
        datasets[name] = {
            "metrics": {cohort: {model: metrics(row for rid, row in rows.items() if rid in ids)
                                    for model, rows in indexed.items()} for cohort, ids in cohorts.items()},
            "paired_vs_base": comparisons,
            "cohort_paired_vs_base": {cohort: {
                model: paired({rid: row for rid, row in base.items() if rid in ids},
                              {rid: row for rid, row in rows.items() if rid in ids})
                for model, rows in indexed.items() if model != "base"} for cohort, ids in cohorts.items()}}
    return {"schema": "r1-lab-report-v1", "scorer": SCHEMA, "datasets": datasets,
            "sources": sources, "scope": "Recomputed from saved verdicts; no new inference or mathematical re-grading. Unconfirmed remains in denominator. Wilson intervals are descriptive and do not cover scorer error, model seed variability, test reuse or pretraining contamination."}
