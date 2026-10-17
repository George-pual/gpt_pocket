"""Final-answer grading only; never a proof or reasoning-step verifier.

No string fallback and no scanning arbitrary reasoning for a matching number.
Linux uses Math-Verify's timeouts; Windows uses a bounded spawn worker because
the library's nested multiprocessing timeout target cannot be pickled there.
"""
from __future__ import annotations

import importlib.metadata
import multiprocessing
import os
import re
from collections import Counter
from functools import lru_cache

SCHEMA = "math-verify-final-answer-v2"
MAX_ANSWER_CHARS = 2048
SCALAR = r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:\s*/\s*[+-]?\d+)?'
UNITS = r'(?:kilowatts?|watts?|feet|foot|meters?|metres?|centimeters?|centimetres?|hours?|minutes?|seconds?|dollars?|containers?)'


def _scalar_unit(value):
    return re.fullmatch(r'(' + SCALAR + r')\s+((?:square\s+|cubic\s+)?' + UNITS + r')\.?', value, re.I)


@lru_cache(maxsize=1)
def require_math_verify():
    try:
        import math_verify  # noqa: F401
        return importlib.metadata.version("math-verify")
    except ImportError as exc:
        raise RuntimeError(
            'Install math-verify==0.8.0 and antlr4-python3-runtime==4.13.2 '
            'from requirements-math-verify.txt; no numeric fallback is used.'
        ) from exc


def completion_text(value):
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return str(value.get("content", ""))
    if isinstance(value, (list, tuple)):
        return "\n".join(completion_text(item) for item in value)
    return ""


def extract_final_answer(text):
    """Return (answer, source, error); malformed/duplicate tags fail closed."""
    text = completion_text(text).strip()
    text = re.sub(r"(?:<\|endoftext\|>|<\|im_end\|>|</s>)\s*$", "", text).strip()
    if re.search(r"</?SOLUTION\b", text, re.I):
        matches = re.findall(r"<SOLUTION>(.*?)</SOLUTION>", text, re.I | re.S)
        if (len(matches) != 1 or len(re.findall(r"<SOLUTION>", text, re.I)) != 1
                or len(re.findall(r"</SOLUTION>", text, re.I)) != 1):
            return None, "solution", "malformed_or_multiple_solution"
        if not re.search(r"</SOLUTION>\s*$", text, re.I):
            return None, "solution", "content_after_solution"
        answer = matches[0].strip()
        return answer or None, "solution", None if answer else "empty_answer"
    boxes = list(re.finditer(r"\\boxed\s*\{", text))
    if boxes:
        start = boxes[-1].end()
        depth = 1
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    # Keep math context for tuples, sets and nested fractions.
                    answer = text[start:i].strip()
                    return answer or None, "boxed", None if answer else "empty_answer"
        return None, "boxed", "unclosed_boxed"
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines:
        match = re.fullmatch(r"(?:Final answer|Answer|最终答案|答案)\s*[:：]\s*(.+)", lines[-1], re.I)
        if match:
            return match[1].strip(), "answer_line", None
    # Deliberately no last-number/last-line fallback into arbitrary reasoning.
    return None, "none", "missing_final_answer"


def _normalize_answer(value, allow_units=False):
    """Normalize whole final answers, never search reasoning for a number."""
    value = value.strip()
    if value.startswith('$$') and value.endswith('$$'):
        value = value[2:-2].strip()
    elif value.startswith('$') and value.endswith('$'):
        value = value[1:-1].strip()
    elif ((value.startswith(r'\(') and value.endswith(r'\)'))
          or (value.startswith(r'\[') and value.endswith(r'\]'))):
        value = value[2:-2].strip()
    # Only recognize an entire clock value, with explicit meridiem.
    clock = re.fullmatch(r'(?:\\text\{)?\s*(\d{1,2}):(\d{2})\s*([ap])\.?\s*m\.?\s*\}?', value, re.I)
    if clock and 1 <= int(clock[1]) <= 12 and int(clock[2]) < 60:
        return r'\text{' + f'{int(clock[1])}:{clock[2]} {clock[3].lower()}.m.' + '}'
    # Plain sqrt notation has unambiguous mathematical meaning.
    previous = None
    while previous != value:
        previous = value
        value = re.sub(r'(?<![A-Za-z\\])sqrt\(([^()]*)\)', r'\\sqrt{\1}', value)
    # Unit suffixes are recognized as a whole scalar answer. Percent is
    # deliberately excluded: 100 and 100% are different quantities.
    scalar, units = SCALAR, UNITS
    if allow_units:
        # One complete declarative answer, not the last sentence of a proof.
        sentence = re.fullmatch(
            r'(?:The (?:probability|number|degree|value) [^.!?;\n]* is |'
            r'Positive value of \$?[A-Za-z]\$? is |The car travels |'
            r'[A-Z][a-z]+ will need )('
            + scalar + r'(?:\s+(?:new\s+|square\s+|cubic\s+)?' + units + r')?)\.?', value)
        if sentence:
            value = sentence[1].replace(' new ', ' ').removesuffix('.')
    unit_answer = re.fullmatch(r'(' + scalar + r')\s+(?:square\s+|cubic\s+)?' + units + r'\.?', value, re.I)
    if allow_units and unit_answer:
        return unit_answer[1]
    return value


def _parse_answer(value, timeout):
    from math_verify import LatexExtractionConfig, parse
    value = value.strip()
    if not value or len(value) > MAX_ANSWER_CHARS or "__" in value:
        return []
    value = _normalize_answer(value)
    # Parse the isolated answer as one LaTeX expression, including bare fractions
    # and symbols. Do not let the plain-number extractor salvage part of garbage.
    if value.startswith("$$") and value.endswith("$$"):
        value = value[2:-2]
    elif value.startswith("$") and value.endswith("$"):
        value = value[1:-1]
    elif ((value.startswith(r"\(") and value.endswith(r"\)"))
          or (value.startswith(r"\[") and value.endswith(r"\]"))):
        value = value[2:-2]
    return parse("$" + value + "$", extraction_config=[LatexExtractionConfig()],
                 fallback_mode="no_fallback", extraction_mode="first_match",
                 parsing_timeout=timeout, raise_on_error=True)


def _grade_pair(prediction, gold, timeout):
    from math_verify import verify
    from math_verify.errors import TimeoutException
    result = {"correct": False, "gold_parsed": False, "prediction_parsed": False,
              "status": "gold_parse_failed", "error": None}
    phase = "gold_parse"
    try:
        # Handle explicitly stated units before the symbolic parser can treat
        # their letters as variables or discard them.
        gold_unit = _scalar_unit(_normalize_answer(gold))
        if gold_unit:
            pred_unit = _scalar_unit(_normalize_answer(prediction))
            if not pred_unit or pred_unit[2].lower() != gold_unit[2].lower():
                result.update(status='incorrect')
                return result
            prediction, gold = pred_unit[1], gold_unit[1]
        reference = _parse_answer(gold, timeout)
        result["gold_parsed"] = bool(reference)
        if not reference:
            return result
        phase = "prediction_parse"
        # A bare numeric reference may omit the unit requested by its problem.
        # Do not erase units on both sides (2 feet must not equal 2 meters).
        normalized_gold = _normalize_answer(gold)
        if re.fullmatch(r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:\s*/\s*[+-]?\d+)?', normalized_gold):
            prediction = _normalize_answer(prediction, allow_units=True)
        candidate = _parse_answer(prediction, timeout)
        result["prediction_parsed"] = bool(candidate)
        if not candidate:
            result["status"] = "prediction_parse_failed"
            return result
        phase = "verify"
        # The LaTeX parser sometimes creates i with incompatible assumptions
        # depending on term order. Resolve only closed numeric complex answers;
        # never reinterpret i in an expression containing other variables.
        from sympy import Basic, I
        def canonical_complex(items):
            out = []
            for item in items:
                if isinstance(item, Basic) and item.free_symbols and all(s.name == 'i' for s in item.free_symbols):
                    item = item.xreplace({s: I for s in item.free_symbols})
                out.append(item)
            return out
        reference, candidate = canonical_complex(reference), canonical_complex(candidate)
        # Ten decimal places rather than the package's default six. This is
        # configured numerical equivalence, not a claim of exact proof checking.
        result["correct"] = bool(verify(reference, candidate, strict=True,
            float_rounding=10, numeric_precision=15, timeout_seconds=timeout,
            raise_on_error=True))
        result["status"] = "correct" if result["correct"] else "incorrect"
    except TimeoutException:
        result.update(status=phase + "_timeout", error="TimeoutException")
    except Exception as exc:
        result.update(status=phase + "_error", error=type(exc).__name__)
    return result


def _worker(connection, prediction, gold):
    try:
        import logging
        logging.getLogger("math_verify.parser").setLevel(logging.ERROR)
        logging.getLogger("math_verify.grader").setLevel(logging.ERROR)
        connection.send(_grade_pair(prediction, gold, None))
    finally:
        connection.close()


@lru_cache(maxsize=4096)
def _bounded_grade(prediction, gold):
    if os.name != "nt":
        return _grade_pair(prediction, gold, 5)
    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(target=_worker, args=(sender, prediction, gold))
    try:
        process.start()
        sender.close()
        if receiver.poll(15):
            try:
                return receiver.recv()
            except EOFError:
                return {"correct": False, "gold_parsed": False, "prediction_parsed": False,
                        "status": "worker_error", "error": "worker_exited"}
        return {"correct": False, "gold_parsed": False, "prediction_parsed": False,
                "status": "verification_timeout", "error": "worker_timeout"}
    finally:
        if process.pid is not None:
            process.join(timeout=0.1)
            if process.is_alive():
                process.terminate()
                process.join()
        receiver.close()
        sender.close()


def verify_final_answer(completion, gold):
    require_math_verify()  # Missing dependency is a configuration error, not a wrong answer.
    prediction, source, error = extract_final_answer(completion)
    base = {"schema": SCHEMA, "answer_source": source, "prediction": prediction,
            "correct": False, "gold_parsed": False, "prediction_parsed": False,
            "status": "extraction_failed", "error": error}
    if error:
        return base
    if gold is None:
        return {**base, "status": "gold_parse_failed", "error": "missing_gold"}
    base.update(_bounded_grade(prediction, str(gold)))
    return base


def verification_summary(results):
    results = list(results)
    n = len(results)
    counts = dict(Counter(r["status"] for r in results))
    return {"schema": SCHEMA, "version": require_math_verify(), "count": n,
            "correct_count": sum(r["correct"] for r in results),
            "accuracy": sum(r["correct"] for r in results) / n if n else None,
            "status_counts": counts,
            "error_counts": dict(Counter(r["error"] for r in results if r.get("error"))),
            "answer_source_counts": dict(Counter(r["answer_source"] for r in results)),
            "failure_count": sum(v for k, v in counts.items() if k not in ("correct", "incorrect")),
            "scope": "Final-answer equivalence only; does not verify reasoning or proofs.",
            "float_rounding": 10, "numeric_precision": 15,
            "fallback_mode": "no_fallback"}


def math_verify_reward(completions, answer, **kwargs):
    if len(completions) != len(answer):
        raise ValueError("Completion/reference batch sizes differ")
    return [float(verify_final_answer(c, a)["correct"]) for c, a in zip(completions, answer)]


def log_verification_metrics(metrics, results):
    if not results:
        return
    for key, value in verification_summary(results).items():
        if key == "accuracy":
            metrics["math_verify/accuracy"].append(value)
        elif key == "failure_count":
            metrics["math_verify/failure_rate"].append(value / len(results))
    counts = Counter(r["status"] for r in results)
    for status in ["correct", "incorrect", "extraction_failed", "gold_parse_failed",
                   "prediction_parse_failed", "gold_parse_error", "prediction_parse_error",
                   "verify_error", "gold_parse_timeout", "prediction_parse_timeout",
                   "verify_timeout", "verification_timeout", "worker_error"]:
        metrics["math_verify/status/" + status].append(counts[status] / len(results))
