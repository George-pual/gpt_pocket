"""Final-answer grading only; never a proof or reasoning-step verifier.

Mathematical grading has no string fallback or arbitrary reasoning search.
Explicit ordered-name questions use whole categorical label sequences instead
of treating English names as products of mathematical symbols.
Linux uses Math-Verify's timeouts; Windows uses a bounded spawn worker because
the library's nested multiprocessing timeout target cannot be pickled there.
"""
from __future__ import annotations

import importlib.metadata
import html
import multiprocessing
import os
import re
from collections import Counter
from functools import lru_cache

SCHEMA = "math-verify-final-answer-v4"
MAX_ANSWER_CHARS = 2048
MAX_FINAL_REGION_CHARS = 32768
SCALAR = r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:\s*/\s*[+-]?\d+)?'
UNITS = r'(?:kilowatts?|watts?|feet|foot|meters?|metres?|centimeters?|centimetres?|cm|inches|inch|hours?|minutes?|seconds?|dollars?|containers?|pounds?|gallons?|degrees?|radians?|years?|students?|pebbles?|canoes?|eggs?|pies|cans?|bows?|sides?|units?|sqft)'


def _word_number(value):
    words = dict(zip('zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen'.split(), range(20)))
    return str(words[value.lower()]) if value.lower() in words else value


def _strip_display(value):
    value = value.strip()
    for left, right in [('$$', '$$'), (r'\(', r'\)'), (r'\[', r'\]'), ('$', '$'), ('**', '**'), ('"', '"'), ("'", "'")]:
        if value.startswith(left) and value.endswith(right) and len(value) > len(left) + len(right):
            return _strip_display(value[len(left):-len(right)])
    return value


def _boxed_values(value):
    values = []
    for match in re.finditer(r'\\boxed\s*\{', value):
        depth = 1
        for i in range(match.end(), len(value)):
            depth += (value[i] == '{') - (value[i] == '}')
            if depth == 0:
                values.append(value[match.end():i].strip())
                break
        else:
            return []
        if len(values) > 8:
            return []
    return values


def _declaration(value, question=None):
    """Recognize a whole answer declaration, never a matching number in reasoning."""
    value = value.strip().removesuffix('.').strip()
    value = re.sub(r'^(?:(?:Therefore|Thus|So|Hence|Finally|Confidently)[,:]?\s*)+', '', value, flags=re.I)
    if question and re.search(r'how many\b.*\bsolutions', question, re.I | re.S):
        if re.fullmatch(r'No (?:positive |real |positive real )?solutions', value, re.I):
            return '0'
    match = re.fullmatch(r'(?:Final answer|Answer|最终答案|答案)\s*[:：]\s*(.+)', value, re.I | re.S)
    if match:
        return match[1]
    # Explicit option/graph labels precede any explanatory numerical choice value.
    match = re.fullmatch(r'(?:The correct answer is|The correct graph is the one labeled as)\s+["\']?([A-E])["\']?(?:\)\s+[^\n]+)?', value, re.I)
    if match:
        return match[1].upper()
    match = re.fullmatch(r'(?:Rectangular form|Simplified form)\s*:\s*(.+)', value, re.I | re.S)
    if match:
        return match[1]
    match = re.fullmatch(r'The graph of .+ is (?:a |an )?(?:non-degenerate )?(parabola|circle|ellipse|hyperbola|point|line|two lines|empty)', value, re.I)
    if match:
        return match[1].lower()
    # These prefixes describe the requested answer, not arbitrary intermediate facts.
    match = re.fullmatch(r'(?:The (?:simplified form|value|sum|fraction|number|units digit|area|volume|angle|distance|length|height|width|perimeter|ratio|largest|greatest|smallest|maximum|minimum|ordered pair|ordered quadruple|coordinates|measure|probability|solution|equation|total|correct answer)[\s\S]*?|We have found that [\s\S]+?)\s+(?:is|are|equals)\s+(.+)', value, re.I | re.S)
    if match:
        return match[1]
    match = re.fullmatch(r'The coordinates of .+ are (.+)', value, re.I | re.S)
    if match:
        return match[1]
    # A count is explicitly placed immediately after "there are"; trailing details
    # are not searched for another number.
    atom = r'(?:\d{1,3}(?:,\d{3})+|' + SCALAR + r'|zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen)'
    match = re.fullmatch(r'There (?:are|is) (' + atom + r')\s+(?:(?:different|distinct|positive|prime|whole|real|integer|possible|two-digit|three-digit|four-digit)\s+)*(?:ways|numbers|integers|values|divisors|pairs|statements|cans|sides|pizzas)\b[^.!?]*(?:!)?', value, re.I)
    if match:
        return _word_number(match[1])
    match = re.fullmatch(r'(' + atom + r')\s+(?:statements|different positive two-digit integers)\b[^.!?]*', value, re.I)
    if match:
        return _word_number(match[1])
    match = re.fullmatch(r'([A-Z][a-z]+) (?:is|weighs|has|will need|will get|can make) (.+)', value, re.S)
    if match:
        return match[2].removesuffix(' old')
    match = re.fullmatch(r'(?:It would take [A-Z][a-z]+|She laid) (' + atom + r')\s+(?:more minutes|eggs)\b[^.!?]*', value, re.I)
    if match:
        return _word_number(match[1])
    match = re.fullmatch(r'(' + atom + r')\s+(?:is|are) the (?:smallest|largest|greatest|least)\b[^.!?]*', value, re.I)
    if match:
        return _word_number(match[1])
    match = re.fullmatch(r'The polygon has (.+) sides', value, re.I)
    if match:
        return _strip_display(match[1])
    match = re.fullmatch(r'[\d\s()+*/.-]+\s*=\s*(' + atom + r')\s+(?:(?:different|distinct|two-digit|three-digit|four-digit)\s+)*(?:numbers|ways|pizzas)\b[^.!?]*', value, re.I)
    if match:
        return match[1]
    return None


def normalize_final_candidate(value, question=None):
    """Gold-independent, conservative fallback within one explicit final region.

    Returns (candidate, rule, error). Unclear answers stay unconfirmed. This is
    deliberately narrower than a human reviewer and never chooses a candidate
    because it matches the reference.
    """
    if not isinstance(value, str) or len(value) > MAX_FINAL_REGION_CHARS or '__' in value:
        return None, None, 'unsafe_or_oversized_final_region'
    text = html.unescape(value).strip()
    if re.search(r'\b(?:but|instead|maybe|perhaps|possibly|might be|could be|not sure|alternatively|incorrect|wrong|invalid|rejected|not correct|not the answer|cannot determine|can.not determine|no unique|does not have a solution|no solution for|please check|not valid|no standard format|will use Python)\b', text, re.I):
        return None, None, 'ambiguous_or_nonanswer'
    # Contradictory explicit answer declarations must not be resolved by gold.
    claims = re.findall(r'(?:^|[.!?]\s*|\n)\s*(?:The answer is|Final answer:|Answer:)\s*([^\n.!?]+)', text, re.I)
    if len({_strip_display(c.strip()) for c in claims}) > 1:
        return None, None, 'conflicting_final_claims'
    boxes = _boxed_values(text)
    if boxes:
        if len({_strip_display(v) for v in boxes}) != 1:
            return None, None, 'conflicting_boxed_answers'
        text, rule = boxes[0], 'consistent_explicit_box'
    else:
        # Alternative results are not selectively reduced to a favorable one.
        if re.search(r'\bor\b|\bactually\b', text, re.I):
            return None, None, 'multiple_or_revised_answers'
        candidate = _declaration(text, question)
        rule = 'whole_final_declaration' if candidate is not None else 'whole_final_notation'
        if candidate is None:
            # Only an explicit conclusion sentence or paragraph is eligible.
            conclusions = list(re.finditer(r'(?:^|[.!?]\s*|\n)\s*(?:Therefore|Thus|So|Hence|Finally)[,:]?\s*', text, re.I))
            if conclusions:
                candidate = _declaration(text[conclusions[-1].start():].lstrip('.!?\n '), question)
                rule = 'explicit_terminal_conclusion'
            if candidate is None:
                match = re.search(r'(?:^|[.\n])\s*(?:Final answer|Answer)\s*:\s*([^\n]+)\s*$', text, re.I)
                if match:
                    candidate, rule = match[1], 'explicit_terminal_answer_line'
            if candidate is None:
                declarations = list(re.finditer(r'(?:^|(?<=[.!?])\s+|\n+)(?=(?:The\b|There\b|[A-Z][a-z]+\s+(?:is|weighs|has|will need|will get|can make)\b))', text))
                if declarations:
                    candidate = _declaration(text[declarations[-1].end():], question)
                    rule = 'explicit_terminal_declaration'
        text = candidate if candidate is not None else text
    orphan_dollar = re.fullmatch(r'(' + SCALAR + r')\s*\n\s*\$', text)
    if orphan_dollar:
        text = orphan_dollar[1]
    text = text.strip().removesuffix('.').strip()
    # Preserve the scope of a leading minus outside an inline math span.
    if re.match(r'^[+-]\s*\$', text):
        return None, None, 'ambiguous_sign_scope'
    text = _strip_display(text)
    text = re.sub(r'(\w+)<sub>(\d+)</sub>', r'\1_\2', text, flags=re.I)
    text = re.sub(r'<(?:negative|minus)(\d+)>', r'-\1', text, flags=re.I)
    # Only trailing output-control tags after an otherwise complete answer.
    text = re.sub(r'(?:</?(?:start_working_out|end_working_out|WORKING_OUT|workedout|SMALLGPT2)>\s*)+$', '', text, flags=re.I).strip()
    text = re.sub(r'^Approximately\s+', '', text, flags=re.I)
    text = re.sub(r',\s+(?:which|since|because)\b.*$', '', text, flags=re.I | re.S)
    # A fixed, entire scalar-unit form; never strip percent or an inequality.
    text = re.sub(r'^(\$\\\$\$|\\\$)\s*(' + SCALAR + r')$', r'\2', text)
    text = _strip_display(text)
    text = _word_number(text)
    text = text.replace('−', '-').replace('²', '^2').replace('³', '^3').replace('π', r'\pi').replace('×', r'\times ')
    text = re.sub(r'√\s*(\d+(?:\.\d+)?)', r'\\sqrt{\1}', text)
    text = re.sub(r'(?<=\d)°', r'^\\circ', text)
    if re.fullmatch(r'[+-]?\d{1,3}(?:,\d{3})+(?:\.\d+)?', text):
        text = text.replace(',', '')
    if re.fullmatch(r'=\s*' + SCALAR, text):
        text = text.lstrip('= ').strip()
    if text.endswith('\\') and not text.endswith('\\\\'):
        text = text[:-1].strip()
    # Unit-bearing scalar declarations can retain a harmless postmodifier.
    count = re.fullmatch(r'(' + SCALAR + r')\s+(?:of a |of an )?(' + UNITS + r')(?:\s+(?:left|out of\b.*|are green))?', text, re.I)
    if count:
        text = count[1] + ' ' + count[2]
    text = text.removesuffix('.').strip()
    if question and re.search(r'\bratio\b', question, re.I):
        ratio = re.fullmatch(r'(' + SCALAR + r')\s*:\s*(' + SCALAR + r')', text)
        if ratio:
            text = r'\frac{' + ratio[1] + '}{' + ratio[2] + '}'
    # No prose, HTML, comparison prefix, or dangling delimiter may reach the
    # symbolic parser, whose permissive extraction could otherwise salvage it.
    if re.search(r'[<>]|\b(?:is|are|answer|there|the|and|or|not|will|could|would|cannot|if|unless|which|because|since)\b', text, re.I):
        return None, None, 'not_an_isolated_answer'
    if not _math_surface(text) and not _split_candidate_unit(text):
        return None, None, 'leftover_prose'
    if text.count('$') or len(text) > MAX_ANSWER_CHARS:
        return None, None, 'unbalanced_or_oversized_answer'
    return text, rule, None


def _scalar_unit(value):
    return re.fullmatch(r'(' + SCALAR + r')\s+((?:square\s+|cubic\s+)?' + UNITS + r')\.?', value, re.I)


def _canonical_unit(unit):
    unit = ' '.join(unit.lower().split())
    aliases = {'foot':'feet','meter':'meters','metre':'meters','metres':'meters',
               'centimeter':'centimeters','centimetre':'centimeters','centimetres':'centimeters','cm':'centimeters',
               'inch':'inches','hour':'hours','minute':'minutes','second':'seconds',
               'pound':'pounds','gallon':'gallons','dollar':'dollars','year':'years',
               'degree':'degrees','radian':'radians'}
    prefix = ''
    for p in ['square ', 'cubic ']:
        if unit.startswith(p):
            prefix, unit = p, unit[len(p):]
            break
    return prefix + aliases.get(unit,unit)


def _question_unit(question):
    if not question:
        return None
    unit = r'((?:square\s+|cubic\s+)?' + UNITS + r')'
    # The output unit in "How many cubic feet ... three cubic yards" is feet.
    match = re.search(r'\bhow many\s+' + unit + r'\b', question, re.I)
    if match:
        return _canonical_unit(match[1])
    matches = re.findall(r'\bin\s+' + unit + r'\s*(?:[?.!,]|$)', question, re.I)
    canonical = {_canonical_unit(m) for m in matches}
    return canonical.pop() if len(canonical)==1 else None


def _math_surface(value):
    """Reject leftover English before a permissive symbolic extraction fallback."""
    plain = re.sub(r'\\[A-Za-z]+', '', value)
    words = re.findall(r'[A-Za-z]{3,}', plain)
    allowed = {'sin','cos','tan','cot','sec','csc','sinh','cosh','tanh','log','exp','sqrt',
               'arcsin','arccos','arctan','pmatrix','bmatrix','matrix',
               'hyperbola','parabola','ellipse','circle','point','line','empty'}
    return all(w.lower() in allowed for w in words)


def _split_candidate_unit(value):
    match = re.fullmatch(r'(.+?)\s+((?:square\s+|cubic\s+)?' + UNITS + r')(?:\^([23]))?', value.strip(), re.I | re.S)
    if not match or not _math_surface(match[1]):
        return None
    unit = match[2]
    if match[3]:
        if unit.lower().startswith(('square ','cubic ')):
            return None
        unit = ('square ' if match[3]=='2' else 'cubic ') + unit
    return match[1], _canonical_unit(unit)


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
        values = _boxed_values(text)
        if len(values) != len(boxes):
            return None, 'boxed', 'unclosed_boxed'
        if len(set(values)) != 1:
            return None, 'boxed', 'conflicting_boxed_answers'
        start = boxes[-1].end()
        depth = 1
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    if not re.fullmatch(r'[\s.$\\()\[\]]*', text[i+1:]):
                        return None, 'boxed', 'content_after_boxed_answer'
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


def _ordered_name_question(question):
    return bool(question and re.search(
        r'\b(?:list|give|write)\b[^.!?]{0,100}\b(?:names|choices|events)\b'
        r'[^.!?]{0,160}\b(?:separated by commas|most popular|least popular)\b', question, re.I))


def _ordered_name_labels(value):
    """Parse one entire comma-separated name list; never salvage prose items."""
    if not isinstance(value, str) or len(value) > MAX_ANSWER_CHARS:
        return None
    value = _strip_display(value.strip())
    for _ in range(3):
        wrapped = re.fullmatch(r'\\(?:text|mathrm|boxed)\{([^{}]*)\}', value, re.S)
        if not wrapped:
            break
        value = _strip_display(wrapped[1])
    parts = [part.strip() for part in value.split(',')]
    if not 2 <= len(parts) <= 8:
        return None
    for part in parts:
        if not re.fullmatch(r"[A-Za-z][A-Za-z'-]*(?:\s+[A-Za-z][A-Za-z'-]*){0,4}", part):
            return None
        if re.search(r'\b(?:or|and|but|not|maybe|answer|is|are|then|because|actually)\b', part, re.I):
            return None
    return tuple(' '.join(part.casefold().split()) for part in parts)


def _grade_pair(prediction, gold, timeout, question=None):
    from math_verify import verify
    from math_verify.errors import TimeoutException
    result = {"correct": False, "gold_parsed": False, "prediction_parsed": False,
                "status": "gold_parse_failed", "error": None,
                "normalized_prediction": None, "normalization_rule": None}
    phase = "gold_parse"
    try:
        if _ordered_name_question(question):
            reference_names = _ordered_name_labels(gold)
            if reference_names is None:
                result.update(error='unsupported_ordered_name_gold')
                return result
            result.update(gold_parsed=True)
            candidate_names = _ordered_name_labels(prediction)
            if candidate_names is None:
                result.update(status='prediction_parse_failed', error='not_a_whole_ordered_name_list')
                return result
            result.update(correct=candidate_names == reference_names, prediction_parsed=True,
                          normalized_prediction=', '.join(candidate_names),
                          normalization_rule='ordered_name_sequence_v4',
                          status='correct' if candidate_names == reference_names else 'incorrect')
            return result
        # Handle explicitly stated units before the symbolic parser can treat
        # their letters as variables or discard them.
        gold_unit = _split_candidate_unit(_normalize_answer(gold))
        if gold_unit:
            pred_unit = _split_candidate_unit(_normalize_answer(prediction))
            if not pred_unit:
                cleaned, rule, _ = normalize_final_candidate(prediction, question)
                if cleaned:
                    pred_unit = _split_candidate_unit(_normalize_answer(cleaned))
                    if pred_unit:
                        result.update(normalized_prediction=cleaned, normalization_rule=rule)
            if not pred_unit or pred_unit[1] != gold_unit[1]:
                result.update(status='incorrect')
                return result
            prediction, gold = pred_unit[0], gold_unit[0]
        reference = _parse_answer(gold, timeout)
        result["gold_parsed"] = bool(reference)
        if not reference:
            return result
        phase = "prediction_parse"
        cleaned, rule, normalize_error = normalize_final_candidate(prediction, question)
        if normalize_error in {'unsafe_or_oversized_final_region', 'ambiguous_or_nonanswer',
                               'conflicting_final_claims', 'conflicting_boxed_answers',
                               'multiple_or_revised_answers', 'ambiguous_sign_scope'}:
            result.update(status='prediction_parse_failed', error=normalize_error)
            return result
        expected_unit = _question_unit(question)
        declared_unit = _split_candidate_unit(_normalize_answer(cleaned or prediction))
        if expected_unit and not declared_unit:
            angle = re.fullmatch(r'(' + SCALAR + r')(?:°|\^\\circ)', (cleaned or prediction).strip())
            if angle:
                declared_unit = angle[1], 'degrees'
        if expected_unit and declared_unit:
            actual_unit = declared_unit[1]
            if actual_unit not in {'units','square units','cubic units'} and actual_unit != expected_unit:
                result.update(status='incorrect', prediction_parsed=True, error='question_unit_mismatch')
                return result
        # A bare numeric reference may omit the unit requested by its problem.
        # Do not erase units on both sides (2 feet must not equal 2 meters).
        normalized_gold = _normalize_answer(gold)
        if re.fullmatch(r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:\s*/\s*[+-]?\d+)?', normalized_gold):
            normalized = _normalize_answer(prediction, allow_units=True)
            if normalized != prediction:
                result.update(normalized_prediction=normalized, normalization_rule='whole_scalar_unit_or_sentence')
            prediction = normalized
        candidate = _parse_answer(prediction, timeout)
        if not candidate and cleaned is not None:
            declared = _split_candidate_unit(cleaned)
            if declared and not gold_unit:
                cleaned = declared[0]
            cleaned = _normalize_answer(cleaned, allow_units=not bool(gold_unit))
            candidate = _parse_answer(cleaned, timeout)
            if candidate:
                result.update(normalized_prediction=cleaned, normalization_rule=rule)
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


def _worker(connection, prediction, gold, question=None):
    try:
        import logging
        logging.getLogger("math_verify.parser").setLevel(logging.ERROR)
        logging.getLogger("math_verify.grader").setLevel(logging.ERROR)
        connection.send(_grade_pair(prediction, gold, None, question))
    finally:
        connection.close()


@lru_cache(maxsize=4096)
def _bounded_grade(prediction, gold, question=None):
    if os.name != "nt":
        return _grade_pair(prediction, gold, 5, question)
    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(target=_worker, args=(sender, prediction, gold, question))
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


def verify_final_answer(completion, gold, question=None):
    require_math_verify()  # Missing dependency is a configuration error, not a wrong answer.
    prediction, source, error = extract_final_answer(completion)
    base = {"schema": SCHEMA, "answer_source": source, "prediction": prediction,
            "correct": False, "gold_parsed": False, "prediction_parsed": False,
            "status": "extraction_failed", "error": error}
    if error:
        return base
    if gold is None:
        return {**base, "status": "gold_parse_failed", "error": "missing_gold"}
    base.update(_bounded_grade(prediction, str(gold), question))
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
