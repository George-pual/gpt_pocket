"""Versioned OFFLINE answer scorer. No model loading or training reward changes.

Extraction never receives the reference answer. Math and format are separate.
Question context controls percent units, named units and the imaginary unit.
Symbolic parsing is applied only to isolated answers, with no text fallback.
Run symbolic operations in the bounded worker provided by rescore_fixed_answers.
"""
from __future__ import annotations

import re
from fractions import Fraction

SCHEMA = 'audited-final-answer-v3.2'
NUMBER = r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:\s*/\s*[+-]?\d+)?'
UNIT_NAMES = {
    'foot': 'feet', 'feet': 'feet', 'ft': 'feet', 'inch': 'inches', 'inches': 'inches',
    'meter': 'meters', 'meters': 'meters', 'metre': 'meters', 'metres': 'meters',
    'centimeter': 'centimeters', 'centimeters': 'centimeters',
    'centimetre': 'centimeters', 'centimetres': 'centimeters',
    'mile': 'miles', 'miles': 'miles', 'yard': 'yards', 'yards': 'yards',
    'dollar': 'dollars', 'dollars': 'dollars', 'cent': 'cents', 'cents': 'cents',
    'hour': 'hours', 'hours': 'hours', 'minute': 'minutes', 'minutes': 'minutes',
    'second': 'seconds', 'seconds': 'seconds', 'kilowatt': 'kilowatts',
    'kilowatts': 'kilowatts', 'watt': 'watts', 'watts': 'watts',
    'degree': 'degrees', 'degrees': 'degrees', 'edge': 'edges', 'edges': 'edges',
    'container': 'containers', 'containers': 'containers', 'unit': 'units', 'units': 'units',
    'kg': 'kilograms', 'kgs': 'kilograms', 'kilogram': 'kilograms', 'kilograms': 'kilograms',
    'ml': 'milliliters', 'milliliter': 'milliliters', 'milliliters': 'milliliters',
    'liter': 'liters', 'liters': 'liters', 'litre': 'liters', 'litres': 'liters',
    'lb': 'pounds', 'lbs': 'pounds', 'pound': 'pounds', 'pounds': 'pounds',
    'euro': 'euros', 'euros': 'euros', 'quarts': 'quarts', 'quart': 'quarts',
    'mb': 'megabytes', 'megabyte': 'megabytes', 'megabytes': 'megabytes',
    'cm': 'centimeters', 'km': 'kilometers', 'kilometer': 'kilometers',
    'kilometers': 'kilometers', 'gm': 'grams', 'gram': 'grams', 'grams': 'grams',
    'mph': 'miles_per_hour', 'student': 'students', 'students': 'students',
    'teacher': 'teachers', 'teachers': 'teachers', 'way': 'ways', 'ways': 'ways',
    'meal': 'meals', 'meals': 'meals', 'digit': 'digits', 'digits': 'digits',
    'multiple': 'multiples', 'multiples': 'multiples', 'ticket': 'tickets', 'tickets': 'tickets',
}


def _boxes(text):
    result = []
    for m in re.finditer(r'\\boxed\s*\{', text):
        depth = 1
        for i in range(m.end(), len(text)):
            if text[i] == '{' and text[i-1] != '\\':
                depth += 1
            elif text[i] == '}' and text[i-1] != '\\':
                depth -= 1
                if depth == 0:
                    result.append((m.start(), i+1, text[m.end():i]))
                    break
    return result


def _terminal_box(text):
    boxes = _boxes(text)
    if boxes:
        start, end, value = boxes[-1]
        tail = text[end:]
        # Only typesetting, punctuation and closing known tags may follow it.
        tail = re.sub(r'</?SOLUTION>|</think>|<end_working_out>', '', tail, flags=re.I)
        tail = re.sub(r'\\(?:[\],.!;)]|end\{(?:equation\*?|align\*?)\})', '', tail)
        if not tail.strip(' \t\r\n.$*'):
            return value.strip()
    return None


def extract_answer(completion, eos=False):
    text = re.sub(r'(?:<\|endoftext\|>|<\|im_end\|>|</s>)\s*$', '', completion.strip()).strip()
    opens = len(re.findall(r'<SOLUTION>', text, re.I))
    closes = len(re.findall(r'</SOLUTION>', text, re.I))
    blocks = list(re.finditer(r'<SOLUTION>(.*?)</SOLUTION>', text, re.I | re.S))
    tag_issue = None
    if opens or closes or re.search(r'</?SOLUTION\b', text, re.I):
        if opens == closes == len(blocks) == 1 and not text[blocks[0].end():].strip():
            return dict(prediction=blocks[0][1].strip(), answer_source='solution', extraction_error=None,
                        tag_issue=None)
        tag_issue = 'malformed_or_multiple_solution'
        last_open = list(re.finditer(r'<SOLUTION>', text, re.I))
        if last_open:
            m=last_open[-1]
            tail=text[m.end():].strip()
            # Nested unclosed tags can still contain one explicit final block.
            # Never choose among two completed, conflicting answer blocks.
            if '</SOLUTION>' not in text[:m.start()].upper():
                closed = re.fullmatch(r'(.*?)</SOLUTION>\s*',tail,re.I|re.S)
                partial = eos and not re.search(r'</?SOLUTION\b',tail,re.I)
                value=closed[1].strip() if closed else tail if partial else ''
                value=re.sub(r'<end_working_out>\s*$','',value).strip()
                if value:
                    return dict(prediction=value,answer_source='solution_recovered_at_eos' if not closed else 'nested_final_solution',
                                extraction_error=None,tag_issue=tag_issue)
    # A terminal explicit boxed answer can establish math correctness even
    # when the required tag structure is invalid. Format is scored separately.
    boxed = _terminal_box(text)
    if boxed is not None:
        return dict(prediction=boxed, answer_source='terminal_boxed', extraction_error=None, tag_issue=tag_issue)
    line = re.search(r'(?:^|\n)\s*(?:Final\s+answer|Answer|最终答案|答案)\s*[:：]\s*([^\n]+)\s*$', text, re.I)
    if line and not tag_issue:
        return dict(prediction=line[1].strip(), answer_source='answer_line', extraction_error=None, tag_issue=None)
    if eos:
        # A terminal equality at a real EOS is an anchored final expression,
        # not the last number anywhere in the text. Never salvage truncations.
        clean=re.sub(r'</?SOLUTION>|</?start_working_out>|<end_working_out>', '',text,flags=re.I).strip()
        if '=' in clean:
            rhs=strip_math(clean.rsplit('=',1)[1].strip().rstrip('.'))
            if rhs and (_safe_expression(rhs) or _unit(rhs) or _single_quantity_statement(rhs)):
                return dict(prediction=rhs,answer_source='terminal_equality_at_eos',extraction_error=None,
                            tag_issue=tag_issue or 'missing_solution')
    return dict(prediction=None, answer_source='none', extraction_error=tag_issue or 'missing_final_answer', tag_issue=tag_issue)


def strip_math(value):
    value = value.strip()
    for _ in range(5):
        before = value
        if value.startswith('$$') and value.endswith('$$'):
            value = value[2:-2].strip()
        elif value.startswith('$') and value.endswith('$'):
            value = value[1:-1].strip()
        elif value.startswith((r'\(', r'\[')) and value.endswith((r'\)', r'\]')):
            value = value[2:-2].strip()
        box = _boxes(value)
        if len(box) == 1 and box[0][0] == 0 and box[0][1] == len(value):
            value = box[0][2].strip()
        numeric_text = re.fullmatch(r'\\(?:text|mathrm|mbox)\{\s*('+NUMBER+r')\s*\}', value)
        if numeric_text:
            value = numeric_text[1]
        if before == value:
            break
    return value


def isolate_expression(raw):
    """Extract a terminal declarative assertion, independently of gold.

    Do not take the last number of a proof. Only a closed answer clause with
    a subject and an explicit answer verb is eligible; its entire RHS is graded.
    """
    value = strip_math(raw)
    box = _terminal_box(value)
    if box is not None:
        return strip_math(box), 'boxed_in_answer'
    # The start can be a sentence boundary. Formula/decimal dots cannot match
    # unless followed by a recognized declarative subject.
    sentences = re.split(r'(?<=[.!?])\s+(?=(?:Therefore|Thus|Hence|So|Finally|The|the|Positive)\b)', value)
    terminal = sentences[-1]
    pattern = (r'^\s*(?:(?:Therefore|Thus|Hence|So|Finally),?\s+)?'
               r'(?:[Tt]he\s+[^\n!?]*\s+(?:is|equals)|'
               r'[Pp]ositive value of\s+.+?\s+is|'
               r'[A-Z][A-Za-z]+\s+will need|[Tt]he\s+\w+\s+travels)\s+(.+?)\s*$')
    matches = list(re.finditer(pattern, terminal, re.S))
    if matches:
        rhs = matches[-1][1].strip()
        # Multi-sentence RHS is not a final single assertion.
        if not re.search(r'[.!?]\s+[A-Za-z]', rhs):
            if rhs.endswith('.'):
                rhs = rhs[:-1].strip()
            return strip_math(rhs.removeprefix('worth ')), 'terminal_declarative_clause'
    return value, 'whole_answer'


def percent_context(question):
    # Input rates somewhere in a problem do not determine the requested unit.
    return bool(re.search(r'(?:what|which)\s+(?:percent(?:age)?)(?:\s|\b)|'
                          r'express\s+(?:your |the )?answer\s+(?:as|in)\s+(?:a\s+)?percent|'
                          r'(?:find|calculate|determine)\s+the\s+percentage', question, re.I))


def _unit(value):
    # Flatten only whole text unit suffixes, never remove arbitrary words.
    value = re.sub(r'\\(?:text|mathrm)\{([^{}]*)\}', r' \1 ', value)
    value = re.sub(r'\\[,;! ]', ' ', value).strip()
    value = re.sub(r'\bunits\s*\^\s*\{?2\}?', 'square units', value)
    value = re.sub(r'\bchildren(?:\x27s)?\s+tickets\b', 'tickets', value)
    value = re.sub('(' + NUMBER + r')(kgs?|m[Ll])\b', r'\1 \2', value)
    m = re.fullmatch(r'(.+?)\s+(?:(square|cubic)\s+)?(?:new\s+)?([A-Za-z]+)\.?', value)
    if m and m[3].lower() in UNIT_NAMES:
        expr = strip_math(m[1])
        if _safe_expression(expr):
            return expr, ((m[2]+' ') if m[2] else '') + UNIT_NAMES[m[3].lower()]
    if re.fullmatch(r'[$€]' + NUMBER, value):
        return value[1:], 'dollars' if value[0]=='$' else 'euros'
    inches=re.fullmatch('('+NUMBER+r')"',value)
    if inches:
        return inches[1], 'inches'
    return None


def _context_unit(unit, question):
    base = unit.removeprefix('square ').removeprefix('cubic ')
    speed_request = re.search(r'(?:^|[.?!]\s+)(?:How fast\b|What (?:is|was|will be) (?:the )?(?:average )?speed\b)[^?]*\?\s*$', question, re.I)
    if speed_request and base != 'miles_per_hour':
        return False
    aliases = [a for a, b in UNIT_NAMES.items() if b == base]
    if base == 'miles_per_hour' and re.search(r'\bmiles?\s+per\s+hour\b', question, re.I):
        return True
    if base == 'miles_per_hour' and re.search(r'\bmiles?\b', question, re.I) and re.search(r'\bhours?\b', question, re.I) and re.search(r'\bhow fast\b|\bspeed\b', question, re.I):
        return True
    if base == 'dollars' and ('$' in question or re.search(r'\b(?:cents?|quarters?|twenties)\b', question,re.I)):
        return True
    if base == 'euros' and '€' in question:
        return True
    if base == 'miles' and re.search(r'mph\b', question, re.I):
        return True
    if base == 'kilograms' and re.search(r'(?:kgs?|kilos?)\b',question,re.I):
        return True
    if base == 'inches' and '"' in question:
        return True
    if base == 'hours' and re.search(r'\bmph\b', question,re.I) and re.search(r'how long',question,re.I):
        return True
    return any(re.search(r'\b' + re.escape(a) + r'\b', question, re.I) for a in aliases)


def _single_quantity_statement(value):
    """A short final quantity or declarative sentence with exactly one scalar.

    Applied inside an already explicit final-answer block, never to reasoning.
    Refuse alternatives, negation, conditions, calculations and multiple numbers.
    Count nouns (cups, downloads, etc.) aren't a finite unit whitelist.
    """
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])|\n+',value.strip())
    value=sentences[-1]
    if len(value)>240 or re.search(r'[\n\\=^<>+*/{}]|\b(?:not|never|no|wrong|invalid|incorrect|maybe|perhaps|either|or|but|if|unless|except|actually|however|rather|instead|alternatively)\b|\bI (?:tried|considered|guessed)\b', value, re.I):
        return None
    token = r'(?<![\w.])[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:\s*%)?(?![\w.])'
    found = list(re.finditer(token, value.rstrip('.')))
    if len(found)!=1:
        return None
    m=found[0]
    prose=value[:m.start()]+value[m.end():]
    if not re.fullmatch(r"[\w\s.,'’$%&\-]*", prose):
        return None
    if re.fullmatch(r'\s*[A-Za-z]{1,2}\s*',prose):
        return None
    starts_quantity = not value[:m.start()].strip(' $')
    declarative = re.search(r'\b(?:is|are|was|were|has|have|had|needs?|will|would|can|ordered|sold|walked|runs?|takes?|took|earns?|gets?|bought|paid|spent|spends|costs?|makes?|made|travels?|receives?|received|uses?|used|eats?|ate|buys?|saves?|owns?|weighs?|contains?|remains?|scored|picked|gathered|lasted|delivers|learns|read|found|gave|finished)\b', value[:m.start()], re.I)
    if starts_quantity or declarative:
        number=m[0].strip()
        if value[:m.start()].rstrip().endswith('$'):
            number='$'+number
        return number
    return None


def _declarative_quantity(value, question):
    """Quantity stated by the main predicate, independent of gold."""
    value=re.split(r'(?<=[.!?])\s+(?=[A-Z])|\n+',value.strip())[-1]
    if re.search(r'\bif\b.*\bnot\b',question,re.I):
        # A hypothetical purchasing question can have a hypothetical answer;
        # removing its condition must not turn a negated quantity positive.
        value=re.sub(r'\s+if\s+\w+\s+had\s+not\s+.*$','',value,flags=re.I)
    if len(value)>300 or re.search(r'[\\=^<>+*/{}]|\b(?:not|never|no|wrong|invalid|incorrect|maybe|perhaps|either|but|unless|except|actually|however|rather|instead|alternatively)\b',value,re.I):
        return None
    if re.search(r'\bor\s+(?:[+-]?\d|one|two|three|four|five|six|seven|eight|nine|ten)\b',value,re.I):
        return None
    leading=re.match('('+NUMBER+r')\s+([A-Za-z]+)\b',value)
    if leading and not re.search(r'\b(?:and|or)\s+\d',value,re.I):
        noun=leading[2]
        if re.search(r'how many\s+(?:\w+\s+){0,2}'+re.escape(noun)+r'\b',question,re.I):
            return leading[1]
    pattern = (r'\b(?:is|are|was|were|has|have|had|needs?|will|would|can|could|'
               r'ordered|sold|walked|runs?|takes?|took|earns?|gets?|bought|paid|spent|spends|'
               r'costs?|makes?|made|travels?|receives?|received|uses?|used|eats?|ate|buys?|'
               r'saves?|owns?|weighs?|contains?|remains?|scored|picked|gathered|lasted|'
               r'delivers|learns|read|found|gave|finished|watched|brought)\b'
               r'(?:\s+[A-Za-z]+){0,5}\s+([$€]?[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:\s*%)?)'
               r'(?!\d|\.\d)')
    m=re.search(pattern,value,re.I)
    if not m:
        return None
    tail=value[m.end():].strip()
    if re.match(r'(?:(?:or|and|but)\s+(?:[+-]?\d|another)|[,;]\s*[+-]?\d)',tail,re.I):
        return None
    unit_match=re.match(r'(?:(square|cubic)\s+)?([A-Za-z]+)\b',tail)
    quantity=m[1]
    if unit_match and unit_match[2].lower() in UNIT_NAMES:
        quantity+=' '+unit_match[0]
    return quantity


def normalize_pair(prediction, gold, question):
    prediction = re.sub(r'\bmiles?\s+per\s+hour\b', 'mph', prediction, flags=re.I)
    gold = re.sub(r'\bmiles?\s+per\s+hour\b', 'mph', str(gold), flags=re.I)
    prediction, isolation = isolate_expression(prediction)
    gold = strip_math(gold)
    notes = [isolation]
    duration=re.fullmatch(r'(\d+)\s+hours?\s+(?:and\s+)?(\d+)\s+minutes?',prediction,re.I)
    if duration and int(duration[2])<60:
        asks_minutes=re.search(r'how many minutes|in minutes',question,re.I)
        only_minutes='minute' in question.lower() and 'hour' not in question.lower()
        if asks_minutes or only_minutes:
            prediction=str(60*int(duration[1])+int(duration[2]))
            notes.append('duration_converted_to_requested_minutes')
    target=re.search(r'how many\s+(\w+)',question,re.I)
    parts=re.fullmatch(r'(\d+)\s+([A-Za-z ]+)\s+and\s+(\d+)\s+([A-Za-z ]+)',prediction,re.I)
    if target and parts:
        # Sum explicitly disjoint categories only when both share the noun
        # requested in the question (e.g. cheese pizzas + pepperoni pizzas).
        noun=target[1].lower().removesuffix('s')
        if all(words.lower().split()[-1].removesuffix('s')==noun for words in (parts[2],parts[4])):
            prediction=str(int(parts[1])+int(parts[3]))
            notes.append('sum_explicit_categories_of_requested_count')
    target_pair=re.search(r'how many\s+(\w+)\s+and\s+(\w+)',question,re.I)
    if target_pair and len(re.findall(r'\d+(?:\.\d+)?',prediction))==2:
        counts=[]
        for noun in target_pair.groups():
            forms={noun.lower()}
            if noun.endswith('s'): forms.add(noun[:-1].lower())
            if noun.endswith('es'): forms.add(noun[:-2].lower())
            found=re.findall(r'\b(\d+)\s+(?:'+ '|'.join(re.escape(x) for x in forms)+r')\b',prediction,re.I)
            counts.append(found)
        if all(len(x)==1 for x in counts) and re.search(r'\band\b',prediction,re.I):
            prediction=str(sum(int(x[0]) for x in counts))
            notes.append('sum_two_categories_explicitly_requested_by_question')
    rate=re.fullmatch(r'([$€]?'+NUMBER+r')\s*(?:([A-Za-z]+)\s*)?(?:/|per\s+)(month|year|day|week|hour)\.?',prediction,re.I)
    if rate and (rate[2] or prediction.startswith(('$','€'))) and re.search(rate[3],question,re.I):
        prediction=rate[1]+(' '+rate[2] if rate[2] else '')
        notes.append('scalar_rate_requested_by_question')
    if not _safe_expression(prediction):
        quantity=_declarative_quantity(prediction,question)
        if quantity:
            prediction=quantity
            notes.append('main_declarative_quantity')
    punit, gunit = _unit(prediction), _unit(gold)
    if punit or gunit:
        if punit and gunit:
            if punit[1] != gunit[1]:
                return prediction, gold, notes, 'unit_mismatch'
            prediction, gold = punit[0], gunit[0]
            notes.append('matching_explicit_units')
        elif punit and _safe_expression(gold):
            if not _context_unit(punit[1], question):
                return prediction, gold, notes, 'unit_not_supported_by_question'
            prediction = punit[0]
            notes.append('prediction_unit_from_question')
        elif gunit:
            gold = gunit[0]
            notes.append('reference_specifies_unit')
    # Keep explicit dimensional-unit mismatches above; after that, recognize
    # ordinary counts and answer sentences without choosing a number by gold.
    # A valid symbolic answer (e.g. 8-i or x-3z=0) must never be reduced
    # to its leading scalar or final RHS by the prose/count recovery rules.
    if not punit and not _safe_expression(prediction):
        ordinal=re.fullmatch(r'(\d+)(?:st|nd|rd|th)\s+place',prediction,re.I)
        if ordinal:
            prediction=ordinal[1]
            notes.append('ordinal_place')
        ordinal_sentence=re.sub(r'\b(\d+)(?:st|nd|rd|th)\s+place\b',r'\1 place',prediction,flags=re.I)
        scalar = _single_quantity_statement(ordinal_sentence)
        if scalar is None and '=' in prediction:
            # Terminal equation RHS inside an explicit final-answer region;
            # never the last number anywhere in the working-out section.
            rhs=prediction.rsplit('=',1)[1].strip()
            scalar=_single_quantity_statement(rhs)
            if scalar is not None:
                notes.append('terminal_equality_rhs_in_final_block')
        if scalar is not None:
            prediction=scalar
            notes.append('single_quantity_in_final_statement')
    if percent_context(question):
        for side, value in [('prediction', prediction), ('gold', gold)]:
            m = re.fullmatch('(' + NUMBER + r')\s*(?:\\?%|percent)?', value, re.I)
            if m:
                if side == 'prediction': prediction = m[1]
                else: gold = m[1]
        notes.append('requested_percent_points')
    def normalize(value):
        value = value.strip()
        # Thousands commas ONLY within a scalar (not tuples or root lists).
        if re.fullmatch(r'[+-]?\d{1,3}(?:,\d{3})+(?:\.\d+)?', value):
            value = value.replace(',', '')
        value = value.replace('−', '-').replace('π', r'\pi')
        literal = re.fullmatch(r'(?:\\text\{)?\s*(even|odd|Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday|\(?[A-E]\)?)\s*\}?', value, re.I)
        if literal:
            token = literal[1].strip('()').lower()
            return r'\text{' + token + '}'
        previous = None
        while previous != value:
            previous = value
            value = re.sub(r'(?<![A-Za-z\\])sqrt\(([^()]*)\)', r'\\sqrt{\1}', value)
        clock = re.fullmatch(r'(?:\\text\{)?\s*(\d{1,2}):(\d{2})\s*([ap])\.?\s*m\.?\s*\}?', value, re.I)
        if clock and 1 <= int(clock[1]) <= 12 and int(clock[2]) < 60:
            return r'\text{' + f'{int(clock[1])}:{clock[2]} {clock[3].lower()}m' + '}'
        return value
    return normalize(prediction), normalize(gold), notes, None


def _safe_expression(value):
    if not value or len(value) > 2048 or '__' in value or '<' in value or '>' in value:
        return False
    if '&' in value and not re.search(r'\\begin\{(?:[pbvBV]?matrix|array)\}',value):
        return False
    # Reject leftover natural language before a permissive LaTeX parser can
    # interpret it as multiplication. Text literals use an explicit whitelist.
    texts = re.findall(r'\\text\{([^{}]*)\}', value)
    for text in texts:
        if not (re.fullmatch(r'(?:even|odd|Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday|\(?[A-E]\)?|\d{1,2}:\d{2} [ap]m)', text, re.I)):
            return False
    residual = re.sub(r'\\(?:text|begin|end)\{[^{}]*\}', '', value)
    residual = re.sub(r'\\[A-Za-z]+', '', residual)
    if re.search(r'[A-Za-z]{3,}', residual):
        return False
    if re.search(r'\b(?:or|and|but)\b', residual, re.I):
        return False
    return True


def grade_isolated(prediction, gold, question):
    """Pure answer equivalence. Caller supplies a process-level timeout."""
    from math_verify import LatexExtractionConfig, parse, verify
    from sympy import Basic, I, Symbol
    from sympy.core.relational import Equality
    p, g, notes, error = normalize_pair(prediction, str(gold), question)
    result = dict(correct=False, normalized_prediction=p, normalized_gold=g,
                  normalization=notes, status='incorrect', error=error,
                  prediction_parsed=False, gold_parsed=False)
    if error:
        return result
    # Literal answers and ordered function values are isolated answer types,
    # not permissive LaTeX parsing of prose. Normalize each side independently.
    def literal(raw):
        value = strip_math(raw)
        wrapper = re.fullmatch(r'\\(?:text|mathrm|mbox)\{([^{}]*)\}',value)
        if wrapper:
            value = wrapper[1].strip()
        value = re.sub(r'\\(?:text|mathrm|mbox)\{([^{}]*)\}',r' \1 ',value).strip()
        value = value.strip().rstrip('.')
        if re.search(r'\b(?:not|maybe|perhaps|but|actually|or)\b',value,re.I):
            return None
        if re.fullmatch(r'even|odd|neither|parabola|circle|ellipse|hyperbola|point|line|two lines|empty|Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday', value,re.I):
            return 'literal', value.lower()
        clock = re.fullmatch(r'(\d{1,2})(?::(\d{2}))?\s*([ap])\.?\s*m\.?',value,re.I)
        if clock and 1<=int(clock[1])<=12 and int(clock[2] or '0')<60 and re.search(r'time|arriv|a\.m\.|p\.m\.',question,re.I):
            return 'clock', (int(clock[1])%12*60+int(clock[2] or '0')+(720 if clock[3].lower()=='p' else 0))
        if re.search(r'letters?|options?|diagrams?|conic|Enter\s+"[A-Z]"',question,re.I):
            letters=value.strip('() ')
            if re.fullmatch(r'[A-Z]+(?:\s*,\s*[A-Z]+)*',letters):
                if re.search(r'separated by commas',question,re.I):
                    tokens=letters.replace(' ','').split(',')
                    if len(tokens)==len(set(tokens)) and all(len(x)==1 for x in tokens):
                        return 'letter_set',tuple(sorted(tokens))
                return 'letter_sequence',letters.replace(' ','')
        if re.search(r'\bwho\b|which student',question,re.I) and re.fullmatch(r'[A-Za-z]+(?: [A-Za-z]+)?',value):
            return 'name',value.lower()
        months=r'January|February|March|April|May|June|July|August|September|October|November|December'
        date=re.fullmatch(r'('+months+r')\s+(\d{1,2})(?:st|nd|rd|th)?',value,re.I)
        if date and 1<=int(date[2])<=31 and re.search(r'date|month',question,re.I):
            return 'date',(date[1].lower(),int(date[2]))
        if re.search(r'increasing order',question,re.I):
            functions=re.split(r'\s*(?:<|>|,)\s*',value)
            if len(functions)>1 and all(re.fullmatch(r'[A-Za-z]\(\s*[+-]?\d+\s*\)',x) for x in functions):
                if '<' in value and '>' in value:
                    return None
                if '>' in value:
                    functions.reverse()
                return 'ordered_functions',tuple(re.sub(r'\s+','',x) for x in functions)
        return None
    plit, glit=literal(prediction),literal(str(gold))
    if glit is not None:
        equal=plit==glit
        result.update(correct=equal,status='correct' if equal else 'incorrect',
                      prediction_parsed=plit is not None,gold_parsed=True,method='typed_literal_exact',
                      normalized_prediction=plit,normalized_gold=glit)
        return result
    # Exact rational comparison is safe and fast for the common scalar case.
    if re.fullmatch(NUMBER, p) and re.fullmatch(NUMBER, g):
        equal = Fraction(p.replace(' ', '')) == Fraction(g.replace(' ', ''))
        result.update(correct=equal, status='correct' if equal else 'incorrect',
                      prediction_parsed=True, gold_parsed=True, method='exact_rational')
        return result
    if not _safe_expression(g):
        result.update(status='gold_unsupported', error='unsupported_reference_expression')
        return result
    if not _safe_expression(p):
        result.update(status='prediction_unsupported', error='unsupported_answer_expression')
        return result
    def parsed(value):
        return parse('$'+value+'$', extraction_config=[LatexExtractionConfig()],
                     fallback_mode='no_fallback', extraction_mode='first_match',
                     parsing_timeout=None, raise_on_error=True)
    phase = 'gold_parse'
    try:
        reference = parsed(g)
        result['gold_parsed'] = bool(reference)
        if not reference:
            result.update(status='gold_parse_failed')
            return result
        phase = 'prediction_parse'
        candidate = parsed(p)
        result['prediction_parsed'] = bool(candidate)
        if not candidate:
            result.update(status='prediction_parse_failed')
            return result
        def whole_relation(items, raw):
            # An explicitly requested value can be restated as g(2)=3 or
            # cos(36 degrees)=... . Keep that legitimate answer notation.
            if raw.count('=') == 1:
                lhs = re.sub(r'\s|\$', '', raw.split('=',1)[0])
                requests = re.split(r'\b(?:find|compute|evaluate|calculate|simplify)\s+', question, flags=re.I)[1:]
                if lhs and any(re.match(re.escape(lhs)+r'(?:[.?,$]|$)', re.sub(r'\s|\$', '', q)) for q in requests):
                    return False
            return any(isinstance(x, Equality) and not (
                (isinstance(x.lhs, Symbol) and not x.rhs.free_symbols)
                or (isinstance(x.rhs, Symbol) and not x.lhs.free_symbols)
                or (not x.lhs.free_symbols and not x.rhs.free_symbols)) for x in items)
        # Math-Verify may itself compare only an equation RHS with a scalar.
        # Permit ordinary assignments such as x=5, but preserve a whole plane
        # or other relation such as x-3z=0 rather than accepting a bare zero.
        if whole_relation(reference, g) != whole_relation(candidate, p):
            result.update(status='incorrect', error='relation_scalar_mismatch')
            return result
        if re.search(r'\bcomplex\b|imaginary|polar form|rectangular form', question, re.I):
            def canon(items):
                return [x.xreplace({s: I for s in x.free_symbols if s.name == 'i'})
                        if isinstance(x, Basic) else x for x in items]
            reference, candidate = canon(reference), canon(candidate)
            result['normalization'].append('imaginary_i_from_question')
        phase = 'verify'
        correct = bool(verify(reference, candidate, strict=True, float_rounding=10,
                              numeric_precision=15, timeout_seconds=None, raise_on_error=True))
        result.update(correct=correct, status='correct' if correct else 'incorrect', method='math_verify')
    except Exception as exc:
        result.update(status=phase+'_error', error=type(exc).__name__)
    return result


def score_answer(completion, gold, question, eos=False):
    result = dict(schema=SCHEMA, **extract_answer(completion,eos), correct=False,
                  status='extraction_failed', error=None)
    if result['extraction_error']:
        result['error'] = result['extraction_error']
        return result
    if not result['prediction']:
        result.update(error='empty_answer')
        return result
    result.update(grade_isolated(result['prediction'], gold, question))
    return result
