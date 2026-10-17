"""Online GRPO rewards: mathematical correctness gates a strict format bonus.

The frozen mathematical verifier remains separate from protocol checking.
Ambiguous extraction/verification is not evidence of mathematical error.
"""
from functools import lru_cache
import re


POLICY = 'correctness_gated_strict_format_v1'
FORMAT_WEIGHT = 0.1
CONFIRMED_ZERO_ERRORS = {'empty_answer', 'missing_final_answer'}


def strict_format_ok(text, ended):
    """START is supplied by the generation prompt, not by the completion."""
    if not ended:
        return False
    text = re.sub(r'(?:<\|endoftext\|>|<\|im_end\|>|</s>)\s*$', '', text.strip()).strip()
    tags = re.findall(r'<\s*/?\s*(?:start_working_out|end_working_out|SOLUTION)\b[^>]*>', text, re.I)
    if tags != ['<end_working_out>', '<SOLUTION>', '</SOLUTION>']:
        return False
    match = re.fullmatch(r'(?P<work>.*?)<end_working_out>\s*<SOLUTION>(?P<answer>.*?)</SOLUTION>', text, re.S)
    return bool(match and match['work'].strip() and match['answer'].strip())


def grading_unconfirmed(verdict):
    status = verdict['status']
    if status in {'correct', 'incorrect'}:
        return False
    return not (status == 'extraction_failed' and verdict.get('error') in CONFIRMED_ZERO_ERRORS)


def _numeric_expression(text):
    """Recognize closed numeric notation; never reinterpret variables as numbers."""
    text = re.sub(r'\\(?:dfrac|tfrac|frac|sqrt|pi|cdot|times|left|right|sin|cos|tan|log|ln|exp)\b', '', text)
    text = re.sub(r'\b(?:sqrt|sin|cos|tan|log|ln|exp|pi|e|i)\b', '', text)
    return bool(text.strip()) and not re.search(r'[A-Za-z]|\\', text)


def check_numeric_equalities(scorer, text):
    """Math-Verify may equate an equation to its RHS even when it is false.

    Check closed numerical steps separately (2+2=5 must not earn reward).
    Symbolic constraints such as x=2 are not numerical identities.
    """
    answer, _, error = scorer.extract_final_answer(text)
    if error or not answer or '=' not in answer:
        return []
    parts = re.split(r'(?<![<>!])=(?!=)', answer)
    checks = []
    for left, right in zip(parts, parts[1:]):
        if _numeric_expression(left) and _numeric_expression(right):
            v = scorer.verify_final_answer('<SOLUTION>' + left.strip() + '</SOLUTION>', right.strip())
            checks.append(dict(left=left.strip(), right=right.strip(), verdict=v))
    return checks


class OnlineRewardScorer:
    def __init__(self, scorer):
        self.scorer = scorer

    @lru_cache(maxsize=4096)
    def score_one(self, text, gold, question, ended):
        if not isinstance(question, str) or not question.strip():
            raise ValueError('Question text is required for contextual mathematical grading')
        raw_verdict = self.scorer.verify_final_answer(text, gold, question=question)
        verdict = dict(raw_verdict)
        if verdict['status'].startswith('gold_'):
            raise RuntimeError('Unsupported reference answer in online reward')
        if verdict['status'] == 'correct' and not verdict['correct']:
            raise RuntimeError('Inconsistent mathematical verdict')
        if verdict['correct'] and verdict['status'] != 'correct':
            raise RuntimeError('Unconfirmed verdict claimed correctness')
        equality_checks = check_numeric_equalities(self.scorer, text) if verdict['status'] in {'correct', 'incorrect'} else []
        if any(c['verdict']['status'] == 'incorrect' for c in equality_checks):
            verdict.update(status='incorrect', correct=False, error='inconsistent_numeric_equality')
        elif any(c['verdict']['status'] != 'correct' for c in equality_checks):
            verdict.update(status='prediction_parse_failed', correct=False, error='numeric_equality_unconfirmed')
        correct = verdict['status'] == 'correct' and bool(verdict['correct'])
        format_valid = strict_format_ok(text, ended)
        correctness_reward = float(correct)
        format_reward = float(correct and format_valid)
        return dict(verdict=verdict, raw_verdict=raw_verdict, numeric_equality_checks=equality_checks,
                    correctness_reward=correctness_reward,
                    format_valid=format_valid, format_reward=format_reward,
                    format_bonus=FORMAT_WEIGHT * format_reward,
                    total_reward=correctness_reward + FORMAT_WEIGHT * format_reward,
                    grading_unconfirmed=grading_unconfirmed(verdict),
                    group_exclusion=not ended or grading_unconfirmed(verdict),
                    reward_policy=POLICY)

    def score_batch(self, completions, answers, questions, ended):
        n = len(completions)
        if not (len(answers) == len(questions) == len(ended) == n):
            raise ValueError('Completion/reference/question/EOS batch sizes differ')
        return [self.score_one(self.scorer.completion_text(c), str(g), p, bool(e))
                for c, g, p, e in zip(completions, answers, questions, ended)]

    def validate_references(self, rows):
        results = {}
        for row in rows:
            verdict = self.scorer.verify_final_answer('<SOLUTION>' + row['answer'] + '</SOLUTION>',
                                                      row['answer'], question=row['problem'])
            results[row['row_id']] = verdict
        return results

    def preflight(self):
        cases = [
            ('1/2 = 2/4.<end_working_out><SOLUTION>2/4</SOLUTION>', '1/2', 1.1, False),
            ('1+1=3.<end_working_out><SOLUTION>3</SOLUTION>', '2', 0.0, False),
            ('Work<end_working_out><SOLUTION>2+2=5</SOLUTION>', '5', 0.0, False),
            ('Answer: 2', '2', 1.0, False),
            ('Work mentions 2; no final answer.', '2', 0.0, False),
            ('Work<end_working_out><SOLUTION>2 or 3</SOLUTION>', '2', 0.0, True),
            ('Work<end_working_out><SOLUTION>2</SOLUTION>actually 3', '2', 0.0, True),
        ]
        for text, gold, reward, unknown in cases:
            record = self.score_one(text, gold, 'Find the value.', True)
            if abs(record['total_reward'] - reward) > 1e-12 or record['grading_unconfirmed'] != unknown:
                raise RuntimeError('Online reward preflight failed')
        return dict(policy=POLICY, cases=len(cases), passed=True)
