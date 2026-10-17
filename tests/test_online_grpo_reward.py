"""Reward rules and mathematical near misses through the online scoring path."""
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import math_verification as scorer
from online_grpo_reward import OnlineRewardScorer, strict_format_ok, grading_unconfirmed


def formatted(answer, work='Calculate the requested value.'):
    return work + '<end_working_out><SOLUTION>' + answer + '</SOLUTION>'


class OnlineRewardTests(unittest.TestCase):
    def setUp(self):
        # Same public extraction/normalization/math verifier; only the timeout
        # process wrapper is removed for fast CPU unit cases. A separate runtime
        # check exercises the unpatched, pinned public interface.
        self.worker = patch.object(scorer, '_bounded_grade',
                                   side_effect=lambda p, g, q=None: scorer._grade_pair(p, g, None, q))
        self.worker.start()
        self.addCleanup(self.worker.stop)
        self.reward = OnlineRewardScorer(scorer)

    def test_three_reward_outcomes(self):
        cases = [(formatted('2'), '2', 1.1), ('Answer: 2', '2', 1.0),
                 (formatted('3'), '2', 0.0), ('Answer: 3', '2', 0.0)]
        for text, gold, expected in cases:
            with self.subTest(text=text):
                actual = self.reward.score_one(text, gold, 'Find the value.', True)
                self.assertAlmostEqual(actual['total_reward'], expected)
                if actual['verdict']['status'] == 'incorrect':
                    self.assertEqual(actual['format_reward'], 0)
                    self.assertEqual(actual['format_bonus'], 0)

    def test_valid_math_equivalence_and_near_misses(self):
        cases = [
            ('2/4', '1/2', 'Find the probability.', True),
            (r'\sqrt{50}', r'5\sqrt2', 'Find the distance.', True),
            ('(x+1)^2', 'x^2+2*x+1', 'Simplify the polynomial.', True),
            ('(2,-3)', '(2,-3)', 'Find the ordered pair.', True),
            ('(2,-3)', '(-3,2)', 'Find the ordered pair.', False),
            ('2,3', '2', 'Find all roots.', False),
            ('50%', '50', 'Find the number.', False),
            ('2 feet', '2', 'How many meters is the distance?', False),
            ('2 or 3', '2', 'Find the value.', False),
            ('2 is wrong', '2', 'Find the value.', False),
            ('The question mentions 2; I cannot determine the answer.', '2', 'Find the value.', False),
            ('2+2=5', '5', 'Find the value.', False),
            ('2+2=4', '4', 'Find the value.', True),
            ('x=2', '2', 'Find x.', True),
            ('x=2+2=5', '5', 'Find x.', False),
        ]
        for answer, gold, question, correct in cases:
            with self.subTest(answer=answer):
                r = self.reward.score_one(formatted(answer), gold, question, True)
                self.assertEqual(r['verdict']['correct'], correct)
                self.assertEqual(r['format_reward'], float(correct))

    def test_no_answer_zero_ambiguous_answer_skip(self):
        for text in ['I cannot solve the problem.', formatted('')]:
            r = self.reward.score_one(text, '2', 'Find the value.', True)
            self.assertEqual(r['total_reward'], 0)
            self.assertFalse(r['grading_unconfirmed'])
        for text in [formatted('2') + 'actually 3', '<SOLUTION>2',
                     '<SOLUTION>2</SOLUTION><SOLUTION>3</SOLUTION>', formatted('2 or 3')]:
            r = self.reward.score_one(text, '2', 'Find the value.', True)
            self.assertTrue(r['grading_unconfirmed'])
            self.assertTrue(r['group_exclusion'])
            self.assertEqual(r['total_reward'], 0)

    def test_eos_and_missing_context(self):
        r = self.reward.score_one(formatted('2'), '2', 'Find the value.', False)
        self.assertEqual(r['format_bonus'], 0)
        self.assertTrue(r['group_exclusion'])
        with self.assertRaises(ValueError):
            self.reward.score_one(formatted('2'), '2', None, True)
        with self.assertRaises(ValueError):
            self.reward.score_batch(['Answer: 2'], ['2'], [], [True])

    def test_runtime_exceptions_never_earn_bonus(self):
        for status in ['prediction_parse_failed', 'prediction_parse_error', 'verify_error', 'verification_timeout']:
            fake = types.SimpleNamespace(verify_final_answer=lambda *args, **kwargs:
                                         dict(status=status, correct=False, error='test'),
                                         completion_text=scorer.completion_text)
            r = OnlineRewardScorer(fake).score_one(formatted('2'), '2', 'Find the value.', True)
            self.assertTrue(r['group_exclusion'])
            self.assertEqual(r['total_reward'], 0)
        for status in ['gold_parse_failed', 'gold_parse_error']:
            fake = types.SimpleNamespace(verify_final_answer=lambda *args, **kwargs:
                                         dict(status=status, correct=False))
            with self.assertRaises(RuntimeError):
                OnlineRewardScorer(fake).score_one(formatted('2'), '2', 'Find the value.', True)

    def test_reference_support_and_reward_preflight(self):
        self.assertTrue(self.reward.preflight()['passed'])
        checks = self.reward.validate_references([dict(row_id='a', answer='2', problem='Find the value.')])
        self.assertEqual(checks['a']['status'], 'correct')


class FormatTests(unittest.TestCase):
    def test_strict_shape(self):
        self.assertTrue(strict_format_ok(formatted('2'), True))
        self.assertTrue(strict_format_ok(formatted('x < 2'), True))
        self.assertTrue(strict_format_ok(formatted('2') + '<|endoftext|>', True))
        for text in [formatted(''), formatted('2', ''), formatted('2') + 'more text',
                     '<start_working_out>' + formatted('2'), formatted('2').replace('SOLUTION', 'solution'),
                     'Work<end_working_out>extra<SOLUTION>2</SOLUTION>']:
            self.assertFalse(strict_format_ok(text, True), text)


if __name__ == '__main__':
    unittest.main()
