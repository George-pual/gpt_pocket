"""Regression for the actual text-list failure and misleading near misses."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from math_verification import verify_final_answer, _grade_pair, SCHEMA

QUESTION = ('List the names of the choices from most popular to least popular, '
            'separated by commas.')
GOLD = r'\text{Softball, Kickball, Picnic}'


class ScorerV4Tests(unittest.TestCase):
    def test_correct_ordered_names(self):
        for prediction in ['Softball, Kickball, Picnic', ' softball,  KICKBALL, Picnic ',
                           r'\text{Softball, Kickball, Picnic}', '$Softball, Kickball, Picnic$']:
            with self.subTest(prediction=prediction):
                result = _grade_pair(prediction, GOLD, None, QUESTION)
                self.assertEqual(result['status'], 'correct')
                self.assertTrue(result['correct'])

    def test_wrong_order_missing_extra_and_duplicate(self):
        for prediction in ['Kickball, Picnic, Softball', 'Softball, Picnic, Kickball',
                           'Softball, Kickball', 'Softball, Kickball, Picnic, Soccer',
                           'Softball, Softball, Picnic', 'Softbal, Kickball, Picnic']:
            with self.subTest(prediction=prediction):
                self.assertEqual(_grade_pair(prediction, GOLD, None, QUESTION)['status'], 'incorrect')

    def test_no_prose_salvage_or_alternative_answer(self):
        for prediction in ['Softball, Kickball, Picnic or Soccer',
                           'The answer is Softball, Kickball, Picnic',
                           'Kickball, Picnic, Softball. Actually Softball, Kickball, Picnic',
                           'Softball, Kickball, Picnic, maybe',
                           r'\text{Softball, Kickball, Picnic}\text{Soccer}']:
            with self.subTest(prediction=prediction):
                result = _grade_pair(prediction, GOLD, None, QUESTION)
                self.assertEqual(result['status'], 'prediction_parse_failed')
                self.assertFalse(result['correct'])

    def test_numeric_lists_still_use_mathematical_verifier(self):
        self.assertTrue(_grade_pair('12, 18', '12,18', None, 'Enter the two smallest solutions, separated by commas.')['correct'])
        self.assertFalse(_grade_pair('12, 18', '18', None, 'Enter the two smallest solutions, separated by commas.')['correct'])
        self.assertTrue(_grade_pair('1/2, 2', r'\frac12,2', None, 'Find all the roots.')['correct'])

    def test_actual_public_extraction_and_schema(self):
        r = verify_final_answer('<SOLUTION>Softball, Kickball, Picnic</SOLUTION>', GOLD, QUESTION)
        self.assertEqual(r['schema'], SCHEMA)
        self.assertTrue(r['correct'])
        self.assertFalse(verify_final_answer('<SOLUTION>Softball, Kickball, Picnic</SOLUTION>actually Soccer', GOLD, QUESTION)['correct'])


if __name__ == '__main__':
    unittest.main()
