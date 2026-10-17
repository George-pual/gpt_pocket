"""Scorer regressions, including plausible wrong and ambiguous answers."""
import unittest
from math_verification import _grade_pair, verify_final_answer, _normalize_answer


class ScorerV2Tests(unittest.TestCase):
    def test_equivalences(self):
        pairs = [
            ('4:30 p.m.', r'\text{4:30 p.m.}'),
            ('04:30 PM', r'\text{4:30 p.m.}'),
            ('2+sqrt(6),2-sqrt(6)', r'2+\sqrt{6},2-\sqrt{6}'),
            (r'-128-128i\sqrt{3}', r'-128-128\sqrt{3}i'),
            ('368 kilowatts', '368'), ('80 square centimeters', '80'),
            ('The car travels 70 feet.', '70'),
            ('Positive value of $b$ is 5.', '5'),
            ('Sean will need 3 new containers.', '3'),
            ('The number of two-digit primes where the sum of the digits equals 8 is 3.', '3'),
            ('(x+1)^2', 'x^2+2x+1'), (r'\frac{1}{2}', '0.5'),
        ]
        for p, g in pairs:
            with self.subTest(p=p):
                self.assertTrue(_grade_pair(p, g, None)['correct'])

    def test_wrong_or_ambiguous(self):
        pairs = [
            ('4:30 a.m.', r'\text{4:30 p.m.}'),
            ('4:31 p.m.', r'\text{4:30 p.m.}'),
            (r'-128+128i\sqrt{3}', r'-128-128\sqrt{3}i'),
            ('2+sqrt(5),2-sqrt(5)', r'2+\sqrt{6},2-\sqrt{6}'),
            ('100', r'100\%'), ('100%', '100'),
            ('367 kilowatts', '368'), ('2 feet', '2 meters'),
            ('70 or 71 feet', '70'), ('70 feet, but actually 71', '70'),
            ('The car travels 71 feet.', '70'), ('x+1', 'y+1'),
            ("__import__('os')", '17'),
        ]
        for p, g in pairs:
            with self.subTest(p=p):
                self.assertFalse(_grade_pair(p, g, None)['correct'])

    def test_final_answer_boundaries(self):
        for text in ['Work says 17 but no final answer', '<SOLUTION>17',
                     '<SOLUTION>17</SOLUTION><SOLUTION>18</SOLUTION>',
                     '<SOLUTION>17</SOLUTION>actually 18']:
            self.assertEqual(verify_final_answer(text, '17')['status'], 'extraction_failed')
        self.assertEqual(_normalize_answer('I tried 70. The answer is 71.'), 'I tried 70. The answer is 71.')


if __name__ == '__main__':
    unittest.main()
