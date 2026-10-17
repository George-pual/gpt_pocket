"""Independent positive/negative regressions, not scorer implementation mirrors."""
import importlib.util
import sys
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from math_verification import _grade_pair, verify_final_answer, normalize_final_candidate

spec=importlib.util.spec_from_file_location('legacy_scorer_tests',ROOT/'closure_grpo_experiment/test_math_verification_v2.py')
legacy=importlib.util.module_from_spec(spec)
spec.loader.exec_module(legacy)


class LegacyRegressionTests(legacy.ScorerV2Tests):
    pass


class ScorerV3Tests(unittest.TestCase):
    def test_clear_answers(self):
        pairs=[
            (r'The simplified form is \((z-2)^2\).','z^2-4z+4'),
            ('Therefore, the value of K is 9.','9'),
            ('There are 12 distinct ways to choose the members.','12'),
            ('Ravi is 7 years old.','7'),
            ('300 bows are green.','300'),
            ('The area of the square is 26 square units.','26'),
            ('The coordinates of point D are (1, -2, 8).','(1,-2,8)'),
            ('The maximum number of odd integers is four.','4'),
            ('Three statements can be true at the same time.','3'),
            ('The correct graph is the one labeled as "B".','B'),
            ('The correct answer is E) 65.14444.','E'),
            (r'Rectangular form: $1-i$','1-i'),
            ('6√11',r'6\sqrt{11}'),
            ('-√3/2',r'-\frac{\sqrt3}{2}'),
            ('2<WORKING_OUT>','2'),
            ('50\n$','50'),
            (r'$\$$23.40','23.40'),
            ('The graph of the equation is a non-degenerate hyperbola.','hyperbola'),
            ('Approximately 77.14°</SMALLGPT2>','77.14'),
            ('The greatest number of pizzas that can be made is 15.','15'),
            ('First we organize the choices. Therefore, there are 24 distinct numbers.','24'),
            ('9+9+9=27.Answer: 27.','27'),
            (r'We calculate. The answer is \boxed{\frac13}. Verification gives \boxed{\frac13}.',r'\frac13'),
            ('It would take Molly 1350 more minutes than Xanthia to finish the book.','1350'),
            ('4 * 3 * 2 = 24 distinct three-digit numbers, since there are four options.','24'),
            ('There are 2 positive integer values of x satisfying the inequality.','2'),
            ('The polygon has \\( 6 \\) sides.','6'),
            ('36 π cubic inches\\',r'36\pi'),
            ('$ = 2.18$','2.18'),
            ('10\\','10'),
            ('We factor the integer.\n\nThe largest prime factor is 13.','13'),
        ]
        for p,g in pairs:
            with self.subTest(prediction=p):
                self.assertTrue(_grade_pair(p,g,None)['correct'])

    def test_wrong_answers_remain_wrong(self):
        pairs=[
            ('The value of K is 8.','9'),
            ('There are 13 distinct ways to choose the members.','12'),
            ('300 bows are green.','15'),
            ('The correct answer is E) 65.14444.','C'),
            ('Ravi is 8 years old.','7'),
            ('The area of the square is 26 square units.','27'),
            ('The coordinates of point D are (1, -2, 7).','(1,-2,8)'),
            ('The answer is 18. The question mentions 17.','17'),
            ('I tried 17. Therefore, the value is 18.','17'),
            ('The number 17 is in the question; I cannot determine the answer.','17'),
            ('Hint: look at 17.','17'),
            ('The value is 70 or 71 feet.','70'),
            ('The answer is 70. The answer is 71.','70'),
            (r'\boxed{70} and \boxed{71}','71'),
            ('The car travels 70 feet, but actually 71.','70'),
            ('-$x-1$','1-x'),
            ('>48','48'),
            ('2√2/4 ≈ 0.707 or 4/sqrt(16) ≈ 0.447','0.707'),
            ('The maximum could be 4 or 5.','4'),
            ('The answer is 100 percent.','100'),
            ('0.5','50'),
            ('50%','50'),
            ('The distance is 2 feet.','2 meters'),
            ('The inverse does not exist.','0'),
            ("__import__('os').system('echo 17')",'17'),
            ('There are 70 ways, maybe 71.','70'),
            ('The answer is 70, which is incorrect.','70'),
            ('The value is 70, although 71 is also possible.','70'),
        ]
        for p,g in pairs:
            with self.subTest(prediction=p):
                self.assertFalse(_grade_pair(p,g,None)['correct'])

    def test_context_is_not_gold(self):
        self.assertEqual(normalize_final_candidate('The value of x is 8.')[0],'8')
        self.assertTrue(_grade_pair('No positive real solutions.','0',None,'How many positive real solutions are there?')['correct'])
        self.assertFalse(_grade_pair('No positive real solutions.','0',None,'Find the positive real solutions.')['correct'])
        self.assertTrue(_grade_pair('The volume ratio is 1:4.',r'\frac14',None,'What is the volume ratio?')['correct'])
        self.assertFalse(_grade_pair('The distance is 2 meters.','2',None,'How many feet is the distance?')['correct'])
        self.assertTrue(_grade_pair('The distance is 2 feet.','2',None,'How many feet is the distance?')['correct'])
        self.assertFalse(_grade_pair('2 square feet','2',None,'How many cubic feet are in the volume?')['correct'])
        self.assertFalse(_grade_pair('90°','90',None,'Find the angle in radians.')['correct'])
        self.assertTrue(_grade_pair('The angle is 90 degrees.','90',None,'Find the angle in degrees.')['correct'])
        self.assertTrue(_grade_pair('90',r'90^\circ',None,'Find the angle in degrees.')['correct'])

    def test_boundaries_still_fail_closed(self):
        for text in ['Work says 17 but no final answer','<SOLUTION>17',
                     '<SOLUTION>17</SOLUTION><SOLUTION>18</SOLUTION>',
                     '<SOLUTION>17</SOLUTION>actually 18']:
            self.assertEqual(verify_final_answer(text,'17')['status'],'extraction_failed')
        for text in [r'\boxed{17} but actually 18',r'\boxed{17} and \boxed{18}']:
            self.assertFalse(verify_final_answer(text,'17')['correct'])


if __name__=='__main__':
    unittest.main()
