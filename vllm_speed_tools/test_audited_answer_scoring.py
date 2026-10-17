"""Gold-independent extraction and positive/negative scorer regressions."""
import unittest
from audited_answer_scoring import extract_answer, grade_isolated, score_answer


class AuditedScoringTests(unittest.TestCase):
    def test_expanded_math_answer_types(self):
        positives=[
            ('Evelyn',r'\text{Evelyn}','Which student has the greatest speed?'),
            ('Dana',r'\text{Dana}','Who has larger meals?'),
            ('575',r'575\text{ students}','Find enrollment in students.'),
            ('81 teachers',r'81\text{ teachers}','How many teachers?'),
            ('65',r'65\text{ children tickets }',"How many children's tickets?"),
            ('18',r'18\text{ ways.}','How many ways?'),
            ('0.25 centimeters',r'0.25\,\text{ cm}','How many centimeters?'),
            ('45',r'45\text{ units}^2','Find area in square units.'),
            ('13',r'\text{13}','How many miles?'),
            ('25 kilometers',r'25\,\text{km}','How many kilometers?'),
            ('12 grams',r'12\text{gm}','Find weight in grams.'),
            ('45 mph',r'45\text{mph}','Find average speed in mph.'),
            ('9:00 AM',r'9\text{ a.m.}','What time does it arrive?'),
            ('June 20th',r'\text{June 20}','On which date during the month of June?'),
            ('neither',r'\text{neither}','Is it even, odd, or neither?'),
            ('ellipse',r'\text{ellipse}','Determine the graph type.'),
            ('parabola',r'\text{parabola}','Determine the graph type.'),
            ('P',r'\text{(P)}','Enter "P" for parabola; which conic?'),
            ('M',r'\text{M}','What letter is in this position?'),
            ('ACF',r'\text{ACF}','Express answer as a block of letters in alphabetic order.'),
            ('D,A',r'\text{A,D}','Enter the letters of the diagrams, separated by commas.'),
            ('Saturday',r'\mbox{Saturday}','What day of the week?'),
            ('f(2), f(1), f(4)','f(2) < f(1) < f(4)','Enter in increasing order.'),
            ('f(4)>f(1)>f(2)','f(2) < f(1) < f(4)','Enter in increasing order.'),
        ]
        negatives=[
            ('Evelyn or Dana',r'\text{Evelyn}','Which student has the greatest speed?'),
            ('Anna',r'\text{Dana}','Who has larger meals?'),
            ('9 pm',r'9\text{ a.m.}','What time does it arrive?'),
            ('June 21',r'\text{June 20}','On which date during the month of June?'),
            ('even',r'\text{neither}','Is it even, odd, or neither?'),
            ('A,D,E',r'\text{A,D}','Enter the letters of the diagrams, separated by commas.'),
            ('A,A,D',r'\text{A,D}','Enter the letters of the diagrams, separated by commas.'),
            ('FCA',r'\text{ACF}','Express answer as a block of letters in alphabetic order.'),
            ('25 meters',r'25\text{km}','How many kilometers?'),
            ('0.25 meters',r'0.25\text{cm}','How many centimeters?'),
            ('45 units',r'45\text{units}^2','Find area in square units.'),
            ('f(2),f(4),f(1)','f(2)<f(1)<f(4)','Enter in increasing order.'),
        ]
        for p,g,q in positives:
            with self.subTest(p=p,g=g):
                self.assertTrue(grade_isolated(p,g,q)['correct'])
        for p,g,q in negatives:
            with self.subTest(p=p,g=g):
                self.assertFalse(grade_isolated(p,g,q)['correct'])

    def test_equivalences(self):
        pairs = [
            ('4:30 p.m.', r'\text{4:30 p.m.}', 'What time?'),
            ('04:30 PM', r'\text{4:30 p.m.}', 'What time?'),
            ('2+sqrt(6),2-sqrt(6)', r'2+\sqrt{6},2-\sqrt{6}', 'Find both roots.'),
            (r'-128-128i\sqrt{3}', r'-128-128\sqrt{3}i', 'Find this complex number.'),
            ('368 kilowatts', '368', 'How many kilowatts?'),
            ('80 square centimeters', '80', 'Area in square centimeters?'),
            ('The car travels 70 feet.', '70', 'How many feet?'),
            ('Positive value of $b$ is 5.', '5', 'Find positive b.'),
            ('Sean will need 3 new containers.', '3', 'How many containers?'),
            ('The degree of f is 3. Therefore, the degree of 2f(x)+4g(x) is 5.', '5', 'What is the degree?'),
            (r'The probability is \( \frac{13}{204} \).', r'\dfrac{13}{204}', 'Find the probability.'),
            ('(x+1)^2', 'x^2+2x+1', 'Expand.'), (r'\frac{1}{2}', '0.5', 'Find the value.'),
            ('90%', '90', 'What percent of households?'),
            ('100', r'100\%', 'Express your answer as a percent.'),
            ('0.9', '90%', 'Find the probability.'),
            ('1,234', '1234', 'How many?'), ('$18', '18', 'How many dollars?'),
            ('Friday', r'\text{Friday}', 'What day?'),
            ("Allen's age 10 years from now is 109.", '109', 'How old will Allen be?'),
            ('Frankie watched 3 30-minute episodes on Wednesday.', '3', 'How many episodes?'),
            ('It takes Grandma 50 minutes to complete the entire 3-mile walk.', '50', 'How many minutes?'),
            ('Bob will have 26 marbles after losing 10 of them.', '26', 'How many marbles?'),
            ('The boat is worth $3528 after the three years.', '3528', 'What is the boat worth in dollars?'),
            ('Tom could have bought 11 more CDs if he had not bought the headphones.', '11', 'How many CDs if he decided not to buy headphones?'),
            ('156kgs', '156', 'Each item weighs 2kgs. Find total weight.'),
            ('€77', '77', 'How much is left from €100?'),
            ('Finley finished in 6th place.', '6', 'What place?'),
            ('72 Mb per hour', '72', 'How many megabytes per hour?'),
            ('5 cars drove through in the first 15 minutes.', '5', 'How many cars drove through?'),
            ('16 people think that horse #12 will win.', '16', 'How many people?'),
            ('The calculation is 130 - 39 = 91 trees.', '91', 'How many trees?'),
            ('Calculations: 100 - 30 = 70\nSandy took 720 minutes to walk.', '720', 'How many minutes?'),
            ('3 hours and 47 minutes', '227', 'How many minutes?'),
            ('3 cheese pizzas and 3 pepperoni pizzas', '6', 'How many pizza pies?'),
            ('Sophia bought a total of 3 onions and 1 potato at the market.', '4', 'How many onions and potatoes in total?'),
            ('86 glasses and 42 plates', '128', 'How many glasses and plates?'),
            ('Work: 5 * $3 = $15. So, Sam would have to spend $60 to make 4 liters of juice.', '60', 'How much money from $100?'),
            ('8-i', '8-i', 'Simplify $(2+3i)(1-2i)$.'),
            ('-2-14i', '-2-14i', 'Simplify $-3(1+4i)+i(-2-i)$.'),
            ('-1 - 5i', '-1 - 5i', 'Find the complex number.'),
            ('x - 3z = 0', 'x - 3z = 0', 'Enter the equation of the plane.'),
            ('3z = x', 'x - 3z = 0', 'Enter the equation of the plane.'),
            ('g(2) = 3', '3', 'Consider $g(x)=3$. Find $g(2)$.'),
            (r'\cos 36^\circ = \frac{\sqrt{5}+1}{4}', r'\frac{1+\sqrt{5}}{4}', r'Compute $\cos 36^\circ.$'),
        ]
        for p, g, q in pairs:
            with self.subTest(p=p, g=g):
                self.assertTrue(grade_isolated(p, g, q)['correct'])

    def test_negative_controls(self):
        pairs = [
            ('4:30 a.m.', r'\text{4:30 p.m.}', 'What time?'),
            (r'-128+128i\sqrt{3}', r'-128-128\sqrt{3}i', 'Find the complex number.'),
            ('2+sqrt(5),2-sqrt(5)', r'2+\sqrt{6},2-\sqrt{6}', 'Find roots.'),
            ('2+sqrt(6)', r'2+\sqrt{6},2-\sqrt{6}', 'Find both roots.'),
            ('1,2,3', '1,2', 'Find all roots.'),
            ('(1,2)', '(2,1)', 'Find the ordered pair.'),
            ('100', r'100\%', 'Find the probability.'),
            ('1%', '1', 'Find the probability.'),
            ('0.9', '90', 'What percent of households?'),
            ('2 feet', '2 meters', 'Find the length.'),
            ('2 cents', '2', 'How many dollars?'),
            ('70 or 71 feet', '70', 'How many feet?'),
            ('70 feet, but actually 71', '70', 'How many feet?'),
            ('The car travels 71 feet.', '70', 'How many feet?'),
            ('x+1', 'y+1', 'Expand.'), ('x+1', 'x+2', 'Expand.'),
            ("__import__('os')", '17', 'Find the value.'),
            ('I tried 17. Then 18.', '17', 'Find the value.'),
            ('The answer is 17 but actually 18.', '17', 'Find the value.'),
            ('The answer is not 17.', '17', 'Find the value.'),
            ('Maybe the answer is 17.', '17', 'Find the value.'),
            ('The car travels 2 feet.', '2 meters', 'Find distance.'),
            ('1.000000001', '1', 'Find the exact value.'),
            ('5 cars drove through in the first 15 minutes.', '15', 'How many cars drove through?'),
            ('Bob will have 26 marbles after losing 10 of them.', '10', 'How many marbles?'),
            ('17 apples and 18 pears', '17', 'How many fruit?'),
            ('3 hours and 47 minutes', '3.47', 'How many hours?'),
            ('3 cheese pizzas and 3 pepperoni pizzas', '3', 'How many pizza pies?'),
            ('86 glasses and 42 plates', '86', 'How many glasses and plates?'),
            ('Bob has 17. actually 18.', '17', 'How many?'),
            ('Bob has 17, 18.', '17', 'How many?'),
            ('8-i', '8', 'Find the value.'),
            ('-2-14i', '-2', 'Find the value.'),
            ('x - 3z = 0', '0', 'Find the value.'),
            ('0', 'x - 3z = 0', 'Enter the equation of the plane.'),
            ('x + 3z = 0', 'x - 3z = 0', 'Enter the equation of the plane.'),
        ]
        for p, g, q in pairs:
            with self.subTest(p=p, g=g):
                self.assertFalse(grade_isolated(p, g, q)['correct'])

    def test_boundaries(self):
        for text in ['Work says 17 but no final answer', '<SOLUTION>17',
                     '<SOLUTION>17</SOLUTION><SOLUTION>18</SOLUTION>',
                     '<SOLUTION>17</SOLUTION>actually 18',
                     r'Work \boxed{17}, but the actual answer is 18',
                     'We computed 17. Try again and get 18.']:
            with self.subTest(text=text):
                self.assertIsNotNone(extract_answer(text)['extraction_error'])
        self.assertTrue(score_answer('<SOLUTION>1/2</SOLUTION>', '0.5', 'Value?')['correct'])
        recovered = score_answer(r'<SOLUTION>work <SOLUTION>\boxed{-5/7}', '-5/7', 'Value?')
        self.assertTrue(recovered['correct'])
        self.assertIsNotNone(recovered['tag_issue'])
        self.assertFalse(score_answer('<SOLUTION>17 or 18</SOLUTION>', '17', 'Value?')['correct'])
        partial=score_answer('<SOLUTION>Each person receives $5.', '5', 'How many dollars?', eos=True)
        self.assertTrue(partial['correct'])
        self.assertIsNotNone(partial['tag_issue'])
        self.assertFalse(score_answer('<SOLUTION>17', '17', 'Value?', eos=False)['correct'])
        self.assertFalse(score_answer('Work says 17 and then ends in 18', '18', 'Value?', eos=True)['correct'])
        terminal=score_answer('Percentage of puppies = 35%</SOLUTION>', '35', 'What percent?', eos=True)
        self.assertTrue(terminal['correct'])
        self.assertIsNotNone(terminal['tag_issue'])
        self.assertFalse(score_answer('Last calculation = 35', '35', 'Value?', eos=False)['correct'])
        self.assertFalse(score_answer('Last calculation = 35, but actually 36', '35', 'Value?', eos=True)['correct'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
