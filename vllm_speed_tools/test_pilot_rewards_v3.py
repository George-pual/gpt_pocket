import unittest
from pilot_rewards_v3 import numeric_answer,grade,reward_pair

class Rewards(unittest.TestCase):
    def test_units_scales_punctuation(self):
        for text,gold in [('42 pages','42'),('85.','85'),('$1,000','1000'),('4 million dollars','4000000'),(r'\frac{1}{2}','0.5')]:
            self.assertEqual(numeric_answer(text),numeric_answer(gold))
        self.assertNotEqual(numeric_answer('4 million dollars'),numeric_answer('4'))
        self.assertIsNone(numeric_answer('42 or 43'))
        self.assertIsNone(numeric_answer('42 + 100'))
        self.assertNotEqual(numeric_answer('50%'),numeric_answer('50'))
    def test_unique_answer(self):
        self.assertTrue(grade([{'role':'assistant','content':'<SOLUTION>42 pages</SOLUTION>'}],'42','gsm8k')['correct'])
        self.assertFalse(grade('<SOLUTION>42</SOLUTION><SOLUTION>43</SOLUTION>','42','gsm8k')['correct'])
        self.assertFalse(grade('I considered 42, but no final answer','42','gsm8k')['correct'])
        self.assertFalse(grade(r'\boxed{41} and \boxed{42}','42','gsm8k')['correct'])
    def test_reward_matrix(self):
        for number,ending,expected in [('42','<end_working_out>',(1.,.1)),('42','</start_working_out>',(1.,0.)),('43','<end_working_out>',(0.,.1)),('43','</start_working_out>',(0.,0.))]:
            c='Compute. '+ending+'<SOLUTION>'+number+'</SOLUTION>'
            self.assertEqual(reward_pair(c,'42','gsm8k',True,.1)[:2],expected)
        self.assertEqual(reward_pair('Compute. <end_working_out><SOLUTION>42</SOLUTION>','42','gsm8k',False,.1)[:2],(1.,0.))

if __name__=='__main__': unittest.main()
