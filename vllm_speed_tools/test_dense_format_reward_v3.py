import unittest
from dense_format_reward_v3 import components, normalized_reward

class DenseFormatTests(unittest.TestCase):
    def test_components(self):
        valid = 'Reasoning.<end_working_out><SOLUTION>42</SOLUTION>'
        self.assertAlmostEqual(normalized_reward(valid,True),1)
        self.assertEqual(normalized_reward(valid,False),0)
        self.assertEqual(components(valid.replace('<end_working_out>','</start_working_out>'),True), {'boundary':0.0,'answer':0.03,'whole':0.0})
        self.assertEqual(components('Reasoning.<end_working_out>',True), {'boundary':0.05,'answer':0.0,'whole':0.0})
        for bad in ['<end_working_out><SOLUTION>42</SOLUTION>',valid+'<end_working_out>',valid+'<SOLUTION>42</SOLUTION>',valid.replace('42',''),valid.replace('42','   '),valid+'extra', '<SOLUTION>42</SOLUTION>Reasoning.<end_working_out>']:
            self.assertEqual(components(bad,True)['whole'],0)
            self.assertLess(normalized_reward(bad,True),1)
        self.assertEqual(components(valid+'<end_working_out>',True)['boundary'],0)
        self.assertEqual(components(valid+'<SOLUTION>42</SOLUTION>',True)['answer'],0)

if __name__ == '__main__': unittest.main()
