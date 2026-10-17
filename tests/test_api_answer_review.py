import json,sys,unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from api_answer_review import parse_extracted_answer,MathFirstAPIReward
import math_verification as scorer
from online_grpo_reward import OnlineRewardScorer

class ExtractionTests(unittest.TestCase):
    def test_verbatim_answer_required(self):
        self.assertEqual(parse_extracted_answer('{"answer":"-2"}','stop','Final answer: -2.'),'-2')
        self.assertIsNone(parse_extracted_answer('{"answer":"2"}','stop','Final answer: three.'))
        self.assertIsNone(parse_extracted_answer('{"answer":"2"}','length','Answer 2'))
        for content in ['正确','```json\n{"answer":"2"}\n```','{"answer":null}',
                        '{"answer":"2","correct":true}','{"answer":2}',
                        '{"answer":"<SOLUTION>2</SOLUTION>"}']:
            self.assertIsNone(parse_extracted_answer(content,'stop','<SOLUTION>2</SOLUTION>'))

class RouterTests(unittest.TestCase):
    def setUp(self):
        p=patch.object(scorer,'_bounded_grade',side_effect=lambda a,b,q=None:scorer._grade_pair(a,b,None,q))
        p.start();self.addCleanup(p.stop);self.primary=OnlineRewardScorer(scorer)
    def reviewer(self,answer,status='extracted',finish='stop'):
        class Stub:
            calls=0
            def review(self,question,completion):
                self.calls+=1
                return dict(status=status,content=json.dumps({'answer':answer}),finish_reason=finish,
                            requested_model='test',returned_model='test')
        return Stub()
    def test_determinate_math_never_calls_api(self):
        reviewer=self.reviewer('2');route=MathFirstAPIReward(self.primary,reviewer)
        for answer in ['2','3']:
            r=route.score_one('Work<end_working_out><SOLUTION>'+answer+'</SOLUTION>','2','Find 1+1.',True)
            self.assertEqual(r['api_review']['status'],'not_called')
        self.assertEqual(reviewer.calls,0)
    def test_local_math_verifier_decides_and_keeps_original_format(self):
        text='Work. SOLUTION>2</SOLUTION>'
        right=MathFirstAPIReward(self.primary,self.reviewer('2')).score_one(text,'2','Find 1+1.',True)
        self.assertEqual(right['total_reward'],1.)
        self.assertEqual(right['format_bonus'],0.)
        self.assertEqual(right['verdict']['grading_source'],'math_verify_after_api_extraction')
        wrong=MathFirstAPIReward(self.primary,self.reviewer('2')).score_one(text,'3','Find 1+2.',True)
        self.assertEqual(wrong['verdict']['status'],'incorrect')
        self.assertEqual(wrong['total_reward'],0.)
    def test_api_cannot_replace_wrong_answer_with_right_answer(self):
        r=MathFirstAPIReward(self.primary,self.reviewer('2')).score_one('SOLUTION>three</SOLUTION>','2','Find 1+1.',True)
        self.assertTrue(r['group_exclusion']);self.assertEqual(r['total_reward'],0.)
    def test_false_numeric_equation_still_fails_after_extraction(self):
        r=MathFirstAPIReward(self.primary,self.reviewer('2+2=5')).score_one('SOLUTION>2+2=5</SOLUTION>','5','Find the value.',True)
        self.assertEqual(r['verdict']['status'],'incorrect')
        self.assertEqual(r['verdict']['error'],'inconsistent_numeric_equality')
        self.assertEqual(r['total_reward'],0.)
    def test_unparseable_extracted_answer_is_not_api_judged(self):
        r=MathFirstAPIReward(self.primary,self.reviewer('二')).score_one('SOLUTION>二</SOLUTION>','2','Find 1+1.',True)
        self.assertTrue(r['grading_unconfirmed']);self.assertTrue(r['group_exclusion'])
    def test_missing_failed_or_truncated_extraction_skips(self):
        for reviewer in [self.reviewer(None),self.reviewer('2','unconfirmed'),self.reviewer('2',finish='length')]:
            r=MathFirstAPIReward(self.primary,reviewer).score_one('SOLUTION>2</SOLUTION>','2','Find 1+1.',True)
            self.assertTrue(r['group_exclusion']);self.assertEqual(r['total_reward'],0.)
        reviewer=self.reviewer('2')
        MathFirstAPIReward(self.primary,reviewer).score_one('Unfinished 2','2','Find 1+1.',False)
        self.assertEqual(reviewer.calls,0)
    def test_repeated_callbacks_make_one_api_request(self):
        reviewer=self.reviewer('2');route=MathFirstAPIReward(self.primary,reviewer)
        for _ in range(3):route.score_one('SOLUTION>2</SOLUTION>','2','Find 1+1.',True)
        self.assertEqual(reviewer.calls,1)

if __name__=='__main__':unittest.main()
