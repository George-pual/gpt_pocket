import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from grpo_batching import fresh_group_indices,mask_uncertain_groups
try:import torch
except ImportError:torch=None

class GroupBatchTests(unittest.TestCase):
    @unittest.skipIf(torch is None,'GPU training environment supplies PyTorch')
    def test_only_bad_question_loses_tokens_and_advantages(self):
        ids=['a']*4+['b']*4+['c']*4
        batch=dict(completion_mask=torch.ones((12,5),dtype=torch.int64),advantages=torch.arange(1,13,dtype=torch.float32))
        records=[dict(group_exclusion=i==5) for i in range(12)]
        groups,dropped=mask_uncertain_groups(batch,ids,records,3)
        self.assertEqual(dropped,dict(a=False,b=True,c=False))
        self.assertEqual(set(groups),{'a','b','c'})
        self.assertTrue((batch['completion_mask'][4:8]==0).all())
        self.assertTrue((batch['advantages'][4:8]==0).all())
        self.assertTrue((batch['completion_mask'][:4]==1).all())
        self.assertTrue((batch['advantages'][8:]==torch.arange(9,13)).all())
        self.assertEqual(batch['num_items_in_batch'],40)
    def test_missing_duplicate_and_interleaved_groups_rejected(self):
        for ids,q in [(['a']*3,1),(['a']*8,2),(['a','b']*4,2),(['a']*4,3)]:
            with self.assertRaises(ValueError):fresh_group_indices(ids,q)
    def test_single_question_compatibility(self):
        self.assertEqual(fresh_group_indices(['a']*4,1),dict(a=[0,1,2,3]))

if __name__=='__main__':unittest.main()
