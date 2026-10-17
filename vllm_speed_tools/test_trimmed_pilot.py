import unittest
import torch
from trimmed_pilot_trainer import kept_width,trim_loss_inputs

class TrimTests(unittest.TestCase):
    def test_valid_tokens_retained(self):
        mask=torch.tensor([[1,1,0,0],[1,1,1,0]])
        x={'completion_mask':mask,'completion_ids':torch.arange(8).reshape(2,4),'old_per_token_logps':torch.randn(2,4),'ref_per_token_logps':torch.randn(2,4),'importance_sampling_ratio':torch.randn(2,4),'advantages':torch.randn(2),'prompt_ids':torch.ones(2,5)}
        y=trim_loss_inputs(x)
        self.assertEqual(y['completion_ids'].shape,(2,3))
        for key in ['old_per_token_logps','ref_per_token_logps','importance_sampling_ratio']:
            self.assertTrue(torch.equal(x[key][mask.bool()],y[key][y['completion_mask'].bool()]))
        self.assertIs(x['advantages'],y['advantages'])
        self.assertIs(x['prompt_ids'],y['prompt_ids'])
    def test_empty_or_noncontiguous_mask(self):
        self.assertEqual(kept_width(torch.zeros(1,9)),1)
        self.assertEqual(kept_width(torch.tensor([[1,0,1,0]])),3)

if __name__=='__main__':unittest.main()
