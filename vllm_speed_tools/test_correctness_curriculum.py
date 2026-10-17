import json
from pathlib import Path
from types import SimpleNamespace, ModuleType
import sys
import tempfile
import unittest
from unittest.mock import patch

from correctness_curriculum import difficulty_decision, informative_indices, apply_selected_update


class SelectionTests(unittest.TestCase):
    def test_threshold_boundaries(self):
        self.assertEqual([n for n in range(17) if difficulty_decision(n)['keep']],list(range(4,12)))

    def test_filter_uses_correctness(self):
        keep,counts = informative_indices([0,0,0,0, 1,1,1,1, 0,1,0,0, 1,1,0,1])
        self.assertEqual(keep,list(range(8,16)))
        self.assertEqual(counts,{'all_wrong':1,'all_correct':1,'mixed':2})

    def test_empty_selection_no_call(self):
        called = []
        self.assertFalse(apply_selected_update([],lambda x:called.append(x)))
        self.assertEqual(called,[])

    def test_incomplete_groups_fail(self):
        with self.assertRaises(ValueError):
            informative_indices([1,0,1])

    def test_real_loop_with_cpu_fake_trainer(self):
        import torch
        from filtered_training import run_filtered_training
        fake_utils = ModuleType('transformers.trainer_utils')
        fake_utils.TrainOutput = lambda step,loss,metrics:SimpleNamespace(global_step=step,metrics=metrics)
        fake_tb = ModuleType('torch.utils.tensorboard')
        fake_tb.SummaryWriter = lambda *a:SimpleNamespace(add_scalar=lambda *a:None,close=lambda:None)
        class FakeState:
            global_step = 0
            def __init__(self):self.log_history=[]
            def save_to_json(self,path):Path(path).write_text('{}')
        class FakeTrainer:
            def __init__(self,path,correctness):
                self.args=SimpleNamespace(bf16=True,loss_type='grpo',num_iterations=1,
                    num_generations=4,per_device_train_batch_size=2,gradient_accumulation_steps=4,
                    max_steps=2,generation_batch_size=16,data_seed=3407,output_dir=path,max_grad_norm=10.,save_steps=25)
                self.accelerator=SimpleNamespace(num_processes=1,
                    prepare=lambda *x:x,autocast=lambda:torch.autocast('cpu',enabled=False),
                    clip_grad_norm_=torch.nn.utils.clip_grad_norm_)
                self.model=torch.nn.Linear(1,1,bias=False)
                self.model.weight.data.fill_(1.)
                self.use_liger_loss=False
                self.state=FakeState()
                self._metrics={'train':{}}
                self.processing_class=SimpleNamespace(save_pretrained=lambda path:None)
                self.correctness=correctness
                self.calls=0
            def create_optimizer_and_scheduler(self,num_training_steps):
                self.optimizer=torch.optim.SGD(self.model.parameters(),lr=.1)
                self.lr_scheduler=torch.optim.lr_scheduler.LambdaLR(self.optimizer,lambda _:1.)
            def _generate_and_score_completions(self,inputs):
                self._latest_binary_correctness=self.correctness
                n=len(inputs)
                return {'completion_ids':torch.ones(n,2,dtype=torch.long),
                        'completion_mask':torch.ones(n,2),'advantages':torch.ones(n)}
            def compute_loss(self,model,batch):
                self.calls+=len(batch['advantages'])
                return model.weight.mean()  # derivative exactly one after weighted accumulation
            def save_model(self,path):Path(path).mkdir(parents=True,exist_ok=True)
        modules={'transformers.trainer_utils':fake_utils,'torch.utils.tensorboard':fake_tb}
        with tempfile.TemporaryDirectory() as path,patch.dict(sys.modules,modules),patch('importlib.metadata.version',return_value='0.23.1'):
            rows=[{'row_id':i} for i in range(4)]
            empty=FakeTrainer(path,[0]*8+[1]*8)
            result=run_filtered_training(empty,rows)
            self.assertEqual(result.global_step,0)
            self.assertEqual(empty.calls,0)
            self.assertEqual(empty.model.weight.item(),1.)
            self.assertEqual(empty.lr_scheduler.last_epoch,0)
            partial=FakeTrainer(path,[0]*12+[0,0,1,0])
            result=run_filtered_training(partial,rows)
            self.assertEqual(result.global_step,1)
            self.assertEqual(partial.calls,4)  # Removed 12 answers never enter the loss, including KL
            self.assertAlmostEqual(partial.model.weight.item(),.9,places=6)
            self.assertEqual(partial.state.log_history[0]['effective_answers'],4)


if __name__=='__main__':unittest.main()
