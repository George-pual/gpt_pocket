"""Avoid masked-right-padding forward/backward work; keep every valid token."""
import time,json
from pathlib import Path
import torch
import torch.nn.functional as F
from train_reward_pilot_v3 import PilotTrainer

def kept_width(mask):
    """Right-padding only: keep through the last active column, minimum one."""
    cols=torch.arange(mask.shape[1],device=mask.device).unsqueeze(0)+1
    return max(1,int((cols*mask.bool()).max().item()))

def trim_loss_inputs(inputs):
    width=kept_width(inputs['completion_mask'])
    names={'completion_ids','completion_mask','old_per_token_logps','ref_per_token_logps','importance_sampling_ratio'}
    result={}
    for name,value in inputs.items():
        result[name]=value[:,:width] if name in names and isinstance(value,torch.Tensor) and value.ndim==2 else value
    return result

class TrimmedPilotTrainer(PilotTrainer):
    def _get_per_token_logps_and_entropies(self,model,input_ids,attention_mask,logits_to_keep,batch_size=None,compute_entropy=False,**kwargs):
        if any(v is not None for v in kwargs.values()):
            raise ValueError('Text-only padding optimization; unexpected multimodal inputs')
        prompt_width=input_ids.shape[1]-logits_to_keep
        logps=[];entropies=[];actual=0
        # Existing experiment uses a training microbatch of one. Forward scoring
        # is streamed one answer at a time, as in the original memory-safe run.
        for i in range(input_ids.shape[0]):
            width=kept_width(attention_mask[i:i+1,prompt_width:])
            actual+=width
            lp,en=super()._get_per_token_logps_and_entropies(model,
                input_ids[i:i+1,:prompt_width+width],attention_mask[i:i+1,:prompt_width+width],
                width,batch_size=1,compute_entropy=compute_entropy)
            logps.append(F.pad(lp,(0,logits_to_keep-width)))
            if compute_entropy:entropies.append(F.pad(en,(0,logits_to_keep-width)))
        if hasattr(self,'padding_counts'):
            self.padding_counts['scoring_valid_positions']+=actual
            self.padding_counts['scoring_original_positions']+=input_ids.shape[0]*logits_to_keep
        return torch.cat(logps),torch.cat(entropies) if compute_entropy else None

    def _compute_loss(self,model,inputs):
        trimmed=trim_loss_inputs(inputs)
        return super()._compute_loss(model,trimmed)

    def _calculate_rewards(self,*args,**kwargs):
        started=time.perf_counter()
        result=super()._calculate_rewards(*args,**kwargs)
        mode='train' if self.model.training else 'eval'
        self._metrics[mode]['timing/rewards_seconds'].append(time.perf_counter()-started)
        return result

    def training_step(self,*args,**kwargs):
        started=time.perf_counter()
        result=super().training_step(*args,**kwargs)
        self._metrics['train']['timing/forward_backward_microbatch_seconds'].append(time.perf_counter()-started)
        return result

    def _generate_and_score_completions(self,inputs):
        self.padding_counts={'scoring_valid_positions':0,'scoring_original_positions':0}
        result=super()._generate_and_score_completions(inputs)
        with (Path(self.args.output_dir)/'padding_audit.jsonl').open('a') as f:
            f.write(json.dumps(dict(step=self.state.global_step,**self.padding_counts))+'\n')
        return result
