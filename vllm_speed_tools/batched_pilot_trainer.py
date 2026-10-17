"""Batch identical probability computations; preserve masks, rewards and update size."""
import torch
import torch.nn.functional as F
from trimmed_pilot_trainer import TrimmedPilotTrainer,kept_width

class BatchedPilotTrainer(TrimmedPilotTrainer):
    def _get_per_token_logps_and_entropies(self,model,input_ids,attention_mask,logits_to_keep,batch_size=None,compute_entropy=False,**kwargs):
        if any(v is not None for v in kwargs.values()):
            raise ValueError('Text only')
        # Use genuine batched forward for a training microbatch. No-grad
        # policy/reference scoring has a separately bounded batch size.
        chunk=input_ids.shape[0] if torch.is_grad_enabled() else self.score_batch
        prompt_width=input_ids.shape[1]-logits_to_keep
        logps=[];entropies=[];actual=0
        # Skip the one-answer loop in TrimmedPilotTrainer, retaining its parent.
        parent=super(TrimmedPilotTrainer,self)
        for i in range(0,input_ids.shape[0],chunk):
            ids=input_ids[i:i+chunk];mask=attention_mask[i:i+chunk]
            width=kept_width(mask[:,prompt_width:]);actual+=width*ids.shape[0]
            lp,en=parent._get_per_token_logps_and_entropies(model,ids[:,:prompt_width+width],mask[:,:prompt_width+width],width,batch_size=ids.shape[0],compute_entropy=compute_entropy)
            logps.append(F.pad(lp,(0,logits_to_keep-width)))
            if compute_entropy:entropies.append(F.pad(en,(0,logits_to_keep-width)))
        if hasattr(self,'padding_counts'):
            self.padding_counts['scoring_valid_positions']+=actual
            self.padding_counts['scoring_original_positions']+=input_ids.shape[0]*logits_to_keep
        return torch.cat(logps),torch.cat(entropies) if compute_entropy else None
