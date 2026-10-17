"""Stream dequantized NF4 + LoRA into vLLM without merging the training model."""
import json
import time
from pathlib import Path
import torch
import bitsandbytes as bnb
from train_closure_grpo import AuditedGRPOTrainer


class SafeVLLMTrainer(AuditedGRPOTrainer):
    def _calculate_rewards(self, inputs, prompts, completions, completion_ids_list):
        rewards = super()._calculate_rewards(inputs, prompts, completions, completion_ids_list)
        if getattr(self, 'filter_correctness_groups', False):
            # First component is the pinned binary Math-Verify reward, never
            # total reward, length, format, strict_numeric_accuracy or variance.
            binary = rewards[:, 0].detach().cpu().tolist()
            if any(value not in (0., 1.) for value in binary):
                raise RuntimeError('Group filter requires binary correctness')
            self._latest_binary_correctness = binary
        return rewards

    def _generate_and_score_completions(self, inputs):
        if getattr(self, 'single_pass', False) and self.model.training:
            from collections import Counter
            counts = Counter(row['row_id'] for row in inputs)
            if any(n != self.args.num_generations for n in counts.values()):
                raise RuntimeError('Incomplete or duplicate prompt groups in generation batch')
            seen = getattr(self, '_seen_prompt_ids', set())
            if seen.intersection(counts):
                raise RuntimeError('Refusing to generate a second training group for the same prompt')
            self._seen_prompt_ids = seen.union(counts)
        return super()._generate_and_score_completions(inputs)

    def _move_model_to_vllm(self):
        start = time.perf_counter()
        policy = self.accelerator.unwrap_model(self.model)
        core = policy.get_base_model()
        frozen = [(p, p.data_ptr(), p._version) for p in core.parameters() if not p.requires_grad]
        worker = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
        count = 0
        with torch.no_grad():
            for name, module in core.named_modules():
                if not hasattr(module, 'lora_A'):
                    continue
                base = module.get_base_layer()
                weight = bnb.functional.dequantize_4bit(base.weight.data, base.weight.quant_state)
                delta = module.get_delta_weight('default')
                merged = (weight.float() + delta.float()).to(torch.bfloat16)
                worker.load_weights([(name + '.weight', merged)])
                count += 1
                del weight, delta, merged
            for name, param in core.named_parameters():
                if '.lora_' in name or '.base_layer.' in name:
                    continue
                worker.load_weights([(name, param.detach().to(torch.bfloat16))])
                count += 1
        if any(p.data_ptr() != ptr or p._version != version for p, ptr, version in frozen):
            raise RuntimeError('Frozen training weights changed during vLLM synchronization')
        self.llm.reset_prefix_cache()
        torch.cuda.empty_cache()
        record = {'step': self.state.global_step, 'tensors': count,
                  'seconds': time.perf_counter() - start, 'frozen_weights_unchanged': True}
        with (Path(self.args.output_dir) / 'vllm_sync.jsonl').open('a') as f:
            f.write(json.dumps(record) + '\n')
        print('SAFE_VLLM_SYNC', record, flush=True)

    def training_step(self, *args, **kwargs):
        loss = super().training_step(*args, **kwargs)
        if not torch.isfinite(loss).all():
            raise RuntimeError('Non-finite training loss')
        grads = [p.grad for p in self.model.parameters() if p.requires_grad and p.grad is not None]
        if not grads or any(not torch.isfinite(g).all() for g in grads):
            raise RuntimeError('Missing or non-finite adapter gradients')
        print('FINITE_GRADIENTS', self.state.global_step,
              sum(float(g.detach().float().square().sum()) for g in grads), flush=True)
        return loss
