"""Pinned TRL 0.23.1 migration of the tutorial's 100-step GRPO run."""
import argparse
import json
import time
from pathlib import Path
import torch
from datasets import Dataset
from transformers import set_seed
from trl import GRPOConfig, GRPOTrainer
from common import (tokenizer, load_model, read_jsonl, save_json, runtime_manifest,
                    csv_callback, file_hash, extract_answer, numeric_equal)
from rewards_article import REWARDS, texts
from closure_rewards import make_end_reward
from math_verification import (verify_final_answer, require_math_verify,
                               log_verification_metrics, math_verify_reward)

class AuditedGRPOTrainer(GRPOTrainer):
    def _generate_and_score_completions(self, inputs):
        started = time.perf_counter()
        policy = self.accelerator.unwrap_model(self.model)
        original_generate = policy.generate
        rollout = getattr(self, 'rollout_model', None)
        if rollout is not None:
            from dense_rollout import sync_adapter
            transfer_started = time.perf_counter()
            sync_adapter(policy, rollout)
            rollout.to(self.accelerator.device)
            torch.cuda.synchronize()
            transfer_seconds = time.perf_counter() - transfer_started
            def decoded_generate(*args, **kwargs):
                decode_started = time.perf_counter()
                try:
                    output = rollout.generate(*args, **kwargs)
                    torch.cuda.synchronize()
                    decode_seconds = time.perf_counter() - decode_started
                    self._metrics['train']['dense_decode_seconds'].append(decode_seconds)
                    print(f'DENSE_DECODE batch={len(inputs)} seconds={decode_seconds:.2f}', flush=True)
                    return output
                finally:
                    # Release dense weights before NF4 reference scoring/backprop.
                    rollout.cpu()
                    torch.cuda.empty_cache()
            policy.generate = decoded_generate
            self._metrics['train']['dense_transfer_seconds'].append(transfer_seconds)
        # PEFT.generate bypasses Accelerate's wrapped forward; Qwen3's FP32
        # normalization and BF16 LM head therefore need explicit autocast.
        try:
            with self.accelerator.autocast():
                result = super()._generate_and_score_completions(inputs)
        finally:
            policy.generate = original_generate
            if rollout is not None:
                rollout.cpu()
                torch.cuda.empty_cache()
        # TRL 0.23.1 unwrap_model_for_generation re-enables checkpointing with
        # default reentrant=True. Restore the configured non-reentrant mode.
        self.model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={'use_reentrant': False})
        elapsed = time.perf_counter() - started
        mode = 'train' if self.model.training else 'eval'
        self._metrics[mode]['generation_scoring_seconds'].append(elapsed)
        print(f'GENERATION_AND_SCORING batch={len(inputs)} seconds={elapsed:.2f}', flush=True)
        return result

    def _calculate_rewards(self, inputs, prompts, completions, completion_ids_list):
        rewards = super()._calculate_rewards(inputs, prompts, completions, completion_ids_list)
        mode = 'train' if self.model.training else 'eval'
        correct = [numeric_equal(extract_answer(c), row['answer'])
                   for c, row in zip(texts(completions), inputs)]
        self._metrics[mode]['strict_numeric_accuracy'].append(sum(correct) / len(correct))
        verified = [verify_final_answer(c, row['answer'])
                    for c, row in zip(texts(completions), inputs)]
        log_verification_metrics(self._metrics[mode], verified)
        eos_id = self.processing_class.eos_token_id
        ended = [eos_id in ids for ids in completion_ids_list]
        hit_cap = [len(ids) >= self.args.max_completion_length for ids in completion_ids_list]
        truncated = [cap and not eos for cap, eos in zip(hit_cap, ended)]
        self._metrics[mode]['pilot/eos_finish_rate'].append(sum(ended) / len(ended))
        self._metrics[mode]['pilot/length_cap_rate'].append(sum(hit_cap) / len(hit_cap))
        self._metrics[mode]['pilot/truncation_rate'].append(sum(truncated) / len(truncated))
        self._metrics[mode]['ab/end_bonus'].append(rewards[:, -2].mean().item())
        self._metrics[mode]['ab/task_reward'].append(rewards[:, :-1].sum(dim=1).mean().item())
        self._metrics[mode]['ab/length_penalty'].append(rewards[:, -1].mean().item())
        if self.accelerator.is_main_process:
            with (Path(self.args.output_dir) / 'rollouts.jsonl').open('a', encoding='utf-8') as f:
                for i, (row, c, ids) in enumerate(zip(inputs, texts(completions), completion_ids_list)):
                    record = {'step': self.state.global_step, 'row_id': row['row_id'],
                              'gold': row['answer'], 'completion': c, 'tokens': len(ids),
                              'strict_correct': correct[i], 'math_verification': verified[i],
                              'rewards': rewards[i].tolist(),
                              'eos_finished': ended[i], 'hit_length_cap': hit_cap[i],
                              'truncated': truncated[i]}
                    f.write(json.dumps(record, ensure_ascii=False) + '\n')
        return rewards

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--data', required=True)
    p.add_argument('--init-adapter', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--max-steps', type=int, default=100)
    p.add_argument('--reward-mode', choices=['article', 'math-verify'], default='article',
                   help='article preserves tutorial rewards; math-verify uses binary final-answer correctness')
    p.add_argument('--end-bonus-weight', type=float, default=0.0)
    p.add_argument('--reward-v2', action='store_true', help='Independent format/EOS bonus plus 12-word repetition penalty')
    p.add_argument('--filter-correctness-groups', action='store_true',
                   help='Experimental single-GPU loop: discard whole all-right/all-wrong groups before backward')
    p.add_argument('--resume')
    p.add_argument('--vllm-model', required=True)
    p.add_argument('--vllm-memory', type=float, default=.50)
    p.add_argument('--beta', type=float, default=0.04)
    p.add_argument('--seed', type=int, default=3407)
    p.add_argument('--microbatch', type=int, default=4)
    p.add_argument('--gradient-accumulation', type=int, default=1)
    p.add_argument('--generation-batch-size', type=int, default=8)
    p.add_argument('--preflight-rollouts', help='Prior rollout JSONL for full-length backward memory validation')
    p.add_argument('--preflight-data', help='Full prepared data containing preflight row IDs')
    p.add_argument('--preflight-only', action='store_true')
    p.add_argument('--length-penalty-weight', type=float, default=0.0)
    p.add_argument('--soft-buffer', type=int, default=512)
    p.add_argument('--dense-rollout', action='store_true',
                   help='Cache decoded NF4 inference weights; policy/reference remain 4-bit')
    p.add_argument('--smoke-completion-length', type=int,
                   help='Diagnostic only: shorter generation; omit for formal reproduction')
    a = p.parse_args()
    # One pass only: never let max_steps silently cycle a small prompt pool.
    if a.resume:
        raise ValueError('Single-pass entry point does not support resume yet; use the archived runner for historical runs.')
    rows = read_jsonl(a.data)
    prompt_keys = [json.dumps(row['prompt'], sort_keys=True, ensure_ascii=False) for row in rows]
    if len(set(prompt_keys)) != len(prompt_keys):
        raise ValueError('Duplicate prompts in training data; deduplicate before training.')
    row_ids = [row['row_id'] for row in rows]
    if len(set(row_ids)) != len(row_ids):
        raise ValueError('Duplicate row_id in training data.')
    answers_per_update = a.microbatch * a.gradient_accumulation
    if (answers_per_update <= 0 or a.generation_batch_size <= 0
            or a.generation_batch_size % 4
            or a.generation_batch_size % answers_per_update):
        raise ValueError('Generation batch must divide into complete updates and four-answer prompt groups.')
    single_pass_steps = (len(rows) // (a.generation_batch_size // 4)
                         * (a.generation_batch_size // answers_per_update))
    if not 0 < a.max_steps <= single_pass_steps:
        raise ValueError(f'Requested {a.max_steps} steps would exceed one pass. '
                         f'{len(rows)} unique prompts allow at most {single_pass_steps} steps '
                         'with complete generation batches; increase the dataset or reduce --max-steps.')
    require_math_verify()
    if a.reward_mode != 'math-verify': raise ValueError('This experiment requires math-verify rewards')
    if not 0 <= a.end_bonus_weight <= .1: raise ValueError('Invalid end bonus')
    selected_rewards = REWARDS if a.reward_mode == 'article' else [math_verify_reward]
    output = Path(a.output)
    if output.exists() and any(output.iterdir()) and not a.resume:
        raise ValueError('Use a fresh output directory or --resume')
    output.mkdir(parents=True, exist_ok=True)
    set_seed(a.seed)
    meta = json.loads(Path(a.data).with_suffix('.metadata.json').read_text())
    if a.filter_correctness_groups:
        import importlib.metadata
        if importlib.metadata.version('trl') != '0.23.1':
            raise ValueError('Filtered runner is pinned to TRL 0.23.1')
        if a.reward_v2 or a.end_bonus_weight or a.length_penalty_weight:
            raise ValueError('First filtered control uses pure correctness; disable behavior rewards')
        screen = meta.get('difficulty_screen')
        if not screen or screen['adapter_sha256'] != file_hash(Path(a.init_adapter)/'adapter_model.safetensors'):
            raise ValueError('Screen data with the exact initial SFT adapter before filtered training')
        if (a.microbatch * a.gradient_accumulation) % 4:
            raise ValueError('Filtered updates require whole four-answer groups')
    completion_length = a.smoke_completion_length or meta['max_completion_length']
    if not 0 <= a.length_penalty_weight or not 0 < a.soft_buffer <= completion_length:
        raise ValueError('Invalid soft penalty settings')
    def soft_overlong_penalty(completions, completion_ids, **kwargs):
        threshold = completion_length - a.soft_buffer
        return [-a.length_penalty_weight * min(1.0, max(0.0, (len(ids)-threshold)/a.soft_buffer))
                for ids in completion_ids]
    t = tokenizer(a.model)
    t.padding_side = 'left'
    end_reward = make_end_reward(t.eos_token_id, a.end_bonus_weight)
    extra_rewards = []
    if a.reward_v2:
        from reward_v2 import make_complete_end_reward, repetition_penalty
        if a.end_bonus_weight != .05:
            raise ValueError('Reward v2 requires --end-bonus-weight .05')
        end_reward = make_complete_end_reward(t.eos_token_id)
        extra_rewards = [repetition_penalty]
    model, dtype = load_model(a.model, a.init_adapter, training=True)
    # Same FP32 casting/checkpoint preparation as policy, then freeze every parameter.
    ref_model, _ = load_model(a.model, a.init_adapter, training=True)
    ref_model.requires_grad_(False)
    ref_model.eval()
    ref_model.gradient_checkpointing_disable()
    for m in [model, ref_model]:
        m.generation_config.eos_token_id = t.eos_token_id
        m.generation_config.pad_token_id = t.pad_token_id
    config = GRPOConfig(
        output_dir=str(output), max_steps=a.max_steps, num_train_epochs=1,
        per_device_train_batch_size=a.microbatch,
        gradient_accumulation_steps=a.gradient_accumulation,
        generation_batch_size=a.generation_batch_size, num_generations=4, num_iterations=1,
        max_prompt_length=meta['max_prompt_length'],
        max_completion_length=completion_length,
        learning_rate=5e-6, weight_decay=0.01, warmup_ratio=0.1,
        lr_scheduler_type='linear', optim='adamw_8bit',
        temperature=1.0, min_p=0.1, top_p=1.0, top_k=-1,
        generation_kwargs=None,
        beta=a.beta, loss_type='grpo', scale_rewards='group',
        use_vllm=True, vllm_mode='colocate', vllm_gpu_memory_utilization=a.vllm_memory,
        vllm_enable_sleep_mode=True, vllm_importance_sampling_correction=True,
        mask_truncated_completions=False,
        bf16=dtype == torch.bfloat16, fp16=dtype == torch.float16,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False},
        logging_steps=1, save_steps=min(25,a.max_steps), save_total_limit=5,
        seed=a.seed, data_seed=a.seed, remove_unused_columns=False,
        report_to=['tensorboard'], logging_dir=str(output / 'tensorboard'))
    # vLLM loads a plain BF16 export; policy stays NF4 with trainable LoRA.
    model.config._name_or_path = a.vllm_model
    from safe_vllm_trainer import SafeVLLMTrainer
    trainer = SafeVLLMTrainer(
        model=model, args=config, processing_class=t, reward_funcs=selected_rewards + extra_rewards + [end_reward, soft_overlong_penalty],
        train_dataset=Dataset.from_list(rows), callbacks=[csv_callback(output)])
    trainer.single_pass = True
    trainer.filter_correctness_groups = a.filter_correctness_groups
    # GRPOTrainer casts the policy's PEFT modules to BF16. Apply the identical
    # pinned helper to the reference, or identical weights yield different logits.
    from trl.trainer.grpo_trainer import prepare_peft_model
    ref_model = prepare_peft_model(ref_model, None, config)
    ref_model.requires_grad_(False)
    ref_model.gradient_checkpointing_disable()
    ref_model.eval()
    # PEFT's default disable_adapter() would reference Base, losing the SFT policy.
    trainer.ref_model = trainer.accelerator.prepare_model(ref_model, evaluation_mode=True)
    model.eval()
    probe = t('Check reference integrity: 1 + 1 =', return_tensors='pt').to('cuda')
    with torch.no_grad(), torch.autocast('cuda', dtype=dtype):
        policy_logits = model(**probe).logits.float()
        ref_logits = ref_model(**probe).logits.float()
    max_difference = (policy_logits - ref_logits).abs().max().item()
    if not torch.allclose(policy_logits, ref_logits, atol=1e-5, rtol=1e-5):
        raise RuntimeError(f'Initial policy/reference mismatch: {max_difference}')
    if any(param.requires_grad for param in ref_model.parameters()):
        raise RuntimeError('Reference model must be frozen')
    del policy_logits, ref_logits, probe
    dense_validation = None
    if a.dense_rollout:
        from dense_rollout import build_dense_rollout
        rollout, _ = build_dense_rollout(a.model, a.init_adapter)
        probe = t('Check reference integrity: 1 + 1 =', return_tensors='pt').to('cuda')
        with torch.no_grad(), torch.autocast('cuda', dtype=dtype):
            original = model(**probe, logits_to_keep=1).logits.float()
            decoded = rollout(**probe, logits_to_keep=1).logits.float()
        dense_difference = (original-decoded).abs().max().item()
        if not torch.allclose(original, decoded, atol=1e-5, rtol=1e-5):
            raise RuntimeError(f'Decoded NF4 logits mismatch: {dense_difference}')
        dense_validation = {'initial_logits_max_difference': dense_difference,
                            'source': 'decoded NF4 weights, not original BF16 Base',
                            'offloaded_to_cpu_between_generation_batches': True,
                            'training_policy_and_reference_remain_nf4': True}
        del original, decoded, probe
        rollout.cpu()
        torch.cuda.empty_cache()
        trainer.rollout_model = rollout
    model.train()
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={'use_reentrant': False})
    runtime_manifest(a, output, a.data)
    save_json(output / 'grpo_config.json', config.to_dict())
    if a.preflight_rollouts:
        rows = read_jsonl(a.preflight_data or a.data)
        by_id = {row['row_id']: row for row in rows}
        audit = read_jsonl(a.preflight_rollouts)[:a.microbatch]
        if len(audit) != a.microbatch:
            raise ValueError('Not enough saved rollouts for microbatch preflight')
        samples = [t.apply_chat_template(by_id[r['row_id']]['prompt'], tokenize=False,
                                        add_generation_prompt=True) + r['completion'] for r in audit]
        probe = t(samples, return_tensors='pt', padding=True, truncation=True,
                  max_length=meta['max_prompt_length'] + completion_length,
                  add_special_tokens=False).to('cuda')
        keep = min(completion_length, probe.input_ids.shape[1] - 1)
        # Accelerate converts wrapped forward outputs to FP32 during training.
        class FloatOutput:
            def __call__(self, **kwargs):
                out = model(**kwargs)
                out.logits = out.logits.float()
                return out
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        preflight_started = time.perf_counter()
        print(f'FULL-LENGTH BACKWARD PREFLIGHT batch={a.microbatch} tokens={probe.input_ids.shape[1]}', flush=True)
        with torch.autocast('cuda', dtype=dtype):
            logps, entropies = trainer._get_per_token_logps_and_entropies(
                FloatOutput(), probe.input_ids, probe.attention_mask, keep,
                batch_size=a.microbatch, compute_entropy=True)
            test_loss = -logps.mean()
        test_loss.backward()
        if not any(p.grad is not None and torch.isfinite(p.grad).all()
                   for n,p in model.named_parameters() if 'lora_' in n):
            raise RuntimeError('No finite LoRA gradient in preflight')
        torch.cuda.synchronize()
        preflight = {'microbatch': a.microbatch, 'sequence_length': probe.input_ids.shape[1],
                     'seconds': time.perf_counter() - preflight_started,
                     'max_allocated_gib': torch.cuda.max_memory_allocated() / 2**30,
                     'max_reserved_gib': torch.cuda.max_memory_reserved() / 2**30,
                     'optimizer_updated': False}
        save_json(output / 'backward_preflight.json', preflight)
        print('BACKWARD PREFLIGHT PASSED', preflight, flush=True)
        model.zero_grad(set_to_none=True)
        del probe, logps, entropies, test_loss
        torch.cuda.empty_cache()
    if a.preflight_only:
        if not a.preflight_rollouts:
            raise ValueError('--preflight-only requires --preflight-rollouts')
        save_json(output / 'preflight_complete.json', {'status': 'complete', 'optimizer_updated': False})
        return
    save_json(output / 'migration.json', {
        'source': 'https://developer.volcengine.com/articles/7526781272350261302',
        'beta_note': 'Article does not specify beta; explicit 0.04 migration assumption.',
        'generation': 'vLLM colocate BF16 copy with safe NF4+LoRA synchronization; importance correction enabled.',
        'batch': {'microbatch': a.microbatch, 'gradient_accumulation': a.gradient_accumulation,
                  'generation_batch_size': a.generation_batch_size,
                  'effective_answers_per_update': a.microbatch * a.gradient_accumulation,
                  'generations_per_prompt': 4},
        'reference': str(a.init_adapter), 'reference_is_frozen': True,
        'reward_v2': a.reward_v2,
        'filter_correctness_groups': a.filter_correctness_groups,
        'reward_components': (['correctness_1', 'repetition_minus_0.1', 'complete_end_0.05', 'soft_length_minus_0.5']
                              if a.reward_v2 else ['correctness', 'correct_end', 'soft_length']),
        'single_pass': {'unique_questions': len(rows), 'maximum_steps': single_pass_steps,
                        'resume_supported': False},
        'initial_logits_max_difference': max_difference,
        'adapter_sha256': file_hash(Path(a.init_adapter) / 'adapter_model.safetensors'),
        'dense_rollout': dense_validation,
        'data': meta,
        'soft_length_penalty': {'weight': a.length_penalty_weight, 'buffer': a.soft_buffer,
                                'threshold': completion_length - a.soft_buffer,
                                'mask_truncated_completions': False}})
    print('REFERENCE VERIFIED; STARTING GRPO', flush=True)
    initial_adapters = {n: p.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad}
    started = time.time()
    if a.filter_correctness_groups:
        from filtered_training import run_filtered_training
        result = run_filtered_training(trainer, read_jsonl(a.data))
    else:
        result = trainer.train(resume_from_checkpoint=a.resume)
    deltas = {n: float((p.detach().cpu().float()-initial_adapters[n].float()).abs().max()) for n,p in model.named_parameters() if n in initial_adapters}
    if not any(x > 0 for x in deltas.values()):
        raise RuntimeError('No adapter parameter update detected')
    save_json(output / 'parameter_update_check.json', {'updated': True, 'maximum_delta': max(deltas.values()), 'updated_tensors': sum(x>0 for x in deltas.values())})
    trainer.save_model(str(output / 'adapter'))
    t.save_pretrained(output / 'adapter')
    save_json(output / 'trainer_log.json', trainer.state.log_history)
    save_json(output / 'completion_summary.json', {
        'status': 'complete', 'steps': trainer.state.global_step,
        'runtime_seconds': time.time() - started, 'metrics': result.metrics,
        'max_allocated_gib': torch.cuda.max_memory_allocated() / 2**30,
        'max_reserved_gib': torch.cuda.max_memory_reserved() / 2**30})

    trainer.llm.llm_engine.engine_core.shutdown()
    from vllm.distributed.parallel_state import destroy_model_parallel
    destroy_model_parallel()
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
    # Output saving and explicit engine cleanup succeeded above.
    # Avoid the observed vLLM/C++ interpreter-finalizer abort.
    import os, sys
    os.sync()
    print('SAVED_AND_ENGINE_CLOSED; BYPASSING_FINALIZER_ABORT', flush=True)
    sys.stderr.flush()
    os._exit(0)

if __name__ == '__main__':
    main()
