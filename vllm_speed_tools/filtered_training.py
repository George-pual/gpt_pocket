"""Explicit single-GPU, BF16, single-pass loop for TRL 0.23.1 GRPO.

Discard entire homogeneous groups BEFORE loss/backward, including their KL.
Empty selections do not call optimizer.step or scheduler.step. Fresh questions
are not resampled to fill the batch; partial effective batches are recorded.
GPU integration smoke test required before a paid formal run.
"""
import importlib.metadata
import json
import random
import time
from pathlib import Path

import torch
from correctness_curriculum import informative_indices, apply_selected_update


def subset_batch(batch, indices):
    size = batch['completion_ids'].shape[0]
    output = {}
    for name, value in batch.items():
        if isinstance(value, torch.Tensor) and value.ndim and value.shape[0] == size:
            output[name] = value[indices]
        else:
            output[name] = value
    output['num_items_in_batch'] = output['completion_mask'].sum()
    return output


def run_filtered_training(trainer, rows):
    from transformers.trainer_utils import TrainOutput
    if importlib.metadata.version('trl') != '0.23.1':
        raise RuntimeError('Private TRL hooks are pinned to 0.23.1')
    if (trainer.accelerator.num_processes != 1 or not trainer.args.bf16
            or trainer.args.loss_type != 'grpo' or trainer.args.num_iterations != 1
            or trainer.use_liger_loss):
        raise ValueError('Filtered runner requires single GPU BF16 GRPO, num_iterations=1, no Liger')
    args = trainer.args
    group = args.num_generations
    target = args.per_device_train_batch_size * args.gradient_accumulation_steps
    if target % group:
        raise ValueError('An optimizer update must contain whole prompt groups')
    trainer.state.max_steps = args.max_steps
    trainer.create_optimizer_and_scheduler(num_training_steps=args.max_steps)
    trainer.model, trainer.optimizer = trainer.accelerator.prepare(trainer.model, trainer.optimizer)
    trainer.model_wrapped = trainer.model
    trainer.model.train()
    # _compute_loss divides by this value. Accumulate weighted microbatch means
    # directly with BF16 backward, without a second Accelerate accumulation divisor.
    trainer.current_gradient_accumulation_steps = 1
    trainer.optimizer.zero_grad(set_to_none=True)
    ordered = list(rows)
    random.Random(args.data_seed).shuffle(ordered)
    questions_per_batch = args.generation_batch_size // group
    output = Path(args.output_dir)
    total = {'all_wrong':0,'all_correct':0,'mixed':0,'empty_generation_batches':0,
             'screened_training_questions':0,'used_training_questions':0}
    losses = []
    started = time.time()
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(str(output/'tensorboard'))

    def checkpoint():
        path = output/f'checkpoint-{trainer.state.global_step}'
        trainer.save_model(str(path))
        trainer.processing_class.save_pretrained(path)
        torch.save(trainer.optimizer.state_dict(),path/'optimizer.pt')
        torch.save(trainer.lr_scheduler.state_dict(),path/'scheduler.pt')
        torch.save({'python':random.getstate(),'cpu':torch.get_rng_state(),
                    'cuda':torch.cuda.get_rng_state_all()},path/'rng_state.pth')
        trainer.state.save_to_json(str(path/'trainer_state.json'))

    try:
        for offset in range(0,len(ordered),questions_per_batch):
            if trainer.state.global_step >= args.max_steps:
                break
            questions = ordered[offset:offset+questions_per_batch]
            inputs = [row for row in questions for _ in range(group)]
            trainer._latest_binary_correctness = None
            batch = trainer._generate_and_score_completions(inputs)
            correctness = trainer._latest_binary_correctness
            if correctness is None or len(correctness) != len(inputs):
                raise RuntimeError('Missing aligned correctness audit')
            indices, counts = informative_indices(correctness,group)
            for name,value in counts.items():
                total[name] += value
            total['screened_training_questions'] += len(questions)
            record = {'generation_batch':offset//questions_per_batch,
                'step_before':trainer.state.global_step,'row_ids':[r['row_id'] for r in questions],
                **counts,'kept_answers':len(indices)}
            if not indices:
                total['empty_generation_batches'] += 1

            def update(selected):
                # Only selected complete groups enter forward/backward. No KL,
                # Adam momentum, weight decay or scheduler changes for removed groups.
                for start in range(0,len(selected),target):
                    if trainer.state.global_step >= args.max_steps:
                        break
                    chosen = selected[start:start+target]
                    trainer.optimizer.zero_grad(set_to_none=True)
                    loss_value = 0.
                    for mb in range(0,len(chosen),args.per_device_train_batch_size):
                        ids = chosen[mb:mb+args.per_device_train_batch_size]
                        micro = subset_batch(batch,ids)
                        with trainer.accelerator.autocast():
                            loss = trainer.compute_loss(trainer.model,micro) * len(ids)/len(chosen)
                        if not torch.isfinite(loss).all():
                            raise RuntimeError('Non-finite filtered loss')
                        loss.backward()
                        loss_value += loss.detach().float().item()
                        del loss,micro
                    grads = [p.grad for p in trainer.model.parameters() if p.requires_grad and p.grad is not None]
                    if not grads or any(not torch.isfinite(g).all() for g in grads):
                        raise RuntimeError('Missing or non-finite gradients')
                    norm = trainer.accelerator.clip_grad_norm_(trainer.model.parameters(),args.max_grad_norm)
                    trainer.optimizer.step()
                    trainer.lr_scheduler.step()
                    trainer.optimizer.zero_grad(set_to_none=True)
                    trainer.state.global_step += 1
                    total['used_training_questions'] += len(chosen)//group
                    losses.append(loss_value)
                    metrics = {'loss':loss_value,'grad_norm':float(norm),
                        'learning_rate':trainer.lr_scheduler.get_last_lr()[0],
                        'effective_answers':len(chosen),**total}
                    # Parent metrics summarize generated groups, separately from loss selection.
                    metrics.update({'generated/'+name:sum(values)/len(values)
                        for name,values in trainer._metrics['train'].items() if values})
                    trainer._metrics['train'].clear()
                    trainer.state.log_history.append({'step':trainer.state.global_step,**metrics})
                    for name,value in metrics.items():
                        writer.add_scalar(name,value,trainer.state.global_step)
                    print('FILTERED_UPDATE',trainer.state.global_step,json.dumps(metrics),flush=True)
                    if trainer.state.global_step % args.save_steps == 0:
                        checkpoint()

            apply_selected_update(indices,update)
            record['step_after'] = trainer.state.global_step
            with (output/'group_filter.jsonl').open('a',encoding='utf-8') as f:
                f.write(json.dumps(record)+'\n')
            print('GROUP_FILTER',record,flush=True)
            del batch
        if trainer.state.global_step:
            checkpoint()
        metrics = {**total,'train_runtime':time.time()-started,
            'actual_optimizer_steps':trainer.state.global_step,
            'stop_reason':'update_budget' if trainer.state.global_step>=args.max_steps else 'unique_pool_exhausted',
            'train_loss':sum(losses)/max(1,len(losses)),
            'unprocessed_questions':len(rows)-total['screened_training_questions']}
        (output/'filtered_summary.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
        return TrainOutput(trainer.state.global_step,metrics['train_loss'],metrics)
    finally:
        writer.close()
