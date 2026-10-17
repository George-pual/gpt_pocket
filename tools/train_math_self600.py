"""Fresh Base QLoRA cold start from reviewed self trajectories; no GRPO launch."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def main():
    p = argparse.ArgumentParser()
    for k in ('model', 'data', 'validation', 'output'):
        p.add_argument('--' + k, required=True)
    p.add_argument('--learning-rate', type=float, default=5e-6)
    p.add_argument('--steps', type=int, default=64)
    p.add_argument('--prepare-only', action='store_true')
    a = p.parse_args()
    from common import tokenizer, load_model
    from math_verification import verify_final_answer
    from prepare_math_self600 import canonical
    assert 0 < a.learning_rate <= 1e-5 and 0 < a.steps <= 64
    out = Path(a.output)
    assert not out.exists(), 'Fresh training output required'
    with Path(a.data).open(encoding='utf-8') as f:
        source = [json.loads(s) for s in f if s.strip()]
    with Path(a.validation).open(encoding='utf-8') as f:
        validation = [json.loads(s) for s in f if s.strip()]
    assert source and len(source) <= 600
    spotcheck_path=Path(a.data).parent/'direct_agent_spotcheck.json'
    spotcheck=json.loads(spotcheck_path.read_text())
    assert spotcheck['status']=='pass' and spotcheck['train_sha256']==sha(a.data), 'Direct source-grounded spotcheck must match this frozen training file'
    assert len({canonical(r['problem']) for r in source}) == len(source)
    assert not {canonical(r['problem']) for r in source} & {canonical(r['problem']) for r in validation}
    assert json.loads((Path(a.model)/'download_manifest.json').read_text())['repo'] == 'Qwen/Qwen3-4B-Base'
    t = tokenizer(a.model)
    t.padding_side = 'right'
    encoded = []
    for row in source:
        assert row['review_approved'] is True and row['reasoning_review_status'] == 'source_grounded_direct_agent_review_pass'
        assert row['completion'].endswith('</SOLUTION><|endoftext|>')
        assert row['completion'].count('<|endoftext|>') == 1
        assert verify_final_answer(row['completion'], row['answer'])['correct']
        prefix = t.apply_chat_template(row['prompt'], tokenize=True, add_generation_prompt=True)
        completion = t(row['completion'], add_special_tokens=False)['input_ids']
        assert completion[-1] == t.eos_token_id and len(prefix)+len(completion) <= 2048
        encoded.append({'input_ids': prefix+completion, 'completion_mask': [0]*len(prefix)+[1]*len(completion)})
    out.mkdir(parents=True)
    write(out/'data_audit.json', dict(train_questions=len(source), validation_questions=len(validation),
         train_sha256=sha(a.data), validation_sha256=sha(a.validation),
         no_exact_train_validation_overlap=True, no_truncation=True,
         max_sequence_tokens=max(len(r['input_ids']) for r in encoded),
         mathematical_review='Every selected solution directly reviewed by the primary agent against the problem and reference; not a formal proof certificate.',
         initialization='Fresh LoRA on original Qwen3-4B-Base; no old adapters', arguments=vars(a)))
    if a.prepare_only:
        print('PREPARE_ONLY_PASS', len(source), flush=True)
        return
    assert importlib.metadata.version('trl') == '0.23.1'
    import torch
    from datasets import Dataset
    from transformers import TrainerCallback, set_seed
    from trl import SFTConfig, SFTTrainer
    assert torch.cuda.is_available()
    free, total = torch.cuda.mem_get_info()
    assert free / total > .8, 'GPU must be idle before training'
    set_seed(3407)
    model, dtype = load_model(a.model, None, training=True)

    class AdapterSnapshots(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step in (8, 16, 32, 64):
                dest = out/'snapshots'/f'step-{state.global_step}'
                kwargs['model'].save_pretrained(dest)
                t.save_pretrained(dest)
                print('ADAPTER_SNAPSHOT', state.global_step, flush=True)

    config = SFTConfig(output_dir=str(out), max_length=2048,
        per_device_train_batch_size=4, gradient_accumulation_steps=4,
        learning_rate=a.learning_rate, max_steps=a.steps, warmup_steps=4,
        weight_decay=.01, lr_scheduler_type='linear', optim='adamw_8bit',
        bf16=dtype==torch.bfloat16, fp16=dtype==torch.float16,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={'use_reentrant': False},
        packing=False, completion_only_loss=True, logging_steps=1,
        save_strategy='no', eval_strategy='no', report_to=['tensorboard'],
        seed=3407, data_seed=3407)
    trainer = SFTTrainer(model=model, args=config, processing_class=t,
                         train_dataset=Dataset.from_list(encoded), callbacks=[AdapterSnapshots()])
    for expected, actual in zip(encoded, trainer.train_dataset):
        assert expected['input_ids'] == actual['input_ids']
        labels = trainer.data_collator([actual])['labels'][0].tolist()
        assert labels == [i if m else -100 for i, m in zip(expected['input_ids'], expected['completion_mask'])]
    batch = trainer.data_collator([trainer.train_dataset[0], trainer.train_dataset[-1]])
    for i, index in enumerate((0, len(encoded)-1)):
        n = len(encoded[index]['input_ids'])
        assert batch['labels'][i,n-1].item() == t.eos_token_id
        assert (batch['labels'][i,n:] == -100).all()
    write(out/'supervision_audit.json', dict(checked=len(encoded), prompt_masked=True,
          padding_masked=True, completion_and_real_eos_supervised=True))
    write(out/'sft_config.json', config.to_dict())
    before = {n:p.detach().float().cpu().clone() for n,p in model.named_parameters() if p.requires_grad}
    start = time.time()
    result = trainer.train()
    trainer.save_model(str(out/'adapter'))
    t.save_pretrained(out/'adapter')
    changed = sum(not torch.equal(before[n], p.detach().float().cpu()) for n,p in model.named_parameters() if n in before)
    assert changed > 0
    write(out/'trainer_log.json', trainer.state.log_history)
    write(out/'completion_summary.json', dict(status='complete', steps=trainer.state.global_step,
         train_questions=len(source), learning_rate=a.learning_rate, effective_batch=16,
         approximate_question_presentations=trainer.state.global_step*16,
         train_metrics=result.metrics, runtime_seconds=time.time()-start,
         changed_trainable_tensors=changed, adapter_sha256=sha(out/'adapter/adapter_model.safetensors'),
         development_generation_evaluation_status='pending', grpo_started=False))
    print('SFT_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
