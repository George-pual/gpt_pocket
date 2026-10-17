"""Frozen SFT sampling and 20%-70% screening. No optimizer or training."""
import argparse
import gc
import json
import os
from pathlib import Path
from types import SimpleNamespace

from correctness_curriculum import difficulty_decision


def main():
    p = argparse.ArgumentParser()
    for name in ['model', 'adapter', 'data', 'output']:
        p.add_argument('--'+name, required=True)
    p.add_argument('--exclude', nargs='+', required=True,
                   help='JSONL files for SFT train/validation and every dev/test set')
    p.add_argument('--trials', type=int, default=16)
    p.add_argument('--low', type=float, default=.2)
    p.add_argument('--high', type=float, default=.7)
    p.add_argument('--seed', type=int, default=3407)
    p.add_argument('--completion-length', type=int, default=3894)
    p.add_argument('--sequence-length', type=int, default=4096)
    p.add_argument('--questions-per-batch', type=int, default=1)
    a = p.parse_args()
    difficulty_decision(0, a.trials, a.low, a.high)
    if a.questions_per_batch < 1:
        raise ValueError('Positive batch size required')
    import torch
    from common import tokenizer, load_model, read_jsonl, save_json, save_jsonl, file_hash
    from prepare_bc import canonical, question
    from math_verification import verify_final_answer, require_math_verify
    from safe_vllm_trainer import SafeVLLMTrainer
    from vllm import LLM, SamplingParams
    require_math_verify()
    out = Path(a.output)
    if out.exists():
        raise ValueError('Fresh output required; never overwrite a screening run')
    out.mkdir(parents=True)
    excluded = {canonical(question(r)) for path in a.exclude for r in read_jsonl(path)}
    seen, rows = set(), []
    for row in read_jsonl(a.data):
        key = canonical(question(row))
        if key in excluded or key in seen:
            continue
        seen.add(key)
        rows.append(row)
    if not rows or len({r['row_id'] for r in rows}) != len(rows):
        raise ValueError('Empty pool or duplicate row IDs')
    meta = json.loads(Path(a.data).with_suffix('.metadata.json').read_text())
    if a.completion_length <= 0 or meta['max_prompt_length']+a.completion_length > a.sequence_length:
        raise ValueError('Prompt plus completion exceeds sequence budget')
    meta.update(max_completion_length=a.completion_length,max_sequence_length=a.sequence_length)
    cap = meta['max_completion_length']
    t = tokenizer(a.model)
    adapter_hash = file_hash(Path(a.adapter)/'adapter_model.safetensors')
    torch.manual_seed(a.seed)
    policy, _ = load_model(a.model, a.adapter, training=True)
    policy.requires_grad_(False)
    policy.eval()
    policy.gradient_checkpointing_disable()
    llm = LLM(model=a.model, dtype='bfloat16', max_model_len=meta['max_sequence_length'],
              gpu_memory_utilization=.5, max_num_seqs=a.trials*a.questions_per_batch,
              seed=a.seed, enable_prefix_caching=False, disable_log_stats=True)
    proxy = SimpleNamespace(model=policy, accelerator=SimpleNamespace(unwrap_model=lambda x:x),
        llm=llm, args=SimpleNamespace(output_dir=str(out)), state=SimpleNamespace(global_step=0))
    SafeVLLMTrainer._move_model_to_vllm(proxy)
    del proxy, policy
    gc.collect()
    torch.cuda.empty_cache()
    params = SamplingParams(n=a.trials, temperature=1., min_p=.1, top_p=1., top_k=-1,
                            max_tokens=cap, stop_token_ids=[t.eos_token_id], seed=a.seed)
    kept, decisions = [], []
    save_json(out/'sampling_manifest.json', {
        'adapter_sha256':adapter_hash, 'source_sha256':file_hash(a.data),
        'exclusions_sha256':{path:file_hash(path) for path in a.exclude},
        'trials':a.trials, 'low':a.low, 'high':a.high, 'seed':a.seed,
        'sampling':{'temperature':1.,'min_p':.1,'top_p':1.,'top_k':-1,'max_tokens':cap},
        'representation':'Frozen decoded NF4 plus LoRA, BF16 vLLM copy',
        'scope':'Estimated final-answer correctness, not reasoning proof verification; canonical dedup only'})
    for offset in range(0, len(rows), a.questions_per_batch):
        batch = rows[offset:offset+a.questions_per_batch]
        prompts = [t.apply_chat_template(r['prompt'],tokenize=True,add_generation_prompt=True) for r in batch]
        if any(len(ids)>meta['max_prompt_length'] for ids in prompts):
            raise ValueError('Prompt exceeds P90 cap; re-prepare the pool without truncation')
        generated = llm.generate([{'prompt_token_ids':ids} for ids in prompts], params, use_tqdm=False)
        for row, result in zip(batch, generated):
            if len(result.outputs) != a.trials:
                raise RuntimeError('Incomplete screening group')
            records = []
            for trial, answer in enumerate(result.outputs):
                text = t.decode(answer.token_ids, skip_special_tokens=True)
                grade = verify_final_answer(text, row['answer'])
                records.append({'row_id':row['row_id'], 'trial':trial, 'completion':text,
                    'math_verification':grade, 'finish_reason':answer.finish_reason,
                    'tokens':len(answer.token_ids)})
            # Store raw evidence before selection. Generation truncations remain wrong
            # unless the unchanged final-answer grader finds a valid correct answer.
            with (out/'screen_rollouts.jsonl').open('a',encoding='utf-8') as f:
                for record in records:
                    f.write(json.dumps(record,ensure_ascii=False)+'\n')
            decision = difficulty_decision(sum(r['math_verification']['correct'] for r in records),
                                           a.trials,a.low,a.high)
            decisions.append({'row_id':row['row_id'], **decision})
            if decision['keep']:
                kept.append(row)
        save_json(out/'progress.json',{'screened':len(decisions),'total':len(rows),'retained':len(kept)})
        print('SCREEN_PROGRESS',len(decisions),len(rows),len(kept),flush=True)
    if file_hash(Path(a.adapter)/'adapter_model.safetensors') != adapter_hash:
        raise RuntimeError('SFT adapter changed during screening')
    save_jsonl(out/'decisions.jsonl',decisions)
    save_jsonl(out/'train.jsonl',kept)
    meta.update(retained_rows=len(kept), prepared_sha256=file_hash(out/'train.jsonl'),
                difficulty_screen={'adapter_sha256':adapter_hash,'trials':a.trials,'low':a.low,'high':a.high})
    save_json(out/'train.metadata.json',meta)
    save_json(out/'summary.json',{'screened':len(rows),'retained':len(kept),
        **{bucket:sum(d['bucket']==bucket for d in decisions) for bucket in ['easy','hard','eligible']}})
    llm.llm_engine.engine_core.shutdown()
    from vllm.distributed.parallel_state import destroy_model_parallel
    destroy_model_parallel()
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
    os.sync()
    print('SCREEN_SAVED',flush=True)
    import sys
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == '__main__':
    main()
