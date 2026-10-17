"""Paired prefix/continuation diagnostic, using frozen E1 SFT weights."""
import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import torch
from transformers import StoppingCriteria, StoppingCriteriaList, set_seed
from common import tokenizer, read_jsonl, save_json, extract_answer, numeric_equal, file_hash
from dense_rollout import build_dense_rollout
from rewards_article import REWARDS
from math_verification import verify_final_answer, verification_summary, require_math_verify


class Progress(StoppingCriteria):
    def __init__(self, prompt_width):
        self.prompt_width = prompt_width
        self.last = 0
        self.started = time.perf_counter()

    def __call__(self, input_ids, scores, **kwargs):
        length = input_ids.shape[1] - self.prompt_width
        if length >= self.last + 256:
            self.last = length
            print(json.dumps({'event': 'generation_progress', 'generated_tokens': length,
                              'seconds': round(time.perf_counter() - self.started, 1),
                              'allocated_gib': round(torch.cuda.memory_allocated()/2**30, 2)}), flush=True)
        return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--adapter', required=True)
    p.add_argument('--data', required=True)
    p.add_argument('--pilot-rollouts', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--questions', type=int, default=6)
    p.add_argument('--generation-batch-size', type=int, default=24)
    p.add_argument('--seed', type=int, default=3407)
    a = p.parse_args()
    require_math_verify()
    out = Path(a.output)
    if out.exists() and any(out.iterdir()):
        raise ValueError('Use a fresh output directory')
    out.mkdir(parents=True, exist_ok=True)
    selected = list(dict.fromkeys(r['row_id'] for r in read_jsonl(a.pilot_rollouts)))[:a.questions]
    by_id = {r['row_id']: r for r in read_jsonl(a.data)}
    rows = [by_id[i] for i in selected]
    if len(rows) != a.questions:
        raise ValueError('Insufficient questions')
    samples = [dict(r, sample_index=j) for r in rows for j in range(4)]
    t = tokenizer(a.model)
    t.padding_side = 'left'
    prompt_lengths = [len(t.apply_chat_template(r['prompt'], tokenize=True,
                                               add_generation_prompt=True)) for r in rows]
    if max(prompt_lengths) > 202:
        raise ValueError('Unexpected prompt length')
    initial_hash = file_hash(Path(a.adapter)/'adapter_model.safetensors')
    manifest = {'purpose': 'Paired length diagnosis; no weight updates or independent benchmark',
                'row_ids': selected, 'prompt_lengths': prompt_lengths,
                'questions': len(rows), 'answers_per_question': 4,
                'limits': [1846, 3894], 'max_prompt_length': 202,
                'seed': a.seed, 'temperature': 1.0, 'min_p': 0.1, 'top_p': 1.0,
                'top_k': 0, 'requested_batch_size': a.generation_batch_size,
                'adapter_sha256': initial_hash, 'data_sha256': file_hash(a.data),
                'method': 'One long sample; first 1846 tokens form its paired short-budget output',
                'arguments': vars(a)}
    save_json(out/'manifest.json', manifest)
    set_seed(a.seed)
    model, dtype = build_dense_rollout(a.model, a.adapter)
    model.generation_config.eos_token_id = t.eos_token_id
    model.generation_config.pad_token_id = t.pad_token_id
    assert not any(x.requires_grad for x in model.parameters())
    records = []
    started = time.perf_counter()

    def generate(batch, batch_seed):
        set_seed(batch_seed)
        prompts = [t.apply_chat_template(r['prompt'], tokenize=False,
                                        add_generation_prompt=True) for r in batch]
        inputs = t(prompts, padding=True, add_special_tokens=False,
                   return_tensors='pt').to('cuda')
        width = inputs.input_ids.shape[1]
        print(json.dumps({'event': 'batch_start', 'answers': len(batch),
                          'prompt_width': width, 'seed': batch_seed}), flush=True)
        with torch.inference_mode(), torch.autocast('cuda', dtype=dtype):
            generated = model.generate(**inputs, max_new_tokens=3894, do_sample=True,
                                       temperature=1.0, min_p=0.1, top_p=1.0, top_k=0,
                                       eos_token_id=t.eos_token_id, pad_token_id=t.pad_token_id,
                                       use_cache=True,
                                       stopping_criteria=StoppingCriteriaList([Progress(width)]))
        result = generated[:, width:].cpu().tolist()
        del generated, inputs
        torch.cuda.empty_cache()
        return result

    for offset in range(0, len(samples), a.generation_batch_size):
        batch = samples[offset:offset+a.generation_batch_size]
        try:
            outputs = generate(batch, a.seed+offset)
        except torch.OutOfMemoryError:
            outputs = None
        if outputs is None:
            # Retry the same question sample set in smaller inference batches.
            # Both length conditions still share each exact generated prefix.
            torch.cuda.empty_cache()
            print('OOM: retrying in batches of 4; paired comparison remains intact', flush=True)
            outputs = []
            for j in range(0, len(batch), 4):
                outputs.extend(generate(batch[j:j+4], a.seed+offset+j))
        for row, ids in zip(batch, outputs):
            if t.eos_token_id in ids:
                ids = ids[:ids.index(t.eos_token_id)+1]
            pair_key = f"{row['row_id']}:{row['sample_index']}"
            for cap in [1846, 3894]:
                cut = ids[:cap]
                eos = t.eos_token_id in cut
                text = t.decode(cut, skip_special_tokens=True)
                rewards = [float(fn(completions=[text], answer=[row['answer']])[0])
                           for fn in REWARDS]
                record = {'pair_key': pair_key, 'row_id': row['row_id'],
                          'sample_index': row['sample_index'], 'limit': cap,
                          'tokens': len(cut), 'eos_finished': eos,
                          'hit_length_cap': len(cut) >= cap,
                          'truncated': not eos and len(cut) >= cap,
                          'strict_correct': numeric_equal(extract_answer(text), row['answer']),
                          'math_verification': verify_final_answer(text, row['answer']),
                          'gold': row['answer'], 'rewards': rewards, 'completion': text}
                records.append(record)
                with (out/'paired_rollouts.jsonl').open('a', encoding='utf-8') as f:
                    f.write(json.dumps(record, ensure_ascii=False)+'\n')
        print(json.dumps({'event': 'batch_saved', 'paired_answers': len(records)//2}), flush=True)
    summary = {'status': 'complete', 'runtime_seconds': time.perf_counter()-started,
               'questions': len(rows), 'paired_answers': len(samples), 'budgets': {}}
    for cap in [1846, 3894]:
        rr = [r for r in records if r['limit'] == cap]
        n = len(rr)
        summary['budgets'][str(cap)] = {
            'math_verify': verification_summary(r['math_verification'] for r in rr),
            'answers': n, 'truncated_count': sum(r['truncated'] for r in rr),
            'eos_count': sum(r['eos_finished'] for r in rr),
            'correct_count': sum(r['strict_correct'] for r in rr),
            'mean_tokens': sum(r['tokens'] for r in rr)/n,
            'mean_reward': sum(sum(r['rewards']) for r in rr)/n,
            'questions_with_any_correct': sum(any(r['strict_correct'] for r in rr if r['row_id']==qid)
                                              for qid in selected)}
    paired = defaultdict(dict)
    for r in records:
        paired[r['pair_key']][r['limit']] = r
    summary['rescued_correct_after_1846'] = sum(not pair[1846]['strict_correct'] and
                                                pair[3894]['strict_correct'] for pair in paired.values())
    summary['lost_correct_after_1846'] = sum(pair[1846]['strict_correct'] and
                                             not pair[3894]['strict_correct'] for pair in paired.values())
    summary['initial_adapter_unchanged'] = initial_hash == file_hash(Path(a.adapter)/'adapter_model.safetensors')
    summary['max_allocated_gib'] = torch.cuda.max_memory_allocated()/2**30
    save_json(out/'summary.json', summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
