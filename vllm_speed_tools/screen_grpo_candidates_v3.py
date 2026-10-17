"""Frozen SFT v3 screening: four independent vLLM samples per question."""
import argparse
import concurrent.futures
import hashlib
import json
import multiprocessing
import os
import time
from collections import Counter
from pathlib import Path

EXPECTED_ADAPTER = '5461e88ae8dba6a3fb536e2bd93bdd8e6f3098a3234943eaef28bc0196017b9b'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    if not path.exists():
        return []
    with path.open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def save(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def append(stream, row):
    stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    stream.flush()
    os.fsync(stream.fileno())


def grade(record):
    from math_verification import verify_final_answer
    from closure_rewards import complete_format
    gold = record['answer']
    gold_check = verify_final_answer('<SOLUTION>' + gold + '</SOLUTION>', gold)
    samples = []
    for sample in record['samples']:
        verdict = verify_final_answer(sample['completion'], gold)
        fmt = complete_format(sample['completion'])
        samples.append({**sample, 'math_verification': verdict,
                        'format_valid': fmt, 'normal_end': fmt and sample['eos_finished'],
                        'correct': verdict['correct']})
    k = sum(s['correct'] for s in samples)
    label = ('ungradable_gold' if not gold_check['correct'] else
             'mixed' if 0 < k < 4 else 'all_correct' if k == 4 else 'all_wrong')
    return {'row_id': record['row_id'], 'answer': gold, 'source': record['source'],
            'level': record['level'], 'gold_verification': gold_check,
            'correct_count': k, 'sample_count': 4, 'observed_accuracy': k / 4,
            'normal_end_count': sum(s['normal_end'] for s in samples),
            'normal_correct_count': sum(s['normal_end'] and s['correct'] for s in samples),
            'group': label, 'samples': samples}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--adapter', required=True)
    p.add_argument('--data', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--max-tokens', type=int, default=3894)
    p.add_argument('--max-model-len', type=int, default=8192)
    p.add_argument('--workers', type=int, default=4)
    args = p.parse_args()
    from common import tokenizer, TEMPLATE
    from math_verification import require_math_verify
    import vllm
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    adapter = Path(args.adapter)
    assert sha(adapter / 'adapter_model.safetensors') == EXPECTED_ADAPTER
    # Verify all 11 uploaded files, not just the weights.
    upload_manifest = adapter.parent / 'download_manifest.json'
    if upload_manifest.exists():
        for name, spec in json.loads(upload_manifest.read_text())['files'].items():
            assert sha(adapter / name) == spec['sha256'], name
    cfg = json.loads((adapter / 'adapter_config.json').read_text())
    assert cfg['r'] == 32 and cfg['peft_type'] == 'LORA'
    math_version = require_math_verify()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    all_rows = read(Path(args.data))
    rows = all_rows[:args.limit] if args.limit else all_rows
    assert len({r['row_id'] for r in rows}) == len(rows)
    by_id = {r['row_id']: r for r in rows}
    manifest = {'data_sha256': sha(args.data), 'adapter_sha256': EXPECTED_ADAPTER,
                'model': args.model, 'model_config_sha256': sha(Path(args.model) / 'config.json'),
                'template': TEMPLATE, 'vllm_version': vllm.__version__,
                'math_verify_version': math_version, 'dtype': 'bfloat16',
                'n': 4, 'temperature': 1.0, 'top_p': 1.0, 'top_k': -1,
                'max_tokens': args.max_tokens, 'eos_token_id': 151643,
                'repetition_penalty': 1.0, 'stop_strings': [],
                'seed_policy': '3407 + first 8 hex digits of sha256(row_id), mod 2^31',
                'questions': len(rows), 'max_model_len': args.max_model_len,
                'selection': '1 <= Math-Verify correct final answers out of 4 <= 3; malformed answers fail closed; invalid gold quarantined',
                'note': 'Sample correctness is independent of format/EOS; four samples are a noisy estimate, not true difficulty.'}
    manifest_path = out / 'run_manifest.json'
    if manifest_path.exists():
        assert json.loads(manifest_path.read_text()) == manifest, 'Resume config mismatch'
    else:
        save(manifest_path, manifest)
    existing_raw = read(out / 'raw_generations.jsonl')
    existing_scores = read(out / 'scored_questions.jsonl')
    raw_ids = {r['row_id'] for r in existing_raw}
    scores = {r['row_id']: r for r in existing_scores}
    assert len(raw_ids) == len(existing_raw) and len(scores) == len(existing_scores)
    assert raw_ids <= by_id.keys() and scores.keys() <= raw_ids
    for record in existing_raw:
        assert len(record['samples']) == 4 and record['answer'] == by_id[record['row_id']]['answer']
    t = tokenizer(str(adapter))
    prompt_ids = {}
    for row in rows:
        assert [m['role'] for m in row['prompt']] == ['system', 'user']
        assert row['prompt'][1]['content'] == row['problem']
        ids = t.apply_chat_template(row['prompt'], tokenize=True, add_generation_prompt=True)
        assert len(ids) + args.max_tokens <= args.max_model_len, f'Prompt too long: {row["row_id"]}'
        prompt_ids[row['row_id']] = ids
    save(out / 'prompt_audit.json', {'questions': len(rows), 'max_prompt_tokens': max(map(len, prompt_ids.values())),
                                    'prompt_truncation': False, 'adapter_upload_all_files_verified': upload_manifest.exists()})
    remaining = [r for r in rows if r['row_id'] not in raw_ids]
    print('PRECHECK', json.dumps({'questions': len(rows), 'remaining': len(remaining),
                                'vllm': vllm.__version__, 'adapter_verified': True}), flush=True)
    started = time.time()
    pending = []
    context = multiprocessing.get_context('spawn')
    with concurrent.futures.ProcessPoolExecutor(args.workers, mp_context=context) as pool, \
            (out / 'raw_generations.jsonl').open('a', encoding='utf-8') as raw_stream, \
            (out / 'scored_questions.jsonl').open('a', encoding='utf-8') as score_stream:
        def collect(wait=False):
            keep = []
            for future in pending:
                if wait or future.done():
                    result = future.result()
                    assert result['row_id'] not in scores
                    append(score_stream, result)
                    scores[result['row_id']] = result
                else:
                    keep.append(future)
            pending[:] = keep
            save(out / 'progress.json', {'target_questions': len(rows), 'generated_questions': len(raw_ids),
                 'generated_answers': 4 * len(raw_ids), 'scored_questions': len(scores),
                 'groups': dict(Counter(s['group'] for s in scores.values())),
                 'elapsed_this_run_seconds': round(time.time()-started, 1), 'updated_unix': time.time()})

        for record in existing_raw:
            if record['row_id'] not in scores:
                pending.append(pool.submit(grade, record))
        if remaining:
            llm = LLM(model=args.model, tokenizer=str(adapter), dtype='bfloat16',
                      generation_config='vllm', enable_lora=True, max_lora_rank=32,
                      max_loras=1, max_model_len=args.max_model_len, gpu_memory_utilization=.88,
                      max_num_seqs=128, max_num_batched_tokens=8192,
                      enable_prefix_caching=True, enable_chunked_prefill=True, seed=3407)
            request = LoRARequest('sft_v3', 1, str(adapter))
            for start in range(0, len(remaining), args.batch_size):
                chunk = remaining[start:start+args.batch_size]
                params = [SamplingParams(n=4, temperature=1.0, top_p=1.0, top_k=-1,
                          repetition_penalty=1.0, max_tokens=args.max_tokens,
                          seed=(3407+int(hashlib.sha256(r['row_id'].encode()).hexdigest()[:8],16)) % (2**31),
                          stop_token_ids=[151643], ignore_eos=False) for r in chunk]
                generated = llm.generate([{'prompt_token_ids': prompt_ids[r['row_id']]} for r in chunk],
                                         params, lora_request=request, use_tqdm=False)
                assert len(generated) == len(chunk)
                for row, response in zip(chunk, generated):
                    assert list(response.prompt_token_ids) == prompt_ids[row['row_id']]
                    assert len(response.outputs) == 4
                    samples = []
                    for sample in sorted(response.outputs, key=lambda x: x.index):
                        samples.append({'sample_index': sample.index, 'completion': sample.text,
                             'tokens': len(sample.token_ids), 'finish_reason': sample.finish_reason,
                             'stop_reason': sample.stop_reason,
                             # vLLM may omit EOS from token_ids and use stop_reason=None.
                             # No string stops are configured and EOS is 151643.
                             'eos_finished': sample.finish_reason == 'stop' and sample.stop_reason in [None, 151643],
                             'truncated': sample.finish_reason == 'length'})
                    record = {k: row[k] for k in ['row_id','source','level','answer']}
                    record['samples'] = samples
                    append(raw_stream, record)
                    raw_ids.add(row['row_id'])
                    pending.append(pool.submit(grade, record))
                collect()
                print(f'PROGRESS generated={len(raw_ids)}/{len(rows)} answers={4*len(raw_ids)} scored={len(scores)} seconds={time.time()-started:.1f}', flush=True)
                # Bound memory if CPU grading lags; results are already persisted.
                if len(pending) > 4 * args.batch_size:
                    pending[0].result()
                    collect()
        collect(wait=True)
    assert len(scores) == len(rows) == len(raw_ids)
    for label in ['mixed', 'all_wrong', 'all_correct', 'ungradable_gold']:
        with (out / (label + '.jsonl')).open('w', encoding='utf-8') as stream:
            for row in rows:
                s = scores[row['row_id']]
                if s['group'] == label:
                    append(stream, {**row, 'screening': {k: s[k] for k in ['correct_count','sample_count','observed_accuracy','normal_end_count','normal_correct_count','group']}})
    summary = {'questions': len(rows), 'answers': 4*len(rows),
               'groups': dict(Counter(s['group'] for s in scores.values())),
               'correct_count_histogram': dict(Counter(s['correct_count'] for s in scores.values())),
               'by_source': {source: dict(Counter(s['group'] for s in scores.values() if s['source']==source)) for source in ['gsm8k','math']},
               'completed_unix': time.time()}
    save(out / 'summary.json', summary)
    print('COMPLETE', json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
