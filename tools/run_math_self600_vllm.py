"""Base-only vLLM generation with atomic per-question evidence and safe resume."""
import argparse
import collections
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(p, value):
    tmp = p.with_suffix(p.suffix + '.tmp')
    with tmp.open('w', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(p)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--batch-size', type=int, default=16)
    args = parser.parse_args()
    from common import tokenizer, messages
    from math_verification import require_math_verify, verify_final_answer
    from vllm import LLM, SamplingParams
    require_math_verify()
    data, out, model = Path(args.data_dir), Path(args.output), Path(args.model)
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / 'generation.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    selection = json.loads((data / 'selection_manifest.json').read_text())
    assert digest(data / 'questions.jsonl') == selection['output_sha256']['questions.jsonl']
    rows = [json.loads(s) for s in (data / 'questions.jsonl').read_text().splitlines() if s.strip()]
    assert len(rows) == len({r['row_id'] for r in rows}) == 600
    assert collections.Counter(r['level'] for r in rows) == {'Level 2': 270, 'Level 3': 330}
    assert json.loads((model / 'download_manifest.json').read_text())['repo'] == 'Qwen/Qwen3-4B-Base'
    t = tokenizer(model)
    prompts = [t.apply_chat_template(messages(r['problem']), tokenize=True, add_generation_prompt=True) for r in rows]
    assert max(map(len, prompts)) + 3894 <= 8192
    params = dict(n=4, temperature=1.0, top_p=1.0, top_k=-1, max_tokens=3894, stop_token_ids=[t.eos_token_id])
    weights = sorted(set(json.loads((model / 'model.safetensors.index.json').read_text())['weight_map'].values()))
    frozen = dict(model=str(model), model_repo='Qwen/Qwen3-4B-Base', representation='Original dense BF16; no adapter',
                  data_sha256=digest(data / 'questions.jsonl'), selection_sha256=digest(data / 'selection_manifest.json'),
                  weights_sha256={f: digest(model / f) for f in weights},
                  sampling=params, seed_policy='20261015 + frozen question index; n=4 per request',
                  prompt_token_ids_sha256=hashlib.sha256(json.dumps(prompts).encode()).hexdigest(),
                  chat_template=t.chat_template, eos_token_id=t.eos_token_id,
                  versions={k: importlib.metadata.version(k) for k in ('vllm', 'transformers', 'torch', 'math-verify')},
                  batch_size=args.batch_size, max_model_len=8192,
                  correctness_boundary='Final-answer equivalence only; reasoning must be reviewed separately.')
    mp = out / 'run_manifest.json'
    if mp.exists():
        assert json.loads(mp.read_text()) == frozen, 'Resume configuration or inputs changed'
    else:
        save(mp, frozen)
    groups = out / 'groups'
    groups.mkdir(exist_ok=True)
    done = {}
    for p in groups.glob('*.json'):
        records = json.loads(p.read_text())
        assert len(records) == 4 and {r['sample_index'] for r in records} == set(range(4))
        key = records[0]['row_id']
        assert key not in done and all(r['row_id'] == key for r in records)
        done[key] = records
    assert set(done) <= {r['row_id'] for r in rows}
    def progress(status):
        save(out / 'progress.json', dict(status=status, completed_questions=len(done), total_questions=600,
             completed_answers=len(done) * 4, total_answers=2400, updated_at_unix=time.time()))
    progress('loading_model')
    pending = [(i, r) for i, r in enumerate(rows) if r['row_id'] not in done]
    if pending:
        llm = LLM(model=str(model), dtype='bfloat16', max_model_len=8192, gpu_memory_utilization=.85,
                  max_num_seqs=64, seed=20261015, enable_prefix_caching=True, disable_log_stats=True)
        for offset in range(0, len(pending), args.batch_size):
            part = pending[offset:offset + args.batch_size]
            batch = [{'prompt_token_ids': prompts[i]} for i, r in part]
            outputs = llm.generate(batch, [SamplingParams(**params, seed=20261015+i) for i, r in part], use_tqdm=False)
            assert len(outputs) == len(part)
            for (i, row), result in zip(part, outputs):
                assert list(result.prompt_token_ids) == prompts[i] and len(result.outputs) == 4
                records = []
                for s in result.outputs:
                    records.append(dict(row_id=row['row_id'], sample_index=s.index,
                        completion=s.text, token_ids=list(s.token_ids), tokens=len(s.token_ids),
                        finish_reason=s.finish_reason, stop_reason=s.stop_reason,
                        eos_finished=s.finish_reason == 'stop' and s.stop_reason in (None, t.eos_token_id),
                        truncated=s.finish_reason == 'length', question_index=i, seed=20261015+i))
                assert {r['sample_index'] for r in records} == set(range(4))
                records.sort(key=lambda r: r['sample_index'])
                save(groups / (hashlib.sha256(row['row_id'].encode()).hexdigest() + '.json'), records)
                done[row['row_id']] = records
            progress('generating')
            print('GENERATION_PROGRESS', len(done), '/600', len(done)*4, '/2400', flush=True)
        llm.llm_engine.engine_core.shutdown()
    progress('grading')
    raw_tmp = out / 'raw_generations.jsonl.tmp'
    scored_tmp = out / 'scored_generations.jsonl.tmp'
    review_tmp = out / 'answer_review_queue.jsonl.tmp'
    summary = collections.Counter()
    with raw_tmp.open('w', encoding='utf-8') as raw, scored_tmp.open('w', encoding='utf-8') as scored, review_tmp.open('w', encoding='utf-8') as review:
        for row in rows:
            candidates = []
            for record in done[row['row_id']]:
                raw.write(json.dumps(record, ensure_ascii=False) + '\n')
                verdict = verify_final_answer(record['completion'], row['answer'])
                judged = dict(record, gold=row['answer'], math_verification=verdict,
                              reasoning_review_status='pending')
                scored.write(json.dumps(judged, ensure_ascii=False) + '\n')
                summary['answers'] += 1
                summary['final_answer_correct'] += bool(verdict['correct'])
                summary['truncated'] += record['truncated']
                summary['eos_finished'] += record['eos_finished']
                if verdict['correct'] and record['eos_finished'] and not record['truncated']:
                    candidates.append(record['sample_index'])
            summary['questions_with_answer_correct_eos_candidate'] += bool(candidates)
            review.write(json.dumps(dict(row_id=row['row_id'], problem=row['problem'], gold=row['answer'],
                         candidate_sample_indices=candidates, reference_solution=row['reference_solution'],
                         reasoning_review_status='pending', final_training_approved=False), ensure_ascii=False) + '\n')
    raw_tmp.replace(out / 'raw_generations.jsonl')
    scored_tmp.replace(out / 'scored_generations.jsonl')
    review_tmp.replace(out / 'answer_review_queue.jsonl')
    save(out / 'summary.json', dict(status='complete', questions=600, **dict(summary),
         data_sha256=frozen['data_sha256'], reasoning_review_status='pending',
         raw_sha256=digest(out / 'raw_generations.jsonl'), scored_sha256=digest(out / 'scored_generations.jsonl')))
    progress('complete')
    print('RUN_COMPLETE', json.dumps(dict(summary)), flush=True)


if __name__ == '__main__':
    main()
