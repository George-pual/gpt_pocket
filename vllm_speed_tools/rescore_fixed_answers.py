"""Reproducible CPU-only rescore of all ten fixed-answer files.

No generation, SSH, model/adapter access or reward edits. Original inputs immutable.
"""
from __future__ import annotations
import argparse
import concurrent.futures as cf
import hashlib
import importlib.metadata
import json
import logging
import platform
import statistics
import time
from collections import Counter
from pathlib import Path
from audited_answer_scoring import SCHEMA, score_answer

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'datasets/gsm8k/base_sftv3_test/results'
SOURCES = {
    'base': ('results_vllm/base', 'results_math_l2_l3/base'),
    'sft_v3': ('results_vllm/sft_v3', 'results_math_l2_l3/sft_v3'),
    'grpo1024': ('grpo1024_test/gsm8k/grpo1024', 'grpo1024_test/math/grpo1024'),
    'grpo1280': ('grpo1280_test/gsm8k/grpo1280', 'grpo1280_test/math/grpo1280'),
    'grpo1536': ('grpo1536_official_eval/results/gsm8k/grpo1536', 'grpo1536_official_eval/results/math/grpo1536'),
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def worker(payload):
    logging.getLogger('math_verify').setLevel(logging.ERROR)
    completion, gold, question, eos = payload
    return score_answer(completion, gold, question, eos)


def bounded_map(tasks, workers, timeout):
    # Keep only workers tasks outstanding, and bound actual in-flight jobs.
    # On a pathological parse kill all workers, retry unaffected tasks once
    # they have a fresh pool. A timed-out task gets an explicit non-correct status.
    pending = list(range(len(tasks)))
    results = {}
    pool = cf.ProcessPoolExecutor(max_workers=workers)
    active = {}
    last_print = time.monotonic()
    try:
        while pending or active:
            while pending and len(active) < workers:
                i = pending.pop(0)
                active[pool.submit(worker, tasks[i])] = (i, time.monotonic())
            done, _ = cf.wait(active, timeout=.1, return_when=cf.FIRST_COMPLETED)
            for f in done:
                i, _ = active.pop(f)
                results[i] = f.result()
            expired = [f for f, (_, started) in active.items() if time.monotonic()-started > timeout]
            if expired:
                for f in expired:
                    i, _ = active.pop(f)
                    results[i] = dict(schema=SCHEMA, correct=False, status='verification_timeout',
                                      error='process_timeout', prediction=None, answer_source='unknown')
                pending = [i for i, _ in active.values()] + pending
                active.clear()
                for process in pool._processes.values():
                    process.terminate()
                pool.shutdown(wait=True, cancel_futures=True)
                pool = cf.ProcessPoolExecutor(max_workers=workers)
            if time.monotonic()-last_print > 20:
                print(f'SCORED {len(results)}/{len(tasks)} unique tasks', flush=True)
                last_print = time.monotonic()
        return [results[i] for i in range(len(tasks))]
    finally:
        for process in (pool._processes or {}).values():
            if process.is_alive(): process.terminate()
        pool.shutdown(wait=True, cancel_futures=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default=str(RESULTS/'audited_scorer_v3_final_r2'))
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--timeout', type=float, default=20)
    args = parser.parse_args()
    out = Path(args.output)
    if out.exists(): raise RuntimeError('Refuse overwrite: '+str(out))
    out.mkdir(parents=True)
    records, tasks, lookup, inputs, configs, baseline = {}, [], {}, {}, {}, {}
    for model, folders in SOURCES.items():
        for kind, folder in zip(('gsm8k','math'), folders):
            path = RESULTS/folder/'answers.jsonl'
            rows = [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines() if x.strip()]
            n = 1319 if kind == 'gsm8k' else 200
            assert len(rows)==len({r['row_id'] for r in rows})==n, (model,kind)
            key = model+'/'+kind
            inputs[key] = dict(path=str(path), sha256=digest(path), count=n)
            config_path = path.parent.parent/'evaluation_config.json'
            config = json.loads(config_path.read_text(encoding='utf-8'))
            configs[key] = config
            signatures = [(r['row_id'],r['question'],r['gold']) for r in rows]
            if kind not in baseline:
                baseline[kind] = (signatures,config)
            else:
                assert signatures==baseline[kind][0], ('question/reference mismatch',key)
                for field in ('data_sha256','prompt_tokens_sha256','sampling','chat_template'):
                    assert config[field]==baseline[kind][1][field], ('generation setting mismatch',key,field)
            records[key] = rows
            for row in rows:
                task = (row['completion'],row['gold'],row['question'],row['eos'] and not row['truncated'])
                if task not in lookup:
                    lookup[task] = len(tasks); tasks.append(task)
    scorer = Path(__file__).with_name('audited_answer_scoring.py')
    manifest = dict(schema=SCHEMA, inputs=inputs, source_configs=configs,
                    scorer_sha256=digest(scorer), runner_sha256=digest(Path(__file__)),
                    regression_tests_sha256=digest(Path(__file__).with_name('test_audited_answer_scoring.py')),
                    packages={p:importlib.metadata.version(p) for p in ('math-verify','sympy','latex2sympy2-extended','antlr4-python3-runtime')},
                    python=platform.python_version(), workers=args.workers, task_timeout_seconds=args.timeout,
                    float_rounding=10, numeric_precision=15, unique_tasks=len(tasks),
                    records=sum(len(x) for x in records.values()), inputs_aligned=True,
                    scope='Offline saved-answer rescore only; no model inference or training.',
                    definitions={'accuracy':'Unambiguous extracted final-answer mathematical equivalence, including question unit context; no last-number fallback.',
                    'correct_and_format':'Answer equivalence AND original format_eos.',
                    'format_eos':'Unchanged original format metric; inspect original evaluation code for historical definition.',
                    'failure':'Any status other than correct/incorrect; counted as not-correct in accuracy, separately reported.'})
    (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print('ALIGNED:',manifest['records'],'records;',len(tasks),'unique tasks',flush=True)
    # Gold self-checks are separate from scored model responses.
    gold_tasks=[('<SOLUTION>'+r[2]+'</SOLUTION>',r[2],r[1],True)
                for signatures,_ in baseline.values() for r in signatures]
    gold_checks=bounded_map(gold_tasks,args.workers,args.timeout)
    assert all(x['correct'] for x in gold_checks), [(t[1],v) for t,v in zip(gold_tasks,gold_checks) if not v['correct']]
    (out/'reference_self_checks.json').write_text(json.dumps(dict(total=len(gold_checks),passed=True,
        status_counts=dict(Counter(x['status'] for x in gold_checks))),indent=2),encoding='utf-8')
    verdicts = bounded_map(tasks,args.workers,args.timeout)
    summary, changes, review = {}, [], []
    for key, rows in records.items():
        model, kind = key.split('/')
        folder = out/kind; folder.mkdir(exist_ok=True)
        for row in rows:
            task=(row['completion'],row['gold'],row['question'],row['eos'] and not row['truncated'])
            verdict=verdicts[lookup[task]]
            row['audited_score_v3']=verdict
            if bool(row['correct']) != verdict['correct']:
                changes.append(dict(model=model,dataset=kind,row_id=row['row_id'],old_correct=row['correct'],
                                    new_correct=verdict['correct'],gold=row['gold'],question=row['question'],
                                    completion=row['completion'],verdict=verdict))
            if verdict['status'] not in ('correct','incorrect'):
                review.append(dict(model=model,dataset=kind,row_id=row['row_id'],gold=row['gold'],question=row['question'],
                                   completion=row['completion'],verdict=verdict))
        (folder/f'{model}.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows),encoding='utf-8')
        n=len(rows); correct=sum(x['audited_score_v3']['correct'] for x in rows)
        summary[key]=dict(total=n,old_correct=sum(x['correct'] for x in rows),correct=correct,accuracy=correct/n,
                          format_eos=sum(x['format_eos'] for x in rows),eos=sum(x['eos'] for x in rows),
                          truncated=sum(x['truncated'] for x in rows),mean_tokens=statistics.mean(x['tokens'] for x in rows),
                          correct_and_format=sum(x['audited_score_v3']['correct'] and x['format_eos'] for x in rows),
                          status_counts=dict(Counter(x['audited_score_v3']['status'] for x in rows)),
                          source_counts=dict(Counter(x['audited_score_v3']['answer_source'] for x in rows)))
        if kind=='math':
            summary[key]['by_level']={level:dict(total=sum(x['level']==level for x in rows),
                correct=sum(x['audited_score_v3']['correct'] for x in rows if x['level']==level)) for level in ('Level 2','Level 3')}
        print(key,summary[key]['old_correct'],'->',correct,'/',n,flush=True)
    for key, item in inputs.items():
        assert digest(Path(item['path']))==item['sha256'], ('input changed',key)
    for name, rows in [('changes',changes),('review_queue',review)]:
        (out/(name+'.jsonl')).write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows),encoding='utf-8')
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    (out/'audited_answer_scoring.py').write_bytes(scorer.read_bytes())
    (out/'rescore_fixed_answers.py').write_bytes(Path(__file__).read_bytes())
    (out/'test_audited_answer_scoring.py').write_bytes(Path(__file__).with_name('test_audited_answer_scoring.py').read_bytes())
    print('COMPLETE; raw input hashes unchanged',flush=True)


if __name__ == '__main__': main()
