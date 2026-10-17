"""Offline regrade existing completions; no model inference or source changes."""
import concurrent.futures
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'closure_grpo_experiment'))

def worker(pair):
    # Parent bounds each outstanding task and terminates workers on timeout.
    import logging
    logging.getLogger('math_verify.parser').setLevel(logging.ERROR)
    logging.getLogger('math_verify.grader').setLevel(logging.ERROR)
    from math_verification import _grade_pair, extract_final_answer, SCHEMA
    text, gold = pair
    pred, source, error = extract_final_answer(text)
    from math_verification import completion_text
    if source == 'boxed' and len(re.findall(r'\\boxed\s*\{', completion_text(text))) != 1:
        error = 'multiple_boxed_answers'
    result = dict(schema=SCHEMA, answer_source=source, prediction=pred,
                  correct=False, status='extraction_failed', error=error)
    if not error:
        result.update(_grade_pair(pred, gold, None))
    return result

def main():
    from math_verification import SCHEMA, require_math_verify
    r = ROOT / 'datasets/gsm8k/base_sftv3_test/results'
    paths = {'base':r/'results_math_l2_l3/base/answers.jsonl',
             'sft_v3':r/'results_math_l2_l3/sft_v3/answers.jsonl',
             'grpo1024':r/'grpo1024_test/math/grpo1024/answers.jsonl',
             'grpo1280':r/'grpo1280_test/math/grpo1280/answers.jsonl',
             'grpo1536':r/'grpo1536_official_eval/results/math/grpo1536/answers.jsonl'}
    out = r/'math_grader_v2_rescore'
    out.mkdir(exist_ok=True)
    summary = dict(schema=SCHEMA, math_verify_version=require_math_verify(), models={},
                   scorer_sha256=hashlib.sha256((ROOT/'closure_grpo_experiment/math_verification.py').read_bytes()).hexdigest())
    pool = concurrent.futures.ProcessPoolExecutor(max_workers=4)
    try:
        for name, path in paths.items():
            original = path.read_bytes()
            rows = [json.loads(x) for x in original.splitlines() if x.strip()]
            assert len(rows)==200
            futures = [pool.submit(worker,(x['completion'],x['gold'])) for x in rows]
            changes=[]
            for row, future in zip(rows, futures):
                verdict=future.result(timeout=30)
                row['math_verification_v2']=verdict
                if bool(row['correct'])!=verdict['correct']:
                    changes.append(dict(row_id=row['row_id'],old_correct=row['correct'],new_correct=verdict['correct'],
                                        prediction=verdict['prediction'],gold=row['gold'],new_status=verdict['status']))
            assert path.read_bytes()==original
            (out/f'{name}.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows),encoding='utf-8')
            stats=dict(total=200,old_correct=sum(x['correct'] for x in rows),
                       new_correct=sum(x['math_verification_v2']['correct'] for x in rows),
                       changes=changes,input_sha256=hashlib.sha256(original).hexdigest())
            summary['models'][name]=stats
            print(name,stats['old_correct'],stats['new_correct'],flush=True)
        (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    finally:
        # Also kills pathological symbolic tasks instead of waiting indefinitely.
        for p in list((getattr(pool,'_processes',None) or {}).values()):
            if p.is_alive(): p.terminate()
        pool.shutdown(wait=True,cancel_futures=True)

if __name__=='__main__':
    main()
