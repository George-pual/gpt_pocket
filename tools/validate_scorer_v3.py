"""Shadow-test the automatic scorer against the independent manual failure audit."""
import collections
import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from math_verification_v3_frozen import verify_final_answer, require_math_verify, SCHEMA

ROOT=Path(__file__).resolve().parents[1]/'datasets/math/base_self600_20261015'
AUDIT=ROOT/'grpo_log_analysis/failure_audit'
OUT=ROOT/'scorer_v3_validation'
SCORER_SOURCE=Path(__file__).with_name('math_verification_v3_frozen.py')


def read(p):
    return [json.loads(s) for s in Path(p).read_text(encoding='utf-8').splitlines() if s.strip()]


def main():
    packets=read(AUDIT/'packets.jsonl'); manual={r['audit_index']:r for r in read(AUDIT/'verified_candidates.jsonl')}
    results={}
    def one(p):
        verdict=verify_final_answer(p['completion'],p['gold'],question=p['problem'])
        return dict(audit_index=p['audit_index'],step=p['step'],sample_index=p['sample_index'],row_id=p['row_id'],
                    prediction=p['prediction'],gold=p['gold'],manual_status=manual[p['audit_index']]['audit_verdict']['status'],
                    verdict=verdict)
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(one,p) for p in packets]
        for f in as_completed(futures):
            r=f.result();results[r['audit_index']]=r
            if len(results)%30==0:print('V3_REAL_SAMPLES_CHECKED',len(results),len(packets),flush=True)
    rows=[results[i] for i in range(len(packets))]
    false_accept=[r['audit_index'] for r in rows if r['verdict']['correct'] and r['manual_status']!='correct']
    wrong_disagreement=[r['audit_index'] for r in rows if r['verdict']['status']=='incorrect' and r['manual_status']=='correct']
    rollout=read(ROOT/'remote_results/grpo_direct367_v1/rollouts.jsonl')
    consumed=read(ROOT/'remote_results/grpo_direct367_v1/train_consumption.jsonl')
    grouped=collections.defaultdict(list)
    for r in rollout:grouped[r['step']].append(r)
    mapped={(r['step'],r['sample_index']):r['verdict'] for r in rows}
    recovered=[]
    for c in consumed:
        ss=grouped[c['step']]
        vs=[mapped.get((c['step'],i),s['verdict']) for i,s in enumerate(ss)]
        if c['dropped_group'] and all(s['eos'] for s in ss) and all(v['status'] in ('correct','incorrect','extraction_failed') for v in vs):
            k=sum(v['correct'] for v in vs)
            recovered.append(dict(step=c['step'],row_id=c['row_id'],k=k,correctness_contrast=0<k<4))
    summary=dict(schema=SCHEMA,math_verify_version=require_math_verify(),samples=len(rows),
                 statuses=dict(collections.Counter(r['verdict']['status'] for r in rows)),
                 manual_agreement_on_confirmed=sum(r['verdict']['status'] in ('correct','incorrect') and r['verdict']['status']==r['manual_status'] for r in rows),
                 false_accept_indices=false_accept,wrong_disagreement_indices=wrong_disagreement,
                 recoverable_groups=len(recovered),recoverable_correctness_contrast=sum(r['correctness_contrast'] for r in recovered),
                 scorer_sha256=hashlib.sha256(SCORER_SOURCE.read_bytes()).hexdigest(),
                 audit_decisions_sha256=hashlib.sha256((AUDIT/'decisions.jsonl').read_bytes()).hexdigest(),
                 scope='Automatic shadow grading of saved training failures; manual audit is a development comparison, not a blind validation set; original scores unchanged')
    OUT.mkdir(exist_ok=True)
    with (OUT/'real_samples.jsonl').open('w',encoding='utf-8') as f:
        for r in rows:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False),flush=True)
    assert not false_accept,'Automatic scorer accepted answers not confirmed by independent manual audit'
    assert not wrong_disagreement,'Automatic scorer disagreed on an audited correct answer'


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
