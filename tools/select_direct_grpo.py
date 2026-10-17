"""Select half the eligible G4 pool, easy:medium:hard = 3:4:3, without replacement."""
import argparse,collections,hashlib,json,math
from pathlib import Path


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return [json.loads(s) for s in Path(p).read_text(encoding='utf-8').splitlines() if s.strip()]
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')


def quotas(total,weights):
    exact={k:total*v/sum(weights.values()) for k,v in weights.items()}
    q={k:math.floor(v) for k,v in exact.items()}
    for k in sorted(q,key=lambda k:(-(exact[k]-q[k]),k))[:total-sum(q.values())]:q[k]+=1
    return q


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--scores',required=True)
    p.add_argument('--output',required=True);p.add_argument('--sft-data',required=True);p.add_argument('--development',required=True)
    p.add_argument('--test',required=True);a=p.parse_args()
    from prepare_math_self600 import canonical
    from common import messages
    from grpo_outcome_policy import POLICY, summarize
    rows=read(a.data);scores=read(a.scores);by_id={r['row_id']:r for r in rows}
    assert len(by_id)==len(rows) and {s['row_id'] for s in scores}==set(by_id)
    original_scores=scores
    score_root=Path(a.scores).parent
    gold=json.loads((score_root/'gold_checks.json').read_text(encoding='utf-8'))
    groups=[json.loads(f.read_text(encoding='utf-8')) for f in sorted((score_root/'groups').glob('*.json'))]
    scores=[summarize(g,gold[g['row_id']]) for g in groups]
    assert len(scores)==len(rows) and {s['row_id'] for s in scores}==set(by_id)
    assert {s['row_id']:s['k'] for s in scores}=={s['row_id']:s['k'] for s in original_scores}
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    with (out/'task_success_scores.jsonl').open('w',encoding='utf-8') as f:
        for s in scores:f.write(json.dumps(s,ensure_ascii=False)+'\n')
    pools={key:[] for key in ['easy','medium','hard']}
    for s in scores:
        if s['eligible']:
            assert s['u']==s['bad_end']==0 and s['gold_parser_supported']
            assert s['difficulty']=={3:'easy',2:'medium',1:'hard'}[s['k']]
            pools[s['difficulty']].append(s)
    total=sum(map(len,pools.values()));target=total//2
    q=quotas(target,dict(easy=3,medium=4,hard=3))
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    report=dict(eligible_pool=total,selected_target=target,half_rule='floor(eligible_pool/2)',
        ratio_target=dict(easy=.3,medium=.4,hard=.3),quotas=q,available={k:len(v) for k,v in pools.items()},
        difficulty_definition='G=4: easy k=3; medium k=2; hard k=1. Sampling estimate, not exact success probability.',
        data_sha256=sha(a.data),scores_sha256=sha(a.scores),seed=20261016)
    report.update(outcome_policy=POLICY,task_success_scores_sha256=sha(out/'task_success_scores.jsonl'),
        protocol_failure_policy='extraction_failed is unsuccessful task response, not proven mathematical error; prediction parse/verification errors remain unconfirmed',
        original_strict_eligible=sum(s['eligible'] for s in original_scores),
        protocol_failure_samples=sum(s['protocol_failures'] for s in scores))
    insufficient={k:q[k]-len(pools[k]) for k in q if q[k]>len(pools[k])}
    if insufficient or not target:
        report.update(status='quota_infeasible',shortfall=insufficient);save(out/'selection_report.json',report)
        raise RuntimeError('Cannot meet requested half and 3:4:3 without duplication: '+str(report))
    chosen=[]
    def rank(r):return hashlib.sha256(('20261016|'+r['row_id']).encode()).hexdigest()
    for difficulty,pool in pools.items():
        strata=collections.defaultdict(list)
        for s in pool:strata[(s['level'],s['type'])].append(s)
        weights={'|'.join(k):len(v) for k,v in strata.items()};subq=quotas(q[difficulty],weights)
        for key,ss in strata.items():chosen+=sorted(ss,key=rank)[:subq['|'.join(key)]]
    chosen=sorted(chosen,key=rank)
    assert len(chosen)==target==len({r['row_id'] for r in chosen})
    selected=[]
    for s in chosen:
        r=by_id[s['row_id']]
        selected.append(dict(row_id=r['row_id'],problem=r['problem'],answer=r['answer'],source='math',
            split='train',level=r['level'],type=r['type'],difficulty=s['difficulty'],screen_correct_count=s['k'],
            prompt=messages(r['problem'])))
    prompts={canonical(r['problem']) for r in selected};assert len(prompts)==len(selected)
    for file in [a.sft_data,a.development,a.test]:
        assert not prompts & {canonical(r['problem']) for r in read(file)},('training/evaluation overlap',file)
    with (out/'train.jsonl').open('w',encoding='utf-8') as f:
        for r in selected:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    report.update(status='complete',selected=len(selected),actual_counts=dict(collections.Counter(r['difficulty'] for r in selected)),
        actual_ratios={k:q[k]/target for k in q},train_sha256=sha(out/'train.jsonl'),
        no_exact_sft_dev_test_overlap=True,per_stratum=dict(collections.Counter(r['level']+' / '+r['type'] for r in selected)))
    save(out/'selection_report.json',report);print('GRPO_SELECTED',json.dumps(report),flush=True)


if __name__=='__main__':main()
