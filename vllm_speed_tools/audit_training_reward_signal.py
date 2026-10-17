"""Counterfactual v2 reward audit of archived MATH rollouts, not retraining."""
import concurrent.futures
import json
from collections import Counter, defaultdict
from pathlib import Path
from rescore_math_v2 import worker

ROOT=Path(__file__).resolve().parents[1]
T=ROOT/'datasets/grpo_candidates_gsm8k_math_l2_l3/training_audit_1536'

def read(p): return [json.loads(x) for x in p.read_bytes().splitlines() if x.strip()]
def sign(x): return 1 if x>1e-8 else -1 if x < -1e-8 else 0

def main():
    phases=['E1_sft_v3_half_math1024_fast_v2','E1_sft_v3_expanded_continue1536_v1']
    rows=[]
    for phase in phases:
        for x in read(T/phase/'B_components/rollouts.jsonl'):
            if x['source']=='math': rows.append(dict(x,phase=phase))
    groups=defaultdict(list)
    changed=[]
    pool=concurrent.futures.ProcessPoolExecutor(max_workers=4)
    try:
        futures=[pool.submit(worker,(x['completion'],x['gold'])) for x in rows]
        for i,(x,f) in enumerate(zip(rows,futures)):
            v=f.result(timeout=30)
            x['v2_verdict']=v
            groups[(x['phase'],x['step'],x['row_id'])].append(x)
            if bool(x['rewards'][0])!=v['correct']: changed.append(x)
            if (i+1)%2000==0: print('graded',i+1,flush=True)
    finally:
        for p in list((getattr(pool,'_processes',None) or {}).values()):
            if p.is_alive(): p.terminate()
        pool.shutdown(wait=True,cancel_futures=True)
    impact=[]
    signs=Counter()
    for (phase,step,rid),g in groups.items():
        assert len(g)==4
        old=[x['rewards'][0]+.1*x['rewards'][1] for x in g]
        new=[float(x['v2_verdict']['correct'])+.1*x['rewards'][1] for x in g]
        if old==new: continue
        dropped=any(x['truncated'] for x in g)
        om,nm=sum(old)/4,sum(new)/4
        os,ns=[sign(x-om) for x in old],[sign(x-nm) for x in new]
        if not dropped:
            for a,b in zip(os,ns): signs[f'{a}->{b}']+=1
        impact.append(dict(phase=phase,step=step,row_id=rid,dropped=dropped,old_total_reward=old,
                           new_total_reward=new,old_signal_sign=os,new_signal_sign=ns,
                           old_correct=sum(x['rewards'][0] for x in g),new_correct=sum(x['v2_verdict']['correct'] for x in g)))
    summary=dict(math_answers=len(rows),math_groups=len(groups),changed_answers=len(changed),
                 zero_to_one=sum(not x['rewards'][0] and x['v2_verdict']['correct'] for x in changed),
                 one_to_zero=sum(bool(x['rewards'][0]) and not x['v2_verdict']['correct'] for x in changed),
                 affected_groups=len(impact),affected_effective_groups=sum(not x['dropped'] for x in impact),
                 effective_signal_sign_transitions=dict(signs),
                 limitation='V2-detected changes only; unrecognized prose and other grader errors may remain. This reconstructs reward-centered signs, not actual logged token gradients or a causal training rerun.')
    out=T/'reward_signal_audit_v2';out.mkdir(exist_ok=True)
    (out/'changed_answers.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in changed),encoding='utf-8')
    (out/'affected_groups.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in impact),encoding='utf-8')
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
