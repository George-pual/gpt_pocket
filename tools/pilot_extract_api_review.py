"""Extract answers from the authorized eight responses; grade locally only."""
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import json,collections
from api_answer_review import APIAnswerReviewer,MathFirstAPIReward,PROMPT_VERSION
from scorer_runtime_guard import load_verified_scorer,load_verified_online_reward

REPO=Path(__file__).resolve().parents[1]
ROOT=REPO/'datasets/math/base_self600_20261015'
RESULTS=ROOT/'grpo_rescreen_v4_results'
OUT=ROOT/'api_answer_review_pilot/extract8_v4'

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    scorer,sruntime=load_verified_scorer()
    module,rruntime=load_verified_online_reward()
    primary=module.OnlineRewardScorer(scorer)
    recovered=[json.loads(l) for l in (RESULTS/'recovered_answer_candidates.jsonl').read_text(encoding='utf-8').splitlines()]
    chosen=[]
    for status in ('correct','incorrect'):chosen += [r for r in recovered if r['regraded_status']==status][:4]
    assert len(chosen)==8
    groups={}
    ids={r['row_id'] for r in chosen}
    for path in (RESULTS/'remote/screen_current_grpo/grpo_final/groups').glob('*.json'):
        g=json.loads(path.read_text(encoding='utf-8'))
        if g['row_id'] in ids:
            for s in g['samples']:groups[(g['row_id'],s['sample_index'])]=s
    rows=[]
    for r in chosen:
        sample=groups[(r['row_id'],r['sample_index'])]
        row=dict(r,completion=sample['completion'],eos=sample['eos'])
        row['primary']=primary.score_one(row['completion'],row['gold'],row['problem'],row['eos'])
        assert row['primary']['verdict']['status'] not in ('correct','incorrect')
        rows.append(row)
    (OUT/'request_manifest.jsonl').write_text(''.join(json.dumps(dict(row_id=r['row_id'],sample_index=r['sample_index'],problem=r['problem'],response=r['completion']),ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
    reviewer=APIAnswerReviewer(REPO/'configs/math_answer_review_api.json',OUT/'cache')
    finished=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        tasks={pool.submit(reviewer.review,r['problem'],r['completion']):r for r in rows}
        for task in as_completed(tasks):
            row=dict(tasks[task]);reply=task.result()
            class SavedReviewer:
                def review(self,*args):return reply
            result=MathFirstAPIReward(primary,SavedReviewer()).score_one(row['completion'],row['gold'],row['problem'],row['eos'])
            row['result']=result
            row['api_verdict']=result['verdict']['status'] if not result['grading_unconfirmed'] else 'unconfirmed'
            finished.append(row)
            (OUT/'records.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in finished),encoding='utf-8')
            print(json.dumps(dict(done=len(finished),total=8,extracted_answer=reply.get('extracted_answer'),result=row['api_verdict']),ensure_ascii=True),flush=True)
    errors=[dict(row_id=r['row_id'],sample_index=r['sample_index'],expected=r['regraded_status'],actual=r['api_verdict'])
            for r in finished if r['api_verdict']!='unconfirmed' and r['api_verdict']!=r['regraded_status']]
    usage=collections.Counter()
    for r in finished:
        for k,v in (r['result']['api_review'].get('usage') or {}).items():
            if isinstance(v,(int,float)):usage[k]+=v
    summary=dict(samples=8,api_calls=reviewer.calls,requested_model=reviewer.config['model'],
        returned_models=sorted({r['result']['api_review'].get('returned_model') for r in finished if r['result']['api_review'].get('returned_model')}),
        outcomes=dict(collections.Counter(r['api_verdict'] for r in finished)),disagreements=errors,
        unconfirmed=sum(r['api_verdict']=='unconfirmed' for r in finished),
        valid_extractions=sum(r['result']['api_review'].get('status')=='extracted' for r in finished),
        usage=dict(usage),scorer_runtime=sruntime,reward_runtime=rruntime,prompt_version=PROMPT_VERSION,
        api_input_fields=['problem','response'],references_sent=False,training_started=False,
        scope='Eight reviewed development cases, not a blind accuracy guarantee')
    (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=True,indent=2),flush=True)

if __name__=='__main__':main()

