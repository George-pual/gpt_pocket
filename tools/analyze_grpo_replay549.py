"""Reconcile replay549 rewards, exclusions and learning signals by question."""
from collections import Counter,defaultdict
import hashlib,json,math,statistics
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]/'datasets/math/base_self600_20261015'
RUN=ROOT/'grpo_replay549_extract_v4_20261016_batch12'
SOURCE=RUN/'training_analysis_source'
OUT=RUN/'training_analysis'
def read(p):return [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines() if s.strip()]
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def mean(values):return statistics.mean(values) if values else None

def main():
    consumed=read(SOURCE/'training/train_consumption.jsonl');samples=read(SOURCE/'training/rollouts.jsonl')
    trainer=[x for x in load(SOURCE/'training/trainer_log.json') if 'grad_norm' in x]
    selected={r['row_id']:r for r in read(RUN/'data/train.jsonl')}
    assert len(consumed)==len(trainer)==len(selected)==549 and len(samples)==2196
    assert {x['step'] for x in trainer}==set(range(1,550))
    assert len({x['row_id'] for x in consumed})==549
    groups=defaultdict(list)
    for s in samples:groups[s['row_id']].append(s)
    rows=[]
    for c in consumed:
        ss=groups[c['row_id']];assert len(ss)==4 and {x['sample_index'] for x in ss}==set(range(4))
        rewards=[x['total_reward'] for x in ss];k=sum(x['verdict']['status']=='correct' for x in ss)
        for s in ss:
            assert math.isclose(s['total_reward'],float(s['verdict']['status']=='correct')+.1*float(s['verdict']['status']=='correct' and s['format_eos']))
            assert s['rewards']==[float(s['verdict']['status']=='correct'),float(s['verdict']['status']=='correct' and s['format_eos'])]
        sd=statistics.stdev(rewards);avg=mean(rewards)
        unknown=any(s['grading_unconfirmed'] for s in ss);bad_end=any(not s['eos'] for s in ss)
        assert c['dropped_group']==(unknown or bad_end)
        assert c['nonzero_advantage']==(not c['dropped_group'] and sd>1e-8)
        primary_unknown=[]
        for s in ss:
            v=dict(s['raw_verdict']);checks=s['numeric_equality_checks']
            if any(x['verdict']['status']=='incorrect' for x in checks):v.update(status='incorrect',correct=False)
            elif any(x['verdict']['status']!='correct' for x in checks):v.update(status='prediction_parse_failed',correct=False)
            primary_unknown.append(v['status'] not in {'correct','incorrect'} and not (v['status']=='extraction_failed' and v.get('error') in {'empty_answer','missing_final_answer'}))
        r=selected[c['row_id']]
        rows.append(dict(**c,online_k=k,subject=r['type'],difficulty=r['difficulty'],level=r['level'],
            unknown=unknown,bad_end=bad_end,math_contrast=not c['dropped_group'] and 0<k<4,
            format_only=not c['dropped_group'] and k in (0,4) and sd>1e-8,
            reward_equal=not c['dropped_group'] and sd<=1e-8,reward_std=sd,
            abs_advantage=mean([abs((x-avg)/(sd+1e-4)) for x in rewards]) if not c['dropped_group'] else 0,
            without_api_would_drop=bad_end or any(primary_unknown),
            group_mean_reward=avg,strict_format_answers=sum(s['format_eos'] for s in ss)))
    def summarize(rr,ll):
        active=[x for x in rr if not x['dropped_group']]
        return dict(questions=len(rr),active=len(active),dropped=len(rr)-len(active),
            math_contrast=sum(x['math_contrast'] for x in rr),format_only=sum(x['format_only'] for x in rr),
            reward_equal=sum(x['reward_equal'] for x in rr),nonzero_advantage=sum(x['nonzero_advantage'] for x in rr),
            mean_abs_advantage_active=mean([x['abs_advantage'] for x in active]),
            mean_reward_std_active=mean([x['reward_std'] for x in active]),
            correct_answers_active=sum(x['online_k'] for x in active),answers_active=4*len(active),
            online_k_active=dict(Counter(x['online_k'] for x in active)),
            strict_format_answers_active=sum(x['strict_format_answers'] for x in active),
            mean_lr=mean([x['learning_rate'] for x in ll]),mean_kl=mean([x['kl'] for x in ll if 'kl' in x]),
            mean_entropy=mean([x['entropy'] for x in ll if 'entropy' in x]),
            median_grad_norm=statistics.median(x['grad_norm'] for x in ll) if ll else None,
            max_grad_norm=max((x['grad_norm'] for x in ll),default=None))
    overall=summarize(rows,trainer);assert overall['active']==481 and overall['dropped']==68 and overall['nonzero_advantage']==413
    assert overall['math_contrast']+overall['format_only']+overall['reward_equal']+overall['dropped']==549
    windows={}
    for lo,hi in [(1,183),(184,366),(367,549),(1,99),(451,549)]:
        rr=[x for x in rows if lo<=x['step']+1<=hi];ll=[x for x in trainer if lo<=x['step']<=hi]
        windows[f'{lo}-{hi}']=summarize(rr,ll)
    api=[s for s in samples if (s.get('api_review') or {}).get('status')!='not_called']
    restored=[s for s in api if s['verdict']['status'] in {'correct','incorrect'} and not s['grading_unconfirmed']]
    oldc=read(ROOT/'remote_results/grpo_direct367_v1/train_consumption.jsonl')
    olds=read(ROOT/'remote_results/grpo_direct367_v1/rollouts.jsonl');oldgroups=defaultdict(list)
    for s in olds:oldgroups[s['row_id']].append(s)
    oldmix=sum(not c['dropped_group'] and 0<sum(s['rewards'][0] for s in oldgroups[c['row_id']])<4 for c in oldc)
    result=dict(overall=overall,windows=windows,
        exclusion_reasons=dict(unconfirmed_only=sum(x['unknown'] and not x['bad_end'] for x in rows),
            abnormal_end_only=sum(x['bad_end'] and not x['unknown'] for x in rows),both=sum(x['unknown'] and x['bad_end'] for x in rows)),
        api=dict(attempted_answers=len(api),extracted_answers=sum(s['api_review']['status']=='extracted' for s in api),
            confirmed_after_extraction=len(restored),confirmed_correct=sum(s['verdict']['status']=='correct' for s in restored),
            confirmed_incorrect=sum(s['verdict']['status']=='incorrect' for s in restored),
            errors=dict(Counter(s['api_review'].get('error') for s in api if s['api_review'].get('error'))),
            without_api_dropped_counterfactual=sum(x['without_api_would_drop'] for x in rows),
            groups_rescued_from_unknown=sum(x['without_api_would_drop'] and not x['dropped_group'] for x in rows),
            groups_newly_unknown_with_api=sum(not x['without_api_would_drop'] and x['dropped_group'] for x in rows)),
        stability=dict(nonfinite_gradient=sum(not math.isfinite(x['grad_norm']) for x in trainer),
            zero_gradient_updates=sum(x['grad_norm']==0 for x in trainer),preclip_grad_norm_over_one=sum(x['grad_norm']>1 for x in trainer),
            maximum_kl=max(x.get('kl',0) for x in trainer),median_kl=statistics.median(x['kl'] for x in trainer if 'kl' in x),
            maximum_policy_clip_fraction=max(x.get('clip_ratio/region_mean',0) for x in trainer),
            mean_policy_clip_fraction=mean([x.get('clip_ratio/region_mean',0) for x in trainer])),
        previous_round=dict(questions=len(oldc),active=sum(not x['dropped_group'] for x in oldc),
            dropped=sum(x['dropped_group'] for x in oldc),math_contrast=oldmix,
            caveat='Different initial model, selected questions and reward policy; descriptive comparison, not controlled causal attribution'),
        by_subject={k:summarize([x for x in rows if x['subject']==k],[]) for k in sorted({x['subject'] for x in rows})},
        by_difficulty={k:summarize([x for x in rows if x['difficulty']==k],[]) for k in ['easy','medium','hard']},
        source_sha256={str(p.relative_to(SOURCE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in SOURCE.rglob('*') if p.is_file()},
        caveats=['Each phase contains different randomly ordered questions, not repeated evaluation of one fixed set.',
            'Advantages are reconstructed from stored rewards with sample standard deviation and epsilon 1e-4, not saved tensors.',
            'Microbatch2/accum2 uses cached generations: three G4 groups share the generation step; do not join each group to a single optimizer update.',
            'The no-API calculation only reconstructs masking on the same saved responses; it does not simulate retraining.',
            'Final held-out test was still running when the snapshot was captured.'])
    OUT.mkdir(exist_ok=True)
    (OUT/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (OUT/'groups.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in {'source_sha256','by_subject','by_difficulty'}},ensure_ascii=True,indent=2),flush=True)

if __name__=='__main__':main()
