"""Offline audit of parser failures and actual correctness-contrast coverage."""
import argparse
import collections
import hashlib
import json
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]/'datasets/math/base_self600_20261015'
REMOTE=ROOT/'remote_results'
OUT=ROOT/'grpo_log_analysis/failure_audit'


def read(p):
    return [json.loads(s) for s in Path(p).read_text(encoding='utf-8').splitlines() if s.strip()]


def save(p,obj):
    Path(p).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')


def grade():
    sys.path.insert(0,str(REMOTE/'tools'))
    from math_verification import verify_final_answer, require_math_verify
    assert require_math_verify()=='0.8.0'
    packets=read(OUT/'packets.jsonl'); decisions=read(OUT/'decisions.jsonl')
    assert len(packets)==len(decisions)==139
    results={}
    def one(d):
        i=d['audit_index']; p=packets[i]
        assert d['row_id']==p['row_id'] and d['step']==p['step'] and d['sample_index']==p['sample_index']
        verdict=verify_final_answer('<SOLUTION>'+d['candidate']+'</SOLUTION>',p['gold'])
        return dict(**d,original_prediction=p['prediction'],gold=p['gold'],audit_verdict=verdict)
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures={pool.submit(one,d):d for d in decisions if d['decision']=='clear_final_answer'}
        for f in as_completed(futures):
            r=f.result();results[r['audit_index']]=r
            if len(results)%20==0:print('AUDIT_GRADED',len(results),len(futures),flush=True)
    for d in decisions:
        if d['decision']=='unresolved':
            p=packets[d['audit_index']]
            results[d['audit_index']]=dict(**d,original_prediction=p['prediction'],gold=p['gold'],
                                         audit_verdict=dict(status='manual_unresolved',correct=False))
    with (OUT/'verified_candidates.jsonl').open('w',encoding='utf-8') as f:
        for i in range(139):f.write(json.dumps(results[i],ensure_ascii=False)+'\n')
    print('AUDIT_VERDICTS',dict(collections.Counter(r['audit_verdict']['status'] for r in results.values())),flush=True)


def analyze():
    consumed=read(REMOTE/'grpo_direct367_v1/train_consumption.jsonl')
    rollouts=read(REMOTE/'grpo_direct367_v1/rollouts.jsonl')
    chosen=read(REMOTE/'grpo_selected_direct367_v1/train.jsonl')
    selected={r['row_id']:r for r in chosen}
    verdicts=read(OUT/'verified_candidates.jsonl')
    recovered={(r['step'],r['sample_index']):r for r in verdicts}
    groups=collections.defaultdict(list)
    for r in rollouts:groups[r['step']].append(r)
    group_rows=[]
    for c in consumed:
        samples=groups[c['step']];assert len(samples)==4
        states=[];new_rewards=[]
        for i,s in enumerate(samples):
            v=s['verdict']; correct=bool(s['rewards'][0])
            if (c['step'],i) in recovered:
                v=recovered[c['step'],i]['audit_verdict'];correct=bool(v['correct'])
            known=v['status'] in ('correct','incorrect','extraction_failed')
            states.append(known);new_rewards.append(float(correct)+.1*s['rewards'][1])
        reliable=all(states) and all(s['eos'] for s in samples)
        newly_recovered=c['dropped_group'] and reliable
        online_k=int(sum(s['rewards'][0] for s in samples))
        new_k=sum(bool(recovered[c['step'],i]['audit_verdict']['correct']) if (c['step'],i) in recovered
                  else bool(s['rewards'][0]) for i,s in enumerate(samples))
        new_sd=statistics.stdev(new_rewards)
        row=selected[c['row_id']]
        group_rows.append(dict(step=c['step'],row_id=c['row_id'],subject=row['type'],level=row['level'],difficulty=row['difficulty'],
                               originally_dropped=c['dropped_group'],original_online_k=online_k,
                               original_correctness_contrast=not c['dropped_group'] and 0<online_k<4,
                               audit_all_answers_confirmed=all(states),eos_ok=all(s['eos'] for s in samples),
                               counterfactual_active=reliable,counterfactual_k=new_k,
                               counterfactual_correctness_contrast=reliable and 0<new_k<4,
                               counterfactual_format_only=reliable and new_k in (0,4) and new_sd>1e-8,
                               counterfactual_total_equal=reliable and new_sd<1e-8,newly_recovered=newly_recovered,
                               active_tokens=c['active_tokens']))
    active=[g for g in group_rows if not g['originally_dropped']]
    mixed=[g for g in group_rows if g['original_correctness_contrast']]
    newly=[g for g in group_rows if g['newly_recovered']]
    audit_counts=dict(collections.Counter(r['audit_verdict']['status'] for r in verdicts))
    coverage={}
    test_sft=read(REMOTE/'test_direct367_v1/sft/answers.jsonl')
    test_grpo=read(REMOTE/'test_after_grpo_direct367_v1/grpo/answers.jsonl')
    for subject in sorted({r['type'] for r in chosen}):
        aa=[r for r in test_sft if r['type']==subject]; bb=[r for r in test_grpo if r['type']==subject]
        coverage[subject]=dict(selected=sum(r['type']==subject for r in chosen),
                               active=sum(g['subject']==subject for g in active),
                               correctness_contrast=sum(g['subject']==subject for g in mixed),
                               contrast_tokens=sum(g['active_tokens'] for g in mixed if g['subject']==subject),
                               test_questions=len(aa),sft_test_correct=sum(r['correct'] for r in aa),
                               grpo_test_correct=sum(r['correct'] for r in bb))
    result=dict(scope='Offline counterfactual grading feasibility only; no new training, no changed frozen test scores',
                reviewed_failures=len(verdicts),clear_candidate_decisions=sum(r['decision']=='clear_final_answer' for r in verdicts),
                audited_answer_statuses=audit_counts,original_affected_groups=128,
                newly_recoverable_groups=len(newly),newly_recoverable_correctness_contrast=sum(g['counterfactual_correctness_contrast'] for g in newly),
                newly_recoverable_format_only=sum(g['counterfactual_format_only'] for g in newly),
                newly_recoverable_total_equal=sum(g['counterfactual_total_equal'] for g in newly),
                remaining_dropped_groups=sum(not g['counterfactual_active'] for g in group_rows),
                counterfactual_active_groups=sum(g['counterfactual_active'] for g in group_rows),
                counterfactual_correctness_contrast_groups=sum(g['counterfactual_correctness_contrast'] for g in group_rows),
                original_practice=dict(selected_unique_questions=len(chosen),generated_unique_questions=len(consumed),
                                       generation_rounds_per_question=1,samples_per_question=4,
                                       correctness_contrast_questions=len(mixed),active_questions=len(active),
                                       active_completion_tokens=sum(g['active_tokens'] for g in active),
                                       correctness_contrast_completion_tokens=sum(g['active_tokens'] for g in mixed)),
                by_subject=coverage,
                source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
                               [OUT/'packets.jsonl',OUT/'decisions.jsonl',OUT/'verified_candidates.jsonl',
                                REMOTE/'grpo_direct367_v1/train_consumption.jsonl',REMOTE/'grpo_direct367_v1/rollouts.jsonl']})
    assert len(active)==548 and len(mixed)==431
    assert sum(not g['counterfactual_active'] for g in group_rows)==151-len(newly)
    save(OUT/'audit_summary.json',result)
    with (OUT/'group_counterfactual.jsonl').open('w',encoding='utf-8') as f:
        for r in group_rows:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    p=argparse.ArgumentParser();p.add_argument('--grade',action='store_true');p.add_argument('--analyze',action='store_true');a=p.parse_args()
    if a.grade:grade()
    if a.analyze:analyze()
