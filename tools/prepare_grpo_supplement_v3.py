"""CPU-only recalibration and a disjoint GRPO supplement; never starts training."""
import argparse
import collections
import hashlib
import json
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path

from math_verification_v3_frozen import SCHEMA, extract_final_answer, _grade_pair, require_math_verify
from prepare_math_self600 import canonical
from grpo_outcome_policy import summarize

ROOT=Path(__file__).resolve().parents[1]/'datasets/math/base_self600_20261015'
REMOTE=ROOT/'remote_results'
OUT=ROOT/'grpo_supplement_v3'


def read(p):
    return [json.loads(s) for s in Path(p).read_text(encoding='utf-8').splitlines() if s.strip()]


def save(p,obj):
    Path(p).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write_rows(p,rows):
    with Path(p).open('w',encoding='utf-8') as f:
        for r in rows:f.write(json.dumps(r,ensure_ascii=False)+'\n')


def initialize():
    import logging
    logging.getLogger('math_verify.parser').setLevel(logging.ERROR)
    logging.getLogger('math_verify.grader').setLevel(logging.ERROR)
    assert require_math_verify()=='0.8.0'


def worker(job):
    # Same v3 verifier body as verify_final_answer. The parent bounds each
    # isolated persistent worker task, avoiding per-answer Windows spawn cost.
    start=time.monotonic(); prediction,source,error=extract_final_answer(job['completion'])
    verdict=dict(schema=SCHEMA,answer_source=source,prediction=prediction,correct=False,
                 gold_parsed=False,prediction_parsed=False,status='extraction_failed',error=error)
    if not error:verdict.update(_grade_pair(prediction,str(job['gold']),None,job['question']))
    return dict(key=job['key'],verdict=verdict,worker_seconds=time.monotonic()-start)


def grade_jobs(jobs):
    OUT.mkdir(exist_ok=True)
    cfg=dict(schema=SCHEMA,scorer_sha256=sha(Path(__file__).with_name('math_verification_v3_frozen.py')),
             data_sha256=sha(REMOTE/'grpo_reserved.jsonl'),
             original_screen_manifest_sha256=sha(REMOTE/'grpo_screen_direct367_v1/manifest.json'),
             jobs=len(jobs),workers=4,task_timeout_seconds=5)
    manifest=OUT/'rescore_manifest.json'
    if manifest.exists():assert json.loads(manifest.read_text(encoding='utf-8'))==cfg
    else:save(manifest,cfg)
    cache=OUT/'verification_cache.jsonl'; done={r['key']:r for r in read(cache)} if cache.exists() else {}
    assert set(done)<={j['key'] for j in jobs}
    pending=collections.deque(j for j in jobs if j['key'] not in done)
    context=multiprocessing.get_context('spawn'); executor=None; running={}; last_notice=time.monotonic()
    def new_executor():return ProcessPoolExecutor(max_workers=4,mp_context=context,initializer=initialize)
    def stop_executor(ex):
        for process in list(ex._processes.values()):
            if process.is_alive():process.terminate()
        ex.shutdown(wait=True,cancel_futures=True)
    with cache.open('a',encoding='utf-8') as sink:
        try:
            executor=new_executor()
            while pending or running:
                while pending and len(running)<4:
                    job=pending.popleft();running[executor.submit(worker,job)]=(job,time.monotonic())
                completed,_=wait(running,timeout=.1,return_when=FIRST_COMPLETED)
                for future in completed:
                    job,_=running.pop(future)
                    result=future.result();assert result['key']==job['key']
                    done[job['key']]=result;sink.write(json.dumps(result,ensure_ascii=False)+'\n');sink.flush()
                expired=[f for f,(_,start) in running.items() if time.monotonic()-start>5]
                if expired:
                    for f in expired:
                        job,_=running.pop(f)
                        result=dict(key=job['key'],verdict=dict(schema=SCHEMA,status='verification_timeout',correct=False,
                                    gold_parsed=False,prediction_parsed=False,error='isolated_task_timeout'),worker_seconds=None)
                        done[job['key']]=result;sink.write(json.dumps(result,ensure_ascii=False)+'\n');sink.flush()
                    interrupted=[job for job,_ in running.values()];running.clear()
                    stop_executor(executor);executor=new_executor();pending.extendleft(reversed(interrupted))
                if len(done)%500==0 or time.monotonic()-last_notice>40:
                    print('CPU_RESCORE',len(done),'/',len(jobs),flush=True)
                    save(OUT/'progress.json',dict(stage='cpu_rescore',completed=len(done),total=len(jobs)))
                    last_notice=time.monotonic()
        finally:
            if executor:executor.shutdown(wait=True,cancel_futures=True)
    assert len(done)==len(jobs)
    return done


def allocate(total,weights,capacity):
    """Deterministic bounded proportional allocation by sequential deficits."""
    assert total<=sum(capacity.values())
    result={k:0 for k in capacity}
    for _ in range(total):
        available=[k for k in result if result[k]<capacity[k]]
        key=max(available,key=lambda k:(weights[k]/(result[k]+1),k))
        result[key]+=1
    return result


def main(target):
    rows=read(REMOTE/'grpo_reserved.jsonl'); by_id={r['row_id']:r for r in rows}
    files=sorted((REMOTE/'grpo_screen_direct367_v1/sft/groups').glob('*.json'))
    groups=[json.loads(p.read_text(encoding='utf-8')) for p in files]
    assert len(groups)==len(rows)==2138 and len(by_id)==2138
    original_manifest=json.loads((REMOTE/'all_results_manifest.json').read_text(encoding='utf-8'))
    assert all(sha(p)==original_manifest[p.relative_to(REMOTE).as_posix()] for p in files)
    jobs=[]
    for i,g in enumerate(groups):
        r=by_id[g['row_id']]
        jobs.append(dict(key=f'{i}:gold',completion='<SOLUTION>'+r['answer']+'</SOLUTION>',gold=r['answer'],question=r['problem']))
        for s in g['samples']:
            jobs.append(dict(key=f"{i}:{s['sample_index']}",completion=s['completion'],gold=r['answer'],question=r['problem']))
    results=grade_jobs(jobs);scores=[];proof=[]
    for i,g in enumerate(groups):
        samples=[]
        for s in g['samples']:
            verdict=results[f"{i}:{s['sample_index']}"]['verdict']
            samples.append(dict(**{k:v for k,v in s.items() if k not in ('verdict','correct','completion','token_ids')},
                                correct=verdict['correct'],verdict=verdict))
            proof.append(dict(row_id=g['row_id'],sample_index=s['sample_index'],verdict=verdict))
        gg=dict(**{k:v for k,v in g.items() if k!='samples'},samples=samples)
        scores.append(summarize(gg,results[f'{i}:gold']['verdict']))
    write_rows(OUT/'v3_scores.jsonl',scores);write_rows(OUT/'sample_verdicts.jsonl',proof)
    previous=read(REMOTE/'grpo_selected_direct367_v1/train.jsonl'); previous_ids={r['row_id'] for r in previous}
    audit=json.loads((REMOTE/'grpo_direct367_v1/effective_data_audit.json').read_text(encoding='utf-8'))
    contrasts=read(ROOT/'grpo_log_analysis/failure_audit/group_counterfactual.jsonl')
    counts=collections.Counter(g['subject'] for g in contrasts if g['original_correctness_contrast'])
    factors={s:cell['selected']/counts[s] for s,cell in audit['by_field']['type'].items()}
    pools={d:[s for s in scores if s['eligible'] and s['difficulty']==d] for d in ['easy','medium','hard']}
    feasible=min(len(pools['easy'])//3,len(pools['medium'])//4,len(pools['hard'])//3)*10
    total=min(target//10*10,feasible)
    assert total>699,('insufficient qualified candidates for an expanded dataset',total)
    quotas=dict(easy=total*3//10,medium=total*4//10,hard=total*3//10)
    selected=[]
    rank=lambda r:hashlib.sha256(('20261016:v3supplement:'+r['row_id']).encode()).hexdigest()
    for difficulty,pool in pools.items():
        retained=[s for s in pool if s['row_id'] in previous_ids]
        assert len(retained)<=quotas[difficulty],('old eligible bucket exceeds new target',difficulty)
        selected.extend(retained); available=[s for s in pool if s['row_id'] not in previous_ids]
        strata=collections.defaultdict(list)
        for s in available:strata[s['type']].append(s)
        cap={s:len(v) for s,v in strata.items()};weights={s:cap[s]*factors[s] for s in cap}
        subject_quota=allocate(quotas[difficulty]-len(retained),weights,cap)
        for subject,n in subject_quota.items():
            levels=collections.defaultdict(list)
            for s in strata[subject]:levels[s['level']].append(s)
            level_cap={lv:len(v) for lv,v in levels.items()}
            level_quota=allocate(n,level_cap,level_cap)
            for lv,k in level_quota.items():selected.extend(sorted(levels[lv],key=rank)[:k])
    selected.sort(key=rank);assert len(selected)==len({s['row_id'] for s in selected})==total
    data=[]
    for s in selected:
        r=by_id[s['row_id']]
        data.append(dict(row_id=r['row_id'],problem=r['problem'],answer=r['answer'],source='math',split='train',
                         level=r['level'],type=r['type'],difficulty=s['difficulty'],screen_correct_count=s['k'],
                         screen_grading_schema=SCHEMA,selection_role='retained_previous' if r['row_id'] in previous_ids else 'supplement',
                         prompt=r['prompt']))
    selected_ids={r['row_id'] for r in data};new=[r for r in data if r['row_id'] not in previous_ids]
    removed=[dict(row_id=r['row_id'],v3_score=next(s for s in scores if s['row_id']==r['row_id'])) for r in previous if r['row_id'] not in selected_ids]
    keys={canonical(r['problem']) for r in data};assert len(keys)==total
    forbidden=[REMOTE/'reviewed_direct/train.jsonl',REMOTE/'development.jsonl',REMOTE/'official_test/questions.jsonl']
    for p in forbidden:assert not keys & {canonical(r['problem']) for r in read(p)},('overlap',p)
    assert not {canonical(r['problem']) for r in new} & {canonical(r['problem']) for r in previous}
    write_rows(OUT/'train_expanded.jsonl',data);write_rows(OUT/'supplement.jsonl',new);write_rows(OUT/'previous_not_retained.jsonl',removed)
    report=dict(status='complete',requested_total=target,expanded_total=total,retained_previous=total-len(new),
                supplemental_new_questions=len(new),previous_not_retained=len(removed),
                v3_eligible_pool=sum(len(v) for v in pools.values()),available_by_difficulty={d:len(v) for d,v in pools.items()},
                actual_difficulty=dict(collections.Counter(r['difficulty'] for r in data)),
                supplement_difficulty=dict(collections.Counter(r['difficulty'] for r in new)),
                expanded_subjects=dict(collections.Counter(r['type'] for r in data)),
                supplement_subjects=dict(collections.Counter(r['type'] for r in new)),
                compensation_factors=factors,weight_basis='Available candidate counts multiplied by previous selected/actual-correctness-contrast count, within each difficulty; test scores never used',
                math_levels=dict(collections.Counter(r['level'] for r in data)),
                no_exact_sft_development_test_overlap=True,no_new_question_repeats_previous_699=True,
                calibration_model='Frozen SFT step64; cached G4 answers, not current GRPO model calibration',
                difficulty_scope='G4 estimate under SFT using automatic scorer v3; not a guarantee of future online mixed groups',
                grader_sha256=sha(Path(__file__).with_name('math_verification_v3_frozen.py')),
                input_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in [REMOTE/'grpo_reserved.jsonl',REMOTE/'grpo_selected_direct367_v1/train.jsonl',*forbidden]},
                train_sha256=sha(OUT/'train_expanded.jsonl'),supplement_sha256=sha(OUT/'supplement.jsonl'),
                server_started=False,training_started=False)
    save(OUT/'selection_report.json',report);save(OUT/'progress.json',dict(status='complete',expanded=total,supplement=len(new)))
    print('SUPPLEMENT_COMPLETE',json.dumps(report,ensure_ascii=False),flush=True)


if __name__=='__main__':
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    ap=argparse.ArgumentParser();ap.add_argument('--target',type=int,default=1200);args=ap.parse_args()
    main(args.target)
