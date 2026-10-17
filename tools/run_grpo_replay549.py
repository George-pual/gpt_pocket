"""Pinned 549-question replay: preflight, extraction pilot, GRPO, official test."""
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime,timezone
import argparse,fcntl,hashlib,json,os,subprocess,sys,time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
PROJECT=Path('/root/autodl-tmp/math_self600_20261015')
MODEL=Path('/root/autodl-tmp/qwen3_hf/model')
ADAPTER=PROJECT/'grpo_direct367_v1/adapter'
def now():return datetime.now(timezone.utc).isoformat()
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def read(p):return [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines() if s.strip()]
def save(p,x):
    t=p.with_suffix('.tmp');t.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');t.replace(p)
def status(stage,**kwargs):save(ROOT/'status.json',dict(stage=stage,pid=os.getpid(),updated_utc=now(),**kwargs))

def preflight():
    plan=json.loads((ROOT/'plan.json').read_text(encoding='utf-8'))
    for name,expected in json.loads((ROOT/'bundle_manifest.json').read_text()).items():assert sha(ROOT/name)==expected,name
    assert sha(ADAPTER/'adapter_model.safetensors')==plan['adapter_sha256']
    for name,expected in plan['base_weight_sha256'].items():assert sha(MODEL/name)==expected,name
    from scorer_runtime_guard import load_verified_scorer,load_verified_online_reward
    scorer,sruntime=load_verified_scorer(plan['grader_sha256'])
    module,rruntime=load_verified_online_reward(plan['reward_sha256']);primary=module.OnlineRewardScorer(scorer)
    primary.preflight()
    rows=read(ROOT/'data/train.jsonl');assert len(rows)==len({r['row_id'] for r in rows})==549
    assert len(rows)%plan['training']['questions_per_generation']==0
    assert sha(ROOT/'data/train.jsonl')==plan['train_sha256']
    checks=primary.validate_references(rows);assert all(v['status']=='correct' for v in checks.values())
    from common import tokenizer
    t=tokenizer(str(MODEL));lengths=[len(t.apply_chat_template(r['prompt'],tokenize=True,add_generation_prompt=True)) for r in rows]
    assert max(lengths)<=1024
    gpu=subprocess.run(['nvidia-smi','--query-gpu=name,memory.total,memory.used','--format=csv,noheader'],check=True,capture_output=True,text=True).stdout.strip()
    report=dict(status='passed',utc=now(),questions=len(rows),max_prompt_tokens=max(lengths),grader=sruntime,reward=rruntime,
        adapter_sha256=plan['adapter_sha256'],gpu=gpu,api_role='extraction only')
    save(ROOT/'preflight.json',report);save(ROOT/'reference_checks.json',checks)
    print('REPLAY_PREFLIGHT_PASSED',json.dumps(report),flush=True)
    return plan,primary

def pilot(primary):
    from api_answer_review import APIAnswerReviewer,MathFirstAPIReward,PROMPT_VERSION
    reviewer=APIAnswerReviewer(ROOT/'configs/api_training.json',ROOT/'pilot_cache')
    rows=read(ROOT/'data/api_extraction_pilot8.jsonl');assert len(rows)==8
    results=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures={pool.submit(reviewer.review,r['problem'],r['completion']):r for r in rows}
        for f in as_completed(futures):
            r=futures[f];reply=f.result()
            class Saved:
                def review(self,*args):return reply
            result=MathFirstAPIReward(primary,Saved()).score_one(r['completion'],r['gold'],r['problem'],r['eos'])
            results.append(dict(row_id=r['row_id'],sample_index=r['sample_index'],expected=r['regraded_status'],
                original_response=r['completion'],api=reply,result=result))
    save(ROOT/'api_extraction_pilot_results.json',results)
    summary=dict(prompt_version=PROMPT_VERSION,samples=8,api_calls=reviewer.calls,
        valid_extractions=sum(r['api']['status']=='extracted' for r in results),
        matched=sum(not r['result']['grading_unconfirmed'] and r['result']['verdict']['status']==r['expected'] for r in results),
        external_fields=['problem','response'],gold_sent=False,api_does_math_grading=False)
    save(ROOT/'api_extraction_pilot_summary.json',summary)
    print('API_EXTRACTION_PILOT',json.dumps(summary),flush=True)
    if summary['matched']!=8:raise RuntimeError('Extraction pilot did not match all eight local reviewed verdicts; training not started')

def main():
    os.umask(0o077)
    ap=argparse.ArgumentParser();ap.add_argument('--preflight-only',action='store_true');args=ap.parse_args()
    lock=(ROOT/'pipeline.lock').open('a')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise RuntimeError('This replay pipeline is already running')
    start=time.time()
    try:
        status('preflight');plan,primary=preflight()
        if args.preflight_only:status('preflight_passed');return
        status('api_extraction_pilot');pilot(primary)
        train=ROOT/'training'
        if train.exists():raise RuntimeError('Fresh training output required; refusing duplicate launch')
        options=plan['training']
        command=[sys.executable,str(ROOT/'tools/train_direct_grpo.py'),'--model',str(MODEL),'--adapter',str(ADAPTER),
            '--data',str(ROOT/'data/train.jsonl'),'--output',str(train),'--expected-grader-sha256',plan['grader_sha256'],
            '--expected-reward-sha256',plan['reward_sha256'],'--api-review-config',str(ROOT/'configs/api_training.json'),
            '--questions-per-generation',str(options['questions_per_generation']),
            '--train-microbatch',str(options['training_microbatch']),
            '--gradient-accumulation',str(options['gradient_accumulation']),
            '--vllm-memory',str(options['vllm_memory'])]
        status('training_running',questions=549,planned_updates=549,started_utc=now())
        with (ROOT/'train.log').open('a') as log:
            p=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=os.environ.copy())
            save(ROOT/'training_launch.json',dict(pid=p.pid,command=command,started_utc=now(),adapter_sha256=plan['adapter_sha256'],data_sha256=plan['train_sha256']))
            code=p.wait()
        if code:raise RuntimeError('GRPO exited with code '+str(code)+'; inspect train.log')
        complete=json.loads((train/'completion_summary.json').read_text());assert complete['status']=='complete' and complete['updates']==549
        status('official_test_running',training=complete,questions=2024)
        command=[sys.executable,str(ROOT/'tools/run_math_stage.py'),'--model',str(MODEL),'--data',str(ROOT/'data/official_test.jsonl'),
            '--output',str(ROOT/'test_after_grpo'),'--arms',json.dumps({'grpo_replay549':str(train/'adapter')}),
            '--batch','64','--expected-grader-sha256',plan['grader_sha256']]
        with (ROOT/'test.log').open('a') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
        test=json.loads((ROOT/'test_after_grpo/comparison.json').read_text())
        assert test['grpo_replay549']['questions']==2024
        assert sha(ADAPTER/'adapter_model.safetensors')==plan['adapter_sha256']
        status('complete_awaiting_retrieval',training=complete,test=test,runtime_seconds=time.time()-start,shutdown=False)
    except BaseException as exc:
        status('failed',error_type=type(exc).__name__,error=str(exc),runtime_seconds=time.time()-start)
        raise

if __name__=='__main__':main()
