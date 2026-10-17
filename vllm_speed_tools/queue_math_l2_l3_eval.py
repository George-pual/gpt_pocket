"""Run MATH test after the paired GSM8K job releases its GPU."""
import json,os,subprocess,time
from pathlib import Path
from pilot_rewards_v3 import grade

ROOT=Path('/root/autodl-tmp/gsm8k_base_sftv3_eval')
def state(status,**kwargs):
    data=dict(status=status,**kwargs)
    (ROOT/'math_queue_status.json').write_text(json.dumps(data,ensure_ascii=False,indent=2))
    print(json.dumps(data,ensure_ascii=False),flush=True)

def main():
    rows=[json.loads(s) for s in (ROOT/'data/math_l2_l3/questions.jsonl').read_text().splitlines()]
    failures=[]
    for r in rows:
        verdict=grade('<SOLUTION>'+r['answer']+'</SOLUTION>',r['answer'],'math')
        if not verdict['correct']:failures.append(dict(row_id=r['row_id'],verdict=verdict))
    (ROOT/'math_gold_grading_check.json').write_text(json.dumps(dict(checked=len(rows),failed=failures),ensure_ascii=False,indent=2))
    assert len(rows)==200 and not failures,failures
    pid=int((ROOT/'eval_vllm.pid').read_text())
    state('waiting_for_gsm8k',gsm8k_pid=pid,math_questions=200)
    while True:
        p=Path(f'/proc/{pid}/stat')
        active=p.exists() and p.read_text().split(') ',1)[1].split()[0] not in {'Z','X'}
        if active:time.sleep(15);continue
        comparison=json.loads((ROOT/'results_vllm/comparison.json').read_text())
        assert set(comparison)=={'base','sft_v3'}
        assert all(v['status']=='complete' and v['questions']==1319 for v in comparison.values())
        gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True)
        if gpu.strip():time.sleep(15);continue
        break
    state('running_math',questions=200)
    env=os.environ.copy();env.update(OMP_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',VLLM_CACHE_ROOT=str(ROOT/'cache/vllm'))
    cmd=['/root/autodl-tmp/vllm_speed_env/bin/python','-u',str(ROOT/'tools/eval_math_l2_l3.py'),
         '--model','/root/autodl-tmp/qwen3_hf/model','--adapter',str(ROOT/'adapter'),
         '--data',str(ROOT/'data/math_l2_l3/questions.jsonl'),'--output',str(ROOT/'results_math_l2_l3'),
         '--dataset-kind','math']
    with (ROOT/'eval_math.log').open('w') as log:
        result=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
    assert result.returncode==0,result.returncode
    comparison=json.loads((ROOT/'results_math_l2_l3/comparison.json').read_text())
    assert all(v['status']=='complete' and v['questions']==200 for v in comparison.values())
    state('complete',questions_per_model=200)

if __name__=='__main__':
    try:main()
    except Exception as exc:
        state('failed_no_further_launch',error=repr(exc))
        raise
