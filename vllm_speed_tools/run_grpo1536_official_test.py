"""Official single-answer evaluation on the training server after internal eval stops."""
import hashlib,json,os,signal,subprocess,time
from pathlib import Path
ROOT=Path('/root/autodl-tmp/grpo1536_official_eval')
TRAIN=Path('/root/autodl-tmp/qwen3_hf/runs/E1_sft_v3_expanded_continue1536_v1')
ADAPTER=TRAIN/'B_components/checkpoint-512'
PY='/root/autodl-tmp/vllm_speed_env/bin/python'
HASH='d03d2a790a4a90e673b556a2a0e337a16f8a1e9e3f781d595308ae163550fabb'
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    out=ROOT/'results';assert not out.exists(),'Refuse overwrite';out.mkdir()
    assert digest(ADAPTER/'adapter_model.safetensors')==HASH
    train=json.loads((TRAIN/'B_components/completion_summary.json').read_text())
    assert train['status']=='complete' and train['cumulative_steps']==1536
    env=os.environ.copy();env.update(OMP_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',VLLM_CACHE_ROOT='/root/autodl-tmp/sft_v3_reward_pilot/cache/vllm')
    results={}
    for kind,data in [('gsm8k','gsm8k_test.jsonl'),('math','math_l2_l3_200.jsonl')]:
        (ROOT/'stage.txt').write_text('evaluating_'+kind)
        ref=json.loads((ROOT/'references'/f'{kind}_comparison.json').read_text())
        assert digest(ROOT/'data'/data)==ref['base']['data_sha256']
        cmd=[PY,'-u',str(ROOT/'tools/eval_gsm8k_base_sftv3.py'),'--model','/root/autodl-tmp/qwen3_hf/model','--adapter',str(ADAPTER),'--data',str(ROOT/'data'/data),'--output',str(out/kind),'--dataset-kind',kind,'--arms','grpo1536','--expected-adapter-sha256',HASH]
        with (ROOT/f'{kind}.log').open('w') as log:subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        s=json.loads((out/kind/'grpo1536/summary.json').read_text())
        assert s['status']=='complete'
        for a in ('base','sft_v3'):
            for key in ('questions','data_sha256','prompt_tokens_sha256','sampling'):assert s[key]==ref[a][key],(kind,a,key)
        rows=[json.loads(x) for x in (out/kind/'grpo1536/answers.jsonl').read_text().splitlines() if x.strip()]
        assert len(rows)==s['questions'] and len({x['row_id'] for x in rows})==len(rows)
        results[kind]=dict(ref,grpo1536=s)
        (out/'comparison.json').write_text(json.dumps(results,ensure_ascii=False,indent=2))
        print('DATASET_COMPLETE',kind,json.dumps(s),flush=True)
    (ROOT/'stage.txt').write_text('complete')
    print('ALL_COMPLETE',flush=True)
    # User previously requested server-side shutdown after training and tests.
    # Internal 4-sample evaluation was explicitly cancelled; validate the new
    # official results instead and never shut down following a failed evaluation.
    (ROOT/'auto_shutdown_status.json').write_text(json.dumps({'status':'waiting_for_gpu_cleanup','authorized_by':'prior user request; current official test replaces internal validation'}))
    for _ in range(60):
        p=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],capture_output=True,text=True,check=True)
        if not p.stdout.strip():break
        time.sleep(10)
    else:raise RuntimeError('GPU still busy; leave server running')
    supervisor=883
    cmdline=Path(f'/proc/{supervisor}/cmdline').read_bytes().replace(b'\0',b' ').decode()
    assert '/bin/supervisord' in cmdline and '/init/supervisor/supervisor.ini' in cmdline
    files=[ADAPTER/'adapter_model.safetensors',out/'comparison.json']
    files += [out/k/'grpo1536'/f for k in ('gsm8k','math') for f in ('answers.jsonl','summary.json')]
    (ROOT/'shutdown_artifact_manifest.json').write_text(json.dumps({str(p):digest(p) for p in files},indent=2))
    (ROOT/'auto_shutdown_status.json').write_text(json.dumps({'status':'validated_shutdown_requested','supervisor_pid':supervisor}))
    os.sync();os.kill(supervisor,signal.SIGTERM)
if __name__=='__main__':
    try:main()
    except Exception as exc:
        (ROOT/'failure.json').write_text(json.dumps({'error':repr(exc),'shutdown':False}))
        print('FAILED_NO_SHUTDOWN',repr(exc),flush=True)
        raise
