"""Server-side completion guard. No local PC or SSH session required."""
import argparse,hashlib,json,os,signal,subprocess,time
from pathlib import Path

DEFAULT='/root/autodl-tmp/qwen3_hf/runs/E1_sft_v3_half_math1024_fast_v2'

def read(p): return json.loads(p.read_text())
def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def validate(root,train_steps=1024,base_step=0,eval_steps=(256,512,1024),checkpoint_steps=(256,512,1024)):
    assert (root/'stage.txt').read_text().strip()=='complete'
    assert (root/'pipeline.exitcode').read_text().strip()=='0'
    train=read(root/'B_components/completion_summary.json')
    assert train['status']=='complete' and train['steps']==train_steps and train['parameter_max_delta']>0
    if base_step:assert train['cumulative_steps']==base_step+train_steps
    hashes={}
    for step,checkpoint_step in zip(eval_steps,checkpoint_steps):
        folder=root/f'eval_step_{step}'
        summary=read(folder/'summary.json')
        assert summary['status']=='complete' and summary['overall']['questions']==100 and summary['overall']['answers']==400
        assert summary['data_sha256']=='bc62d6a765fb073149ed74773d0e9f30961c7a56fbf753fe7ac70f82514a5762'
        assert summary['sampling']['n']==4
        rows=[json.loads(x) for x in (folder/'answers.jsonl').read_text().splitlines() if x.strip()]
        assert len(rows)==400
        ids={r['row_id'] for r in rows}
        assert len(ids)==100 and len({(r['row_id'],r['sample_index']) for r in rows})==400
        assert all({r['sample_index'] for r in rows if r['row_id']==i}=={0,1,2,3} for i in ids)
        for key in ('correct','format_eos','truncated'):
            assert sum(bool(r[key]) for r in rows)==summary['overall'][key]
        weight=root/f'B_components/checkpoint-{checkpoint_step}/adapter_model.safetensors'
        assert weight.stat().st_size>100_000_000
        hashes[str(weight.relative_to(root))]=digest(weight)
        assert hashes[str(weight.relative_to(root))]==summary['adapter_sha256']
        for p in (folder/'answers.jsonl',folder/'summary.json'):
            hashes[str(p.relative_to(root))]=digest(p)
    final=root/'B_components/adapter/adapter_model.safetensors'
    assert final.stat().st_size>100_000_000
    hashes[str(final.relative_to(root))]=digest(final)
    return hashes

def alive(pid):
    p=Path(f'/proc/{pid}/stat')
    try:return p.read_text().split(') ',1)[1].split()[0] not in {'Z','X'}
    except FileNotFoundError:return False

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(DEFAULT))
    p.add_argument('--pipeline-pid',type=int,default=14753);p.add_argument('--supervisor-pid',type=int,default=120)
    p.add_argument('--train-steps',type=int,default=1024);p.add_argument('--base-step',type=int,default=0)
    p.add_argument('--eval-steps',default='256,512,1024');p.add_argument('--checkpoint-steps',default='256,512,1024')
    p.add_argument('--check-only',action='store_true');a=p.parse_args();root=a.root
    eval_steps=tuple(map(int,a.eval_steps.split(',')));checkpoint_steps=tuple(map(int,a.checkpoint_steps.split(',')))
    assert len(eval_steps)==len(checkpoint_steps)>0
    def record(status,**kw):
        data=dict(status=status,time_unix=time.time(),pipeline_pid=a.pipeline_pid,**kw)
        (root/'auto_shutdown_status.json').write_text(json.dumps(data,ensure_ascii=False,indent=2))
        print(json.dumps(data,ensure_ascii=False),flush=True)
    if a.check_only:
        print(json.dumps(dict(mode='check_only',armed=(root/'auto_shutdown_armed.json').exists(),stage=(root/'stage.txt').read_text().strip(),pipeline_alive=alive(a.pipeline_pid))))
        return
    armed=read(root/'auto_shutdown_armed.json')
    assert armed['authorized'] is True and armed['pipeline_pid']==a.pipeline_pid and armed['root']==str(root)
    record('waiting_for_success')
    while True:
        stage=(root/'stage.txt').read_text().strip()
        if stage=='complete' and (root/'pipeline.exitcode').exists() and (root/'pipeline.exitcode').read_text().strip()=='0':
            try:
                hashes=validate(root,a.train_steps,a.base_step,eval_steps,checkpoint_steps)
                gpu=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],capture_output=True,text=True,check=True)
                if gpu.stdout.strip():
                    record('waiting_for_gpu_cleanup',gpu_pids=gpu.stdout.strip());time.sleep(30);continue
                cmdline=Path(f'/proc/{a.supervisor_pid}/cmdline').read_bytes().replace(b'\x00',b' ').decode()
                assert '/bin/supervisord' in cmdline and '/init/supervisor/supervisor.ini' in cmdline
                (root/'shutdown_artifact_manifest.json').write_text(json.dumps(hashes,indent=2))
                record('validated_shutdown_requested',artifact_hashes=hashes,shutdown_method='AutoDL supervisor SIGTERM; matches platform shutdown script without unrelated Trash deletion')
                os.sync()
                # AutoDL's /usr/bin/shutdown signals this supervisor to stop the
                # instance. Preserve unrelated files by omitting its Trash cleanup.
                os.kill(a.supervisor_pid,signal.SIGTERM)
                return
            except Exception as e:
                record('blocked_no_shutdown',error=repr(e));return
        if not alive(a.pipeline_pid):
            record('pipeline_stopped_without_success_no_shutdown',stage=stage);return
        time.sleep(30)

if __name__=='__main__':main()
