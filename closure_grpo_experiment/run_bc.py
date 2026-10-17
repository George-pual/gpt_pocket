"""Bounded B/C experiment only; never repeats the user's SFT baseline A."""
import argparse,datetime,fcntl,json,os,subprocess,sys,traceback
from pathlib import Path
from common import file_hash,save_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--storage',default='/root/autodl-tmp/qwen3_hf');p.add_argument('--output',required=True)
    a=p.parse_args();root,out=Path(a.storage),Path(a.output);scripts=Path(__file__).resolve().parent
    if out.exists() and any(out.iterdir()):raise ValueError('Use a fresh output directory')
    out.mkdir(parents=True);(out/'logs').mkdir()
    lock=(root/'closure_grpo_bc.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    adapter=scripts/'inputs/adapter';expected='2b63b1e8917f535e2c876a4b3a4631444351816a55aac98f7e016ead703ba05c'
    assert file_hash(adapter/'adapter_model.safetensors')==expected,'Unexpected SFT adapter'
    state={'status':'running','runner_pid':os.getpid(),'stage':'setup','completed_stages':[],
        'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'initial_adapter_sha256':expected,'experiments':['B','C'],'steps_per_arm':10,
        'generation_batch_size':8,'learning_rate':5e-6,'num_generations':4,
        'loss_type':'grpo','length_penalty_weight':.5,'end_bonus_B':0.,'end_bonus_C':.1,
        'max_completion_length':3894,'soft_buffer':512,'scripts_sha256':{x.name:file_hash(x) for x in scripts.glob('*.py')}}
    save_json(out/'status.json',state)
    def stage(name,command,allow_oom=False):
        state.update(stage=name,stage_started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        save_json(out/'status.json',state);save_json(out/'logs'/(name+'.command.json'),command)
        print('START',name,flush=True)
        with (out/'logs'/(name+'.log')).open('w') as log:
            process=subprocess.Popen(command,cwd=scripts,stdout=log,stderr=subprocess.STDOUT)
            state['child_pid']=process.pid;save_json(out/'status.json',state);code=process.wait()
        if code:
            text=(out/'logs'/(name+'.log')).read_text(errors='replace')
            if allow_oom and ('CUDA out of memory' in text or 'torch.OutOfMemoryError' in text):
                state.setdefault('oom_preflights',[]).append(name);save_json(out/'status.json',state);return False
            raise RuntimeError(f'{name} failed (exit {code}); inspect {name}.log')
        state['completed_stages'].append(name);save_json(out/'status.json',state);print('DONE',name,flush=True);return True
    def train_args(dest,bonus,micro):
        return [sys.executable,'-u',str(scripts/'train_closure_grpo.py'),'--model',str(root/'model'),
            '--data',str(out/'data/train.jsonl'),'--init-adapter',str(adapter),'--output',str(dest),
            '--max-steps','10','--microbatch',str(micro),'--gradient-accumulation',str(4//micro),
            '--generation-batch-size','8','--dense-rollout','--reward-mode','math-verify',
            '--length-penalty-weight','.5','--soft-buffer','512','--end-bonus-weight',str(bonus)]
    try:
        # Refuse to compete with another active GPU job.
        gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
        if gpu:raise RuntimeError('GPU has an active compute process; refusing concurrent training')
        stage('reward_tests',[sys.executable,'-u',str(scripts/'test_closure_rewards.py')])
        stage('prepare',[sys.executable,'-u',str(scripts/'prepare_bc.py'),'--storage',str(root),'--output',str(out/'data')])
        micro=None
        for candidate in [4,2]:
            ok=stage('preflight_m'+str(candidate),train_args(out/('preflight_m'+str(candidate)),.1,candidate)+[
                '--preflight-only','--preflight-rollouts',str(out/'data/preflight_rollouts.jsonl'),
                '--preflight-data',str(root/'prepared/grpo_article_pilot10.jsonl')],allow_oom=True)
            if ok:micro=candidate;break
        if micro is None:raise RuntimeError('No microbatch passed full-length backward validation')
        state.update(microbatch=micro,gradient_accumulation=4//micro);save_json(out/'status.json',state)
        for name,bonus in [('B',0.),('C',.1)]:
            stage('train_'+name,train_args(out/name,bonus,micro))
            result=json.loads((out/name/'completion_summary.json').read_text())
            assert result['status']=='complete' and result['steps']==10
            weight=file_hash(out/name/'adapter/adapter_model.safetensors')
            assert weight==file_hash(out/name/'checkpoint-10/adapter_model.safetensors')
            state[name+'_adapter_sha256']=weight;save_json(out/'status.json',state)
        for name in ['B','C']:
            stage('eval_'+name,[sys.executable,'-u',str(scripts/'eval_bc.py'),'--model',str(root/'model'),
                '--adapter',str(out/name/'adapter'),'--data',str(out/'data/eval.jsonl'),'--output',str(out/('eval_'+name))])
        results={n:json.loads((out/('eval_'+n)/'summary.json').read_text()) for n in ['B','C']}
        assert results['B']['data_sha256']==results['C']['data_sha256']
        assert file_hash(adapter/'adapter_model.safetensors')==expected
        results.update(status='complete',initial_adapter_unchanged=True,
            scope='B/C only; ten-step pilot and eight held-out within-source questions, no benchmark improvement claim')
        save_json(out/'comparison.json',results)
        state.update(status='complete',stage='complete',finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        save_json(out/'status.json',state);os.sync();print(json.dumps(results),flush=True)
    except Exception as e:
        state.update(status='failed',error=str(e));save_json(out/'status.json',state);traceback.print_exc();raise


if __name__=='__main__':main()
