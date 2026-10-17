"""Audit both completed GRPO phases without changing training artifacts."""
import csv,hashlib,json,math,re,statistics
from collections import Counter,defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'datasets/grpo_candidates_gsm8k_math_l2_l3/training_audit_1536'
RUNS=[('E1_sft_v3_half_math1024_fast_v2',0,1024),('E1_sft_v3_expanded_continue1536_v1',1024,512)]
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def lines(p):return [json.loads(s) for s in p.read_bytes().splitlines() if s.strip()]
def nonfinite(value,path=''):
    if isinstance(value,float) and not math.isfinite(value):return [path]
    if isinstance(value,dict):return [p for k,v in value.items() for p in nonfinite(v,path+'/'+str(k))]
    if isinstance(value,list):return [p for i,v in enumerate(value) for p in nonfinite(v,path+'/'+str(i))]
    return []
combined=[];stats={};seen=set();reward_examples=[];hashes={}
for name,base,steps in RUNS:
    r=OUT/name;b=r/'B_components';logs=read(b/'trainer_log.json');updates=[x for x in logs if 'loss' in x]
    assert [x['step'] for x in updates]==list(range(1,steps+1))
    done=read(b/'completion_summary.json');assert done['status']=='complete' and done['steps']==steps
    roll=lines(b/'rollouts.jsonl');assert len(roll)==steps*8
    groups=defaultdict(list)
    for x in roll:groups[x['row_id']].append(x)
    assert all(len(g)==4 for g in groups.values())
    assert not seen&groups.keys();seen.update(groups)
    assert not nonfinite(logs) and not nonfinite(roll)
    assert all(x['rewards'][0]==float(x['verdict']['correct']) for x in roll)
    assert all(0<=x['rewards'][1]<=1 for x in roll)
    sync=lines(b/'vllm_sync.jsonl');assert len(sync)==steps//4
    assert [x['step'] for x in sync]==list(range(0,steps,4))
    assert all(x['frozen_weights_unchanged'] and x['tensors']==398 for x in sync)
    cfg=read(b/'grpo_config.json')
    key_stats={}
    for k in ['loss','grad_norm','kl','entropy','learning_rate','clip_ratio/region_mean','pilot/correct_rate','pilot/format_eos_rate','pilot/eos_rate','pilot/dropped_group_fraction','frac_reward_zero_std','sampling/importance_sampling_ratio/mean','sampling/sampling_logp_difference/mean','generation_scoring_seconds']:
        vs=[x[k] for x in updates if k in x];window=min(64,len(vs))
        key_stats[k]=dict(count=len(vs),minimum=min(vs),maximum=max(vs),mean=statistics.mean(vs),first_window_mean=statistics.mean(vs[:window]),last_window_mean=statistics.mean(vs[-window:]),window_records=window)
    text=(r/'B_components.log').read_text(encoding='utf-8',errors='replace')
    errors=[x[:300] for x in text.splitlines() if re.search(r'Traceback|CUDA out of memory|(^|\s)ERROR\s|Nonfinite|No parameter update',x,re.I)]
    assert not errors,(name,errors)
    source_stats={s:dict(answers=len(xs),correct=sum(x['verdict']['correct'] for x in xs),truncated=sum(x['truncated'] for x in xs),statuses=dict(Counter(x['verdict']['status'] for x in xs))) for s in ('gsm8k','math') if (xs:=[x for x in roll if x['source']==s])}
    stats[name]=dict(base_step=base,updates=steps,runtime_seconds=done['runtime_seconds'],unique_questions=len(groups),answers=len(roll),truncated_answers=sum(x['truncated'] for x in roll),dropped_groups=sum(any(x['truncated'] for x in g) for g in groups.values()),dropped_group_rate=sum(any(x['truncated'] for x in g) for g in groups.values())/len(groups),nonfinite_values=0,fatal_log_matches=errors,weight_sync_records=len(sync),source_stats=source_stats,metrics=key_stats,grad_spikes=[dict(cumulative_step=base+x['step'],grad_norm=x['grad_norm'],kl=x['kl']) for x in updates if x['grad_norm']>2],kl_spikes=[dict(cumulative_step=base+x['step'],kl=x['kl']) for x in updates if x['kl']>.1],max_grad_norm=cfg['max_grad_norm'],beta=cfg['beta'],optimizer_reset=done.get('optimizer_reset',False),parameter_max_delta=done['parameter_max_delta'])
    for x in updates:combined.append(dict(phase=name,cumulative_step=base+x['step'],**x))
    # Specific semantic false-negative examples found by manual inspection,
    # distinct from the total number of extraction/parse failures.
    examples={('math:algebra/train-00000-of-00001.parquet:1636','The car travels 70 feet.'),('math:number_theory/train-00000-of-00001.parquet:321','Positive value of $b$ is 5.')}
    for x in roll:
        pred=x['verdict'].get('prediction')
        if (x['row_id'],pred) in examples or (base and x['row_id']=='math:geometry/train-00000-of-00001.parquet:190' and pred and pred.endswith('answer: 6.25')):
            assert not x['verdict']['correct'] and x['rewards'][0]==0
            reward_examples.append(dict(phase=name,cumulative_generation_step=base+x['step'],**x))
    for p in [r/'B_components.log',b/'trainer_log.json',b/'metrics.csv',b/'rollouts.jsonl',b/'completion_summary.json',b/'grpo_config.json']:
        hashes[str(p.relative_to(OUT))]=hashlib.sha256(p.read_bytes()).hexdigest()
assert len(combined)==1536 and len(seen)==3072
keys=list(dict.fromkeys(k for x in combined for k in x))
with (OUT/'combined_metrics_0_1536.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(combined)
(OUT/'audit_summary.json').write_text(json.dumps(dict(runs=stats,total_updates=1536,unique_questions=3072,total_answers=12288,artifact_sha256=hashes,confirmed_zero_reward_examples=len(reward_examples)),ensure_ascii=False,indent=2),encoding='utf-8')
(OUT/'confirmed_reward_false_negative_examples.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in reward_examples),encoding='utf-8')
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(12,7),layout='constrained')
    for ax,key,label in zip(axes.flat,['loss','grad_norm','kl','entropy'],['GRPO loss (signed surrogate)','Gradient norm before clipping','KL to original SFT reference','Token entropy']):
        for name,base,_ in RUNS:
            xs=[x for x in combined if x['phase']==name];xx=[x['cumulative_step'] for x in xs];yy=[x[key] for x in xs]
            ax.plot(xx,yy,alpha=.23,lw=.6,color='#4776b4')
            means=[statistics.mean(yy[max(0,i-31):i+1]) for i in range(len(yy))]
            ax.plot(xx,means,lw=1.4,color='#173f74')
        ax.axvline(1024,color='#b34c32',ls='--',lw=1);ax.set_title(label);ax.set_xlabel('Cumulative optimizer update');ax.grid(alpha=.2)
        if key=='grad_norm':ax.set_yscale('log');ax.axhline(1,color='gray',ls=':',lw=1)
    fig.suptitle('GRPO 0–1536: raw values and trailing 32-update mean\nDashed boundary: new data, optimizer reset, lower learning rate')
    fig.savefig(OUT/'training_diagnostics.png',dpi=150);plt.close(fig)
except ImportError:pass
print(json.dumps(dict(updates=len(combined),unique_questions=len(seen),confirmed_reward_examples=len(reward_examples),phase_stats={k:{x:v[x] for x in ('answers','truncated_answers','dropped_groups','runtime_seconds')} for k,v in stats.items()}),indent=2))
