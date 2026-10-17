"""Conservative diagnostic gate; four-step probes cannot prove equal quality."""
import argparse,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
stats={}
for n in (8,16,32,64):
    d=a.root/f'probe_gen{n}'
    s=json.loads((d/'completion_summary.json').read_text())
    logs=[r for r in json.loads((d/'trainer_log.json').read_text()) if 'loss' in r]
    rollouts=[json.loads(x) for x in (d/'rollouts.jsonl').read_text().splitlines()]
    steps=8 if n==64 else 4
    assert s['steps']==steps and len(rollouts)==steps*8 and len({r['row_id'] for r in rollouts})==steps*2
    stats[n]={'runtime_seconds':s['runtime_seconds'],'trainer_runtime_seconds':s['metrics']['train_runtime'],'seconds_per_update':s['metrics']['train_runtime']/steps,'max_kl':max(r['kl'] for r in logs),'max_mean_logp_difference':max(r['sampling/sampling_logp_difference/mean'] for r in logs if 'sampling/sampling_logp_difference/mean' in r),'tokens':sum(r['tokens'] for r in rollouts),'truncated':sum(r['truncated'] for r in rollouts)}
eligible=[n for n in (8,16,32,64) if stats[n]['max_kl']<.02 and stats[n]['max_mean_logp_difference']<=stats[8]['max_mean_logp_difference']+.03 and stats[n]['truncated']==0]
if not eligible: raise RuntimeError('No probe passes basic stability gate')
chosen=min(eligible,key=lambda n:stats[n]['seconds_per_update'])
report={'probes':stats,'chosen_generation_batch':chosen,'selection_note':'Fastest measured seconds/update passing preliminary KL/logp/truncation gate.8/16/32 use8questions,64 uses16questions including those8; answer lengths and trajectories differ, so timings are approximate.16/32/64 span2/4/8 optimizer updates.4-8step probes cannot establish equal final quality; track full-run metrics and heldout checkpoints.','target_seconds_per_step':10800/1024}
(a.root/'generation_comparison.json').write_text(json.dumps(report,indent=2))
(a.root/'chosen_generation_batch.txt').write_text(str(chosen)+'\n')
print(json.dumps(report,indent=2))
