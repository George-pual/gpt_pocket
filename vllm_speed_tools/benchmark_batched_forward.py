"""GPU equivalence and timing on the same saved rollout batch, no updates."""
import argparse,json,time,inspect
from pathlib import Path
from types import SimpleNamespace
from collections import defaultdict
import torch
from trl import GRPOTrainer
from common import tokenizer,load_model,read_jsonl,save_json
from math_verification import completion_text
from batched_pilot_trainer import BatchedPilotTrainer as TrimmedPilotTrainer
from trimmed_pilot_trainer import TrimmedPilotTrainer as OriginalTrimmedPilotTrainer

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
    root=Path('/root/autodl-tmp');modelpath=root/'qwen3_hf/model';adapter=root/'qwen3_hf/runs/E1_sft_v3/adapter'
    rows=read_jsonl(root/'sft_v3_reward_pilot/data/train_candidates.jsonl');byid={r['row_id']:r for r in rows}
    groups=defaultdict(list)
    for r in read_jsonl(root/'qwen3_hf/runs/E1_sft_v3_reward_pilot_v1/B_correctness_format/rollouts.jsonl'):groups[r['step']].append(r)
    candidates=[v for v in groups.values() if all(r['ended'] for r in v) and 500<=max(r['tokens'] for r in v)<=1600]
    batch=max(candidates,key=lambda v:8*max(r['tokens'] for r in v)/sum(r['tokens'] for r in v))
    t=tokenizer(adapter);prefix=[t.apply_chat_template(byid[r['row_id']]['prompt'],tokenize=True,add_generation_prompt=True) for r in batch]
    outputs=[t(completion_text(r['completion']),add_special_tokens=False)['input_ids']+[t.eos_token_id] for r in batch]
    pw=max(map(len,prefix));cw=max(map(len,outputs))
    ids=torch.tensor([[t.pad_token_id]*(pw-len(pr))+pr+co+[t.pad_token_id]*(cw-len(co)) for pr,co in zip(prefix,outputs)],device='cuda')
    mask=torch.tensor([[0]*(pw-len(pr))+[1]*(len(pr)+len(co))+[0]*(cw-len(co)) for pr,co in zip(prefix,outputs)],device='cuda')
    model,dtype=load_model(modelpath,adapter,training=True);model.eval()
    fake=object.__new__(TrimmedPilotTrainer);fake.model=model;fake.temperature=1.;fake.score_batch=2;fake.model_kwarg_keys=list(inspect.signature(model.get_base_model().forward).parameters);fake.args=SimpleNamespace(report_to=[])
    def legacy(x,m,keep):return OriginalTrimmedPilotTrainer._get_per_token_logps_and_entropies(fake,model,x,m,keep,batch_size=1,compute_entropy=True)
    def trimmed(x,m,keep):return fake._get_per_token_logps_and_entropies(model,x,m,keep,batch_size=1,compute_entropy=True)
    results={};saved={}
    # Warm both code paths, then median of two identical-batch measurements.
    with torch.no_grad(),torch.autocast('cuda',dtype=dtype):
        legacy(ids[:1],mask[:1],cw);trimmed(ids[:1],mask[:1],cw)
        for name,fn in [('legacy',legacy),('trimmed',trimmed)]:
            times=[]
            for _ in range(2):
                torch.cuda.synchronize();start=time.perf_counter();lp,en=fn(ids,mask,cw);torch.cuda.synchronize();times.append(time.perf_counter()-start)
            saved[name]=(lp,en);results[name]=dict(seconds_mean=sum(times)/len(times),seconds_runs=times)
    valid=mask[:,pw:].bool();diff=(saved['legacy'][0]-saved['trimmed'][0]).abs()[valid]
    entropy_diff=(saved['legacy'][1]-saved['trimmed'][1]).abs()[valid]
    results.update(dict(batch_answers=len(batch),padded_completion_width=cw,real_completion_positions=int(valid.sum()),padded_completion_positions=int(valid.numel()),max_valid_logp_difference=float(diff.max()),mean_valid_logp_difference=float(diff.mean()),max_valid_entropy_difference=float(entropy_diff.max()),forward_speed_ratio=results['legacy']['seconds_mean']/results['trimmed']['seconds_mean']))
    # Compare gradients from the same short answer with/without trailing padding.
    i=min(range(len(outputs)),key=lambda i:len(outputs[i]));gradients={}
    for name,fn in [('legacy',legacy),('trimmed',trimmed)]:
        model.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=dtype):
            lp,_=fn(ids[[i,i]],mask[[i,i]],cw);cm=mask[[i,i],pw:];loss=-(lp*cm).sum()/cm.sum()
        loss.backward();gradients[name]={n:p.grad.detach().float().cpu().clone() for n,p in model.named_parameters() if p.requires_grad and p.grad is not None}
    va=torch.cat([v.flatten() for v in gradients['legacy'].values()]);vb=torch.cat([gradients['trimmed'][n].flatten() for n in gradients['legacy']])
    def gradient_compare(a,b):
        # Accumulate in FP64; FP32 reductions across ~66M coordinates can
        # noticeably distort the norm and even yield cosine values above 1.
        dot=na=nb=err=0.
        for n in a:
            x=a[n].double().flatten();y=b[n].double().flatten()
            dot+=float(x.dot(y));na+=float(x.dot(x));nb+=float(y.dot(y));err+=float((x-y).square().sum())
        return dict(cosine=dot/(na*nb)**.5,relative_l2=(err/na)**.5)
    results['bf16_gradient_comparison']=gradient_compare(gradients['legacy'],gradients['trimmed'])
    del va,vb,gradients
    # Independently check the algorithm at full FP32 compute. BF16 kernel
    # selection can change with sequence shape; do not mistake this for an
    # alignment bug, and do not accept BF16 drift without an FP32 cross-check.
    torch.backends.cuda.matmul.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    for module in model.modules():
        if hasattr(module,'compute_dtype'):module.compute_dtype=torch.float32
    small=min(cw,256)
    assert len(outputs[i])<=small
    xx=ids[[i,i],:pw+small];mm=mask[[i,i],:pw+small];fp={};gg={}
    for name,fn in [('legacy',legacy),('trimmed',trimmed)]:
        model.zero_grad(set_to_none=True)
        lp,_=fn(xx,mm,small);cm=mm[:,pw:];loss=-(lp*cm).sum()/cm.sum()
        fp[name]=lp.detach().cpu();loss.backward()
        gg[name]={n:p.grad.detach().float().cpu().clone() for n,p in model.named_parameters() if p.requires_grad and p.grad is not None}
    diff32=(fp['legacy']-fp['trimmed']).abs()[mm[:,pw:].bool().cpu()]
    results['fp32_crosscheck']=dict(max_valid_logp_difference=float(diff32.max()),**gradient_compare(gg['legacy'],gg['trimmed']))
    results['optimizer_updated']=False
    check=results['fp32_crosscheck'];bg=results['bf16_gradient_comparison']
    results['passed']=bool(check['max_valid_logp_difference']<.0002 and check['cosine']>.99999 and check['relative_l2']<.001 and results['max_valid_logp_difference']<.3 and results['mean_valid_logp_difference']<.005 and bg['cosine']>.995 and bg['relative_l2']<.08)
    save_json(a.output,results);print(json.dumps(results,indent=2),flush=True)
    if not results['passed']:raise RuntimeError('Padding optimization equivalence failed')

if __name__=='__main__':main()
