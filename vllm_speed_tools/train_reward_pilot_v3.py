"""Paired single-GPU short-run experiment pinned to TRL 0.23.1."""
import argparse,json,time,importlib.metadata
from pathlib import Path
import torch
from datasets import Dataset
from transformers import set_seed
from trl import GRPOConfig,GRPOTrainer
from common import tokenizer,load_model,read_jsonl,save_json,file_hash,csv_callback,runtime_manifest
from safe_vllm_trainer import SafeVLLMTrainer
from pilot_rewards_v3 import grade,format_ok,VERSION

EXPECTED='5461e88ae8dba6a3fb536e2bd93bdd8e6f3098a3234943eaef28bc0196017b9b'

class PilotTrainer(SafeVLLMTrainer):
    def _calculate_rewards(self,inputs,prompts,completions,completion_ids_list):
        # Bypass legacy reward logging, which uses the old verifier.
        values=GRPOTrainer._calculate_rewards(self,inputs,prompts,completions,completion_ids_list)
        self._pilot_values=values.detach().clone()
        ended=[self.processing_class.eos_token_id in ids for ids in completion_ids_list]
        self._pilot_ended=ended
        mode='train' if self.model.training else 'eval'
        self._metrics[mode]['pilot/correct_rate'].append(values[:,0].mean().item())
        self._metrics[mode]['pilot/format_eos_rate'].append(sum(format_ok(c,e) for c,e in zip(completions,ended))/len(ended))
        self._metrics[mode]['pilot/eos_rate'].append(sum(ended)/len(ended))
        with (Path(self.args.output_dir)/'rollouts.jsonl').open('a',encoding='utf-8') as f:
            for r,c,ids,v,e in zip(inputs,completions,completion_ids_list,values.tolist(),ended):
                f.write(json.dumps(dict(step=self.state.global_step,row_id=r['row_id'],source=r['source'],gold=r['answer'],completion=c,tokens=len(ids),ended=e,truncated=not e,rewards=v,verdict=grade(c,r['answer'],r['source'])),ensure_ascii=False)+'\n')
        return values

    def _generate_and_score_completions(self,inputs):
        batch=super()._generate_and_score_completions(inputs)
        if self.accelerator.num_processes!=1: raise RuntimeError('Pilot is single GPU only')
        # Drop the ENTIRE group when any sample truncates, so valid siblings
        # cannot receive advantages based on the truncated sample's zero score.
        bad=torch.tensor([not e for e in self._pilot_ended],device=batch['advantages'].device)
        bad=bad.view(-1,4).any(1).repeat_interleave(4)
        batch['completion_mask'][bad]=0
        batch['advantages'][bad]=0
        batch['num_items_in_batch']=batch['completion_mask'].sum()
        if not batch['completion_mask'].any(): raise RuntimeError('Entire batch truncated; stop without update')
        mode='train' if self.model.training else 'eval'
        self._metrics[mode]['pilot/dropped_group_fraction'].append(bad.float().mean().item())
        if not torch.isfinite(batch['advantages']).all(): raise RuntimeError('Nonfinite advantages')
        return batch

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--adapter',required=True)
    p.add_argument('--data-dir',required=True);p.add_argument('--output',required=True);p.add_argument('--format-weight',type=float,choices=[0.,.1],required=True)
    p.add_argument('--steps',type=int,default=32);p.add_argument('--vllm-memory',type=float,default=.40)
    a=p.parse_args();out=Path(a.output)
    assert importlib.metadata.version('trl')=='0.23.1'
    assert file_hash(Path(a.adapter)/'adapter_model.safetensors')==EXPECTED
    if out.exists() and any(out.iterdir()): raise RuntimeError('Refuse overwrite')
    out.mkdir(parents=True,exist_ok=True);set_seed(3407)
    data=Path(a.data_dir);candidates=read_jsonl(data/'train_candidates.jsonl')
    scores={r['row_id']:r for r in read_jsonl(data/'train_candidate_scores.jsonl')}
    wanted={'gsm8k':48,'Level 2':12,'Level 3':4};rows=[];audit=[]
    for r in candidates:
        bucket='gsm8k' if r['source']=='gsm8k' else r['level']
        if wanted[bucket]==0: continue
        verdicts=[grade(s['completion'],r['answer'],r['source']) for s in scores[r['row_id']]['samples']]
        k=sum(v['correct'] for v in verdicts)
        audit.append(dict(row_id=r['row_id'],old_correct=scores[r['row_id']]['correct_count'],new_correct=k))
        if 0<k<4: rows.append(r);wanted[bucket]-=1
    if any(wanted.values()): raise RuntimeError('Not enough regraded mixed questions: '+str(wanted))
    assert len(rows)==64
    assert not {r['row_id'] for r in rows}&{r['row_id'] for r in read_jsonl(data/'validation.jsonl')}
    save_json(out/'selected_train_ids.json',[r['row_id'] for r in rows]);save_json(out/'regrading_audit.json',audit)
    t=tokenizer(a.adapter);t.padding_side='left'
    prompt_lengths=[len(t.apply_chat_template(r['prompt'],tokenize=True,add_generation_prompt=True)) for r in rows]
    if max(prompt_lengths)>1024: raise RuntimeError('Prompt would truncate')
    def correctness(completions,answer,source,**kwargs):
        result=[grade(c,g,s) for c,g,s in zip(completions,answer,source)]
        if any(v['status'].startswith('gold_') for v in result): raise RuntimeError('Invalid gold')
        return [float(v['correct']) for v in result]
    def complete_format_eos(completions,completion_ids,**kwargs):
        return [float(format_ok(c,t.eos_token_id in ids)) for c,ids in zip(completions,completion_ids)]
    model,dtype=load_model(a.model,a.adapter,training=True)
    ref,_=load_model(a.model,a.adapter,training=True);ref.requires_grad_(False);ref.eval();ref.gradient_checkpointing_disable()
    for m in [model,ref]:m.generation_config.eos_token_id=t.eos_token_id;m.generation_config.pad_token_id=t.pad_token_id
    config=GRPOConfig(output_dir=str(out),max_steps=a.steps,num_train_epochs=1,
        per_device_train_batch_size=1,gradient_accumulation_steps=8,generation_batch_size=8,
        num_generations=4,num_iterations=1,max_prompt_length=1024,max_completion_length=3894,
        learning_rate=5e-6,weight_decay=.01,warmup_steps=2,lr_scheduler_type='linear',optim='adamw_8bit',
        temperature=1.,min_p=0.,top_p=1.,top_k=-1,beta=.04,loss_type='grpo',scale_rewards='group',
        reward_weights=[1.,a.format_weight],use_vllm=True,vllm_mode='colocate',vllm_gpu_memory_utilization=a.vllm_memory,
        vllm_enable_sleep_mode=True,vllm_importance_sampling_correction=True,mask_truncated_completions=True,
        bf16=True,gradient_checkpointing=True,gradient_checkpointing_kwargs={'use_reentrant':False},
        logging_steps=1,save_strategy='no',seed=3407,data_seed=3407,remove_unused_columns=False,report_to=['tensorboard'])
    model.config._name_or_path=a.model
    trainer=PilotTrainer(model=model,args=config,processing_class=t,reward_funcs=[correctness,complete_format_eos],train_dataset=Dataset.from_list(rows),callbacks=[csv_callback(out)])
    trainer.single_pass=True
    from trl.trainer.grpo_trainer import prepare_peft_model
    ref=prepare_peft_model(ref,None,config);ref.requires_grad_(False);ref.eval();ref.gradient_checkpointing_disable()
    trainer.ref_model=trainer.accelerator.prepare_model(ref,evaluation_mode=True)
    model.eval();probe=t('Check reference integrity: 1 + 1 =',return_tensors='pt').to('cuda')
    with torch.no_grad(),torch.autocast('cuda',dtype=dtype):
        x=model(**probe).logits.float();y=ref(**probe).logits.float()
    diff=(x-y).abs().max().item()
    if not torch.allclose(x,y,atol=1e-5,rtol=1e-5):raise RuntimeError('Initial SFT reference mismatch')
    del x,y,probe
    initial={n:p.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad}
    model.train();model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    runtime_manifest(a,out,data/'train_candidates.jsonl');save_json(out/'grpo_config.json',config.to_dict())
    save_json(out/'experiment.json',dict(reward_version=VERSION,format_weight=a.format_weight,train_questions=64,internal_validation_questions=100,reference_is_sft_v3=True,initial_reference_logit_difference=diff,truncation_policy='Whole group removed from policy and KL loss when any completion has no EOS; entire invalid batch aborts',no_length_or_repetition_reward=True))
    print('PILOT_REFERENCE_VERIFIED_START',a.format_weight,flush=True);started=time.time()
    result=trainer.train()
    delta=max(float((p.detach().cpu().float()-initial[n].float()).abs().max()) for n,p in model.named_parameters() if n in initial)
    if not delta>0:raise RuntimeError('No parameter update')
    trainer.save_model(str(out/'adapter'));t.save_pretrained(out/'adapter')
    save_json(out/'trainer_log.json',trainer.state.log_history)
    save_json(out/'completion_summary.json',dict(status='complete',steps=trainer.state.global_step,runtime_seconds=time.time()-started,parameter_max_delta=delta,original_adapter_sha256=file_hash(Path(a.adapter)/'adapter_model.safetensors'),metrics=result.metrics))
    assert file_hash(Path(a.adapter)/'adapter_model.safetensors')==EXPECTED
    print('PILOT_COMPLETE',a.format_weight,flush=True)
    trainer.llm.llm_engine.engine_core.shutdown()
    from vllm.distributed.parallel_state import destroy_model_parallel
    destroy_model_parallel()
    if torch.distributed.is_initialized():torch.distributed.destroy_process_group()
    import os,sys
    os.sync();sys.stdout.flush();os._exit(0)

if __name__=='__main__':main()
