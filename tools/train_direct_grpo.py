"""One pass over the selected half-pool with fresh online G4 vLLM rollouts.

Policy and frozen reference both start at the newly selected SFT adapter.
Never merge or modify the NF4 training base when synchronizing vLLM.
"""
import argparse,hashlib,importlib.metadata,json,time
from pathlib import Path


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',required=True);ap.add_argument('--adapter',required=True)
    ap.add_argument('--data',required=True);ap.add_argument('--output',required=True)
    ap.add_argument('--vllm-memory',type=float,default=.40);ap.add_argument('--smoke',action='store_true')
    ap.add_argument('--questions-per-generation',type=int,default=1)
    ap.add_argument('--train-microbatch',type=int,default=1)
    ap.add_argument('--gradient-accumulation',type=int,default=4)
    ap.add_argument('--expected-grader-sha256')
    ap.add_argument('--expected-reward-sha256')
    ap.add_argument('--api-review-config',help='Optional DeepSeek answer extraction followed by local Math-Verify; credential stays in environment')
    a=ap.parse_args()
    from grpo_batching import fresh_group_indices,mask_uncertain_groups
    assert a.questions_per_generation>0 and a.train_microbatch>0 and a.gradient_accumulation>0
    assert a.train_microbatch*a.gradient_accumulation==4,'Preserve four responses per optimizer update on average'
    assert (4*a.questions_per_generation)%a.train_microbatch==0
    from scorer_runtime_guard import load_verified_scorer,load_verified_online_reward
    scorer,grader_runtime=load_verified_scorer(a.expected_grader_sha256)
    verify_final_answer,completion_text,require_math_verify,SCHEMA=scorer.verify_final_answer,scorer.completion_text,scorer.require_math_verify,scorer.SCHEMA
    print('GRADER_VERIFIED',json.dumps(grader_runtime),flush=True)
    reward_module,reward_runtime=load_verified_online_reward(a.expected_reward_sha256)
    OnlineRewardScorer,POLICY,FORMAT_WEIGHT=reward_module.OnlineRewardScorer,reward_module.POLICY,reward_module.FORMAT_WEIGHT
    reward_scorer=OnlineRewardScorer(scorer)
    reward_preflight=reward_scorer.preflight()
    rows=[json.loads(s) for s in Path(a.data).read_text(encoding='utf-8').splitlines() if s.strip()]
    assert rows and len(rows)==len({r['row_id'] for r in rows})
    assert len(rows)%a.questions_per_generation==0,'Refuse dropping or duplicating a partial generation batch'
    out=Path(a.output);assert not out.exists(),'Fresh GRPO output required'
    gold_checks=reward_scorer.validate_references(rows)
    if any(v['status']!='correct' or not v['correct'] for v in gold_checks.values()):
        raise RuntimeError('Reference-answer parser self-check failed; refusing model loading')
    out.mkdir(parents=True)
    save(out/'gold_checks.json',gold_checks)
    save(out/'reward_preflight.json',reward_preflight)
    api_reviewer=None
    if a.api_review_config:
        from api_answer_review import APIAnswerReviewer,MathFirstAPIReward
        api_reviewer=APIAnswerReviewer(a.api_review_config,out/'api_review_cache')
        reward_scorer=MathFirstAPIReward(reward_scorer,api_reviewer)
    print('REWARD_AND_REFERENCES_VERIFIED',len(rows),json.dumps(reward_preflight),flush=True)
    assert importlib.metadata.version('trl')=='0.23.1'
    import torch
    import torch.nn.functional as F
    import bitsandbytes as bnb
    from datasets import Dataset
    from transformers import set_seed,TrainerCallback
    from trl import GRPOConfig,GRPOTrainer
    from trl.trainer.grpo_trainer import prepare_peft_model
    from common import tokenizer,load_model
    set_seed(3407)
    t=tokenizer(a.model);t.padding_side='left'
    assert max(len(t.apply_chat_template(r['prompt'],tokenize=True,add_generation_prompt=True)) for r in rows)<=1024
    adapter_sha=sha(Path(a.adapter)/'adapter_model.safetensors')
    base_hashes={p.name:sha(p) for p in Path(a.model).glob('model-*.safetensors')}
    def reward_records(completions,answer,completion_ids,problem):
        return reward_scorer.score_batch(completions,answer,problem,[t.eos_token_id in ids for ids in completion_ids])
    def correctness(completions,answer,completion_ids,problem,**kwargs):
        return [r['correctness_reward'] for r in reward_records(completions,answer,completion_ids,problem)]
    def format_reward(completions,answer,completion_ids,problem,**kwargs):
        return [r['format_reward'] for r in reward_records(completions,answer,completion_ids,problem)]
    def width(mask):
        cols=torch.arange(mask.shape[1],device=mask.device).unsqueeze(0)+1
        return max(1,int((cols*mask.bool()).max().item()))

    class DirectTrainer(GRPOTrainer):
        def _move_model_to_vllm(self):
            start=time.perf_counter();policy=self.accelerator.unwrap_model(self.model);core=policy.get_base_model()
            frozen=[(p,p.data_ptr(),p._version) for p in core.parameters() if not p.requires_grad]
            worker=self.llm.llm_engine.model_executor.driver_worker.model_runner.model
            count=0
            with torch.no_grad():
                for name,module in core.named_modules():
                    if not hasattr(module,'lora_A'):continue
                    base=module.get_base_layer()
                    weight=bnb.functional.dequantize_4bit(base.weight.data,base.weight.quant_state)
                    delta=module.get_delta_weight('default')
                    merged=(weight.float()+delta.float()).to(torch.bfloat16)
                    worker.load_weights([(name+'.weight',merged)]);count+=1
                    del weight,delta,merged
                for name,param in core.named_parameters():
                    if '.lora_' in name or '.base_layer.' in name:continue
                    worker.load_weights([(name,param.detach().to(torch.bfloat16))]);count+=1
            assert all(p.data_ptr()==ptr and p._version==version for p,ptr,version in frozen)
            self.llm.reset_prefix_cache();torch.cuda.empty_cache()
            with (out/'vllm_sync.jsonl').open('a') as f:
                f.write(json.dumps(dict(step=self.state.global_step,tensors=count,seconds=time.perf_counter()-start,frozen_weights_unchanged=True))+'\n')

        def _calculate_rewards(self,inputs,prompts,completions,completion_ids_list):
            self.latest_eos=[t.eos_token_id in ids for ids in completion_ids_list]
            self.latest_reward_records=reward_scorer.score_batch(completions,[r['answer'] for r in inputs],
                [r['problem'] for r in inputs],self.latest_eos)
            self.latest_verdicts=[r['verdict'] for r in self.latest_reward_records]
            rewards=super()._calculate_rewards(inputs,prompts,completions,completion_ids_list)
            assert torch.isfinite(rewards).all()
            expected=torch.tensor([[r['correctness_reward'],r['format_reward']] for r in self.latest_reward_records],
                dtype=rewards.dtype,device=rewards.device)
            assert rewards.shape==expected.shape and torch.allclose(rewards,expected),'Reward callback mismatch'
            self.latest_ids=[r['row_id'] for r in inputs]
            self.latest_rollout_records=[]
            question_sample_counts={}
            with (out/'rollouts.jsonl').open('a',encoding='utf-8') as f:
                for sample_index,(row,c,ids,v,e,record) in enumerate(zip(inputs,completions,completion_ids_list,rewards.tolist(),self.latest_eos,self.latest_reward_records)):
                    text=completion_text(c)
                    question_sample_index=question_sample_counts.get(row['row_id'],0)
                    question_sample_counts[row['row_id']]=question_sample_index+1
                    rollout=dict(step=self.state.global_step,row_id=row['row_id'],sample_index=question_sample_index,batch_sample_index=sample_index,difficulty=row['difficulty'],
                        problem=row['problem'],gold=row['answer'],completion=text,token_ids=ids,tokens=len(ids),eos=e,truncated=not e,
                        format_eos=record['format_valid'],format_bonus=record['format_bonus'],rewards=v,
                        total_reward=record['total_reward'],reward_policy=POLICY,
                        verdict=record['verdict'],raw_verdict=record['raw_verdict'],
                        numeric_equality_checks=record['numeric_equality_checks'],
                        api_review=record.get('api_review'),
                        api_extraction_recheck=record.get('api_extraction_recheck'),
                        grading_unconfirmed=record['grading_unconfirmed'])
                    self.latest_rollout_records.append(rollout)
                    f.write(json.dumps(rollout,ensure_ascii=False)+'\n')
            return rewards

        def _generate_and_score_completions(self,inputs):
            row_ids=[r['row_id'] for r in inputs]
            groups=fresh_group_indices(row_ids,a.questions_per_generation)
            seen=getattr(self,'seen',set());assert not seen.intersection(groups),'Refuse repeated question in one-pass run'
            self.seen=seen.union(groups)
            with self.accelerator.autocast():batch=super()._generate_and_score_completions(inputs)
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
            assert self.accelerator.num_processes==1
            groups,dropped=mask_uncertain_groups(batch,row_ids,self.latest_reward_records,a.questions_per_generation)
            # Skipped groups retain no policy or KL token; zero loss must stay connected to autograd.
            self.entire_group_dropped=all(dropped.values())
            assert torch.isfinite(batch['advantages']).all()
            for rid,indices in groups.items():
                records=[self.latest_reward_records[i] for i in indices]
                unknown=any(r['grading_unconfirmed'] for r in records)
                review_reasons=[]
                if unknown:review_reasons.append('unconfirmed_grading')
                if not all(self.latest_eos[i] for i in indices):review_reasons.append('abnormal_end')
                if any(r['verdict']['status']=='extraction_failed' for r in records):review_reasons.append('answer_extraction_failure')
                if any(r['verdict'].get('error') in ('inconsistent_numeric_equality','numeric_equality_unconfirmed') for r in records):review_reasons.append('numeric_equality_guard')
                if any(r.get('api_review',{}).get('status') not in (None,'not_called') for r in records):review_reasons.append('api_review_used')
                if review_reasons:
                    with (out/'grading_review_queue.jsonl').open('a',encoding='utf-8') as f:
                        f.write(json.dumps(dict(step=self.state.global_step,row_id=rid,
                            reasons=review_reasons,dropped_group=dropped[rid],reward_policy=POLICY,
                            handling='Offline review only; never retroactively changes this optimizer update',
                            samples=[self.latest_rollout_records[i] for i in indices]),ensure_ascii=False)+'\n')
                with (out/'train_consumption.jsonl').open('a') as f:
                    f.write(json.dumps(dict(step=self.state.global_step,row_id=rid,generated_group=True,
                        dropped_group=dropped[rid],unconfirmed_grading=unknown,outcome_policy=POLICY,
                        protocol_failures=sum(r['verdict']['status']=='extraction_failed' for r in records),active_tokens=int(batch['completion_mask'][indices].sum()),
                        nonzero_advantage=bool(batch['advantages'][indices].abs().any())))+'\n')
            return batch

        def _get_per_token_logps_and_entropies(self,model,input_ids,attention_mask,logits_to_keep,batch_size=None,compute_entropy=False,**kwargs):
            assert not any(v is not None for v in kwargs.values()),'Text-only trainer'
            prompt_width=input_ids.shape[1]-logits_to_keep;lp=[];en=[]
            for i in range(input_ids.shape[0]):
                w=width(attention_mask[i:i+1,prompt_width:])
                x,y=super()._get_per_token_logps_and_entropies(model,input_ids[i:i+1,:prompt_width+w],
                    attention_mask[i:i+1,:prompt_width+w],w,batch_size=1,compute_entropy=compute_entropy)
                lp.append(F.pad(x,(0,logits_to_keep-w)))
                if compute_entropy:en.append(F.pad(y,(0,logits_to_keep-w)))
            return torch.cat(lp),torch.cat(en) if compute_entropy else None

        def _compute_loss(self,model,inputs):
            if not inputs['completion_mask'].any():
                return sum(p.sum()*0 for p in model.parameters() if p.requires_grad)
            w=width(inputs['completion_mask']);names={'completion_ids','completion_mask','old_per_token_logps','ref_per_token_logps','importance_sampling_ratio'}
            trimmed={k:v[:,:w] if k in names and isinstance(v,torch.Tensor) and v.ndim==2 else v for k,v in inputs.items()}
            return super()._compute_loss(model,trimmed)

        def training_step(self,*args,**kwargs):
            loss=super().training_step(*args,**kwargs)
            assert torch.isfinite(loss).all()
            grads=[p.grad for p in self.model.parameters() if p.requires_grad and p.grad is not None]
            assert grads and all(torch.isfinite(g).all() for g in grads),'Missing/nonfinite adapter gradients'
            return loss

    model,dtype=load_model(a.model,a.adapter,training=True)
    ref,_=load_model(a.model,a.adapter,training=True);ref.requires_grad_(False);ref.eval();ref.gradient_checkpointing_disable()
    for m in [model,ref]:
        m.generation_config.eos_token_id=t.eos_token_id;m.generation_config.pad_token_id=t.pad_token_id
    steps=2 if a.smoke else len(rows)
    config=GRPOConfig(output_dir=str(out),max_steps=steps,num_train_epochs=1,
        per_device_train_batch_size=a.train_microbatch,gradient_accumulation_steps=a.gradient_accumulation,generation_batch_size=4*a.questions_per_generation,
        num_generations=4,num_iterations=1,max_prompt_length=1024,max_completion_length=3894,
        learning_rate=5e-6,weight_decay=.01,warmup_steps=min(10,max(1,steps//10)),lr_scheduler_type='linear',optim='adamw_8bit',
        temperature=1.,min_p=0.,top_p=1.,top_k=-1,beta=.04,loss_type='grpo',scale_rewards='group',
        reward_weights=[1.,FORMAT_WEIGHT],use_vllm=True,vllm_mode='colocate',vllm_gpu_memory_utilization=a.vllm_memory,
        vllm_enable_sleep_mode=True,vllm_importance_sampling_correction=True,mask_truncated_completions=True,
        bf16=True,gradient_checkpointing=True,gradient_checkpointing_kwargs={'use_reentrant':False},
        logging_steps=1,save_strategy='steps',save_steps=max(1,steps//4),save_total_limit=2,
        save_only_model=True,seed=3407,data_seed=3407,remove_unused_columns=False,report_to=['tensorboard'])
    model.config._name_or_path=a.model
    trainer=DirectTrainer(model=model,args=config,processing_class=t,reward_funcs=[correctness,format_reward],train_dataset=Dataset.from_list(rows))
    ref=prepare_peft_model(ref,None,config);ref.requires_grad_(False);ref.eval();ref.gradient_checkpointing_disable()
    trainer.ref_model=trainer.accelerator.prepare_model(ref,evaluation_mode=True)
    model.eval();probe=t('Check reference integrity: 1 + 1 =',return_tensors='pt').to('cuda')
    with torch.no_grad(),torch.autocast('cuda',dtype=dtype):
        x=model(**probe).logits.float();y=ref(**probe).logits.float()
    diff=float((x-y).abs().max());assert torch.allclose(x,y,atol=1e-5,rtol=1e-5),diff
    assert not any(p.requires_grad for p in ref.parameters())
    del x,y,probe
    initial={n:p.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad}
    model.train();model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    save(out/'grpo_config.json',config.to_dict())
    save(out/'experiment.json',dict(model=a.model,adapter=a.adapter,adapter_sha256=adapter_sha,data_sha256=sha(a.data),
        grading_schema=SCHEMA,grader_sha256=grader_runtime['sha256'],math_verify_version=require_math_verify(),grader_runtime=grader_runtime,
        selected_questions=len(rows),planned_updates=steps,questions_per_update=1,generations_per_question=4,
        questions_per_generation=a.questions_per_generation,training_microbatch=a.train_microbatch,gradient_accumulation=a.gradient_accumulation,
        initial_reference_max_logit_difference=diff,reference='Frozen provided initial adapter; previous final GRPO adapter for replay',
        truncation_policy='Mask entire G4 group if any completion misses EOS or final-answer grading is unconfirmed',format_weight=FORMAT_WEIGHT,
        reward_policy=POLICY,reward_policy_sha256=sha(Path(__file__).with_name('online_grpo_reward.py')),
        reward_runtime=reward_runtime,
        reward_formula='correctness + 0.1 * correctness * strict_format_with_eos',reward_preflight=reward_preflight,
        grading_review_queue='grading_review_queue.jsonl; all G4 completions and original metadata; offline review only',
        api_review_enabled=bool(api_reviewer),api_review_config=api_reviewer.config if api_reviewer else None,
        api_review_policy='Only after an indeterminate primary verdict; API extracts a verbatim original answer only; pinned local Math-Verify grades it; original local format checker remains authoritative',
        api_review_source_sha256=sha(Path(__file__).with_name('api_answer_review.py')) if api_reviewer else None,
        gold_selfcheck='Parser support only; not a proof that dataset reference answers are mathematically correct',
        initial_representation='NF4 policy/reference; vLLM synchronized decoded NF4 + current LoRA',base_weights_sha256=base_hashes))
    print('GRPO_REFERENCE_VERIFIED_START',steps,flush=True);start=time.time()
    result=trainer.train()
    delta=max(float((p.detach().cpu().float()-initial[n].float()).abs().max()) for n,p in model.named_parameters() if n in initial)
    assert delta>0,'No adapter update'
    trainer.save_model(str(out/'adapter'));t.save_pretrained(out/'adapter')
    save(out/'trainer_log.json',trainer.state.log_history)
    consumed=[json.loads(s) for s in (out/'train_consumption.jsonl').read_text().splitlines()]
    assert len(consumed)==steps and len({r['row_id'] for r in consumed})==steps
    if not a.smoke:assert {r['row_id'] for r in consumed}=={r['row_id'] for r in rows}
    assert sha(Path(a.adapter)/'adapter_model.safetensors')==adapter_sha
    assert base_hashes=={p.name:sha(p) for p in Path(a.model).glob('model-*.safetensors')}
    save(out/'completion_summary.json',dict(status='complete',updates=trainer.state.global_step,
        different_questions_generated=len(consumed),groups_participating=sum(not r['dropped_group'] for r in consumed),
        groups_dropped=sum(r['dropped_group'] for r in consumed),nonzero_advantage_groups=sum(r['nonzero_advantage'] for r in consumed),
        runtime_seconds=time.time()-start,parameter_max_delta=delta,adapter_sha256=sha(out/'adapter/adapter_model.safetensors'),metrics=result.metrics))
    print('GRPO_COMPLETE',flush=True)
    trainer.llm.llm_engine.engine_core.shutdown()
    import os,sys
    os.sync();sys.stdout.flush();os._exit(0)


if __name__=='__main__':main()
