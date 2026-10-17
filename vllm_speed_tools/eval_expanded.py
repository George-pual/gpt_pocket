"""Paired inference only; BF16 vLLM copy of frozen decoded NF4+LoRA."""
import argparse, collections, gc, json, os, time
from pathlib import Path
from types import SimpleNamespace
import torch
from common import tokenizer,load_model,read_jsonl,save_json,save_jsonl,file_hash
from closure_rewards import complete_format
from math_verification import verify_final_answer,verification_summary
from reward_v2 import repetition_ratio
from safe_vllm_trainer import SafeVLLMTrainer

p=argparse.ArgumentParser()
for key in ['model','adapter','data','output']:p.add_argument('--'+key,required=True)
a=p.parse_args();out=Path(a.output)
if out.exists():raise RuntimeError('Fresh evaluation output required')
out.mkdir(parents=True)
rows=read_jsonl(a.data);assert len(rows)==200
t=tokenizer(a.model)
before=file_hash(Path(a.adapter)/'adapter_model.safetensors')
torch.manual_seed(3407)
policy,_=load_model(a.model,a.adapter,training=True)
policy.requires_grad_(False);policy.eval();policy.gradient_checkpointing_disable()
from vllm import LLM,SamplingParams
llm=LLM(model=a.model,dtype='bfloat16',max_model_len=4096,gpu_memory_utilization=.5,
        max_num_seqs=16,seed=3407,enable_prefix_caching=False,disable_log_stats=True)
proxy=SimpleNamespace(model=policy,accelerator=SimpleNamespace(unwrap_model=lambda x:x),
                      llm=llm,args=SimpleNamespace(output_dir=str(out)),
                      state=SimpleNamespace(global_step=0))
SafeVLLMTrainer._move_model_to_vllm(proxy)
del proxy,policy;gc.collect();torch.cuda.empty_cache()
params=SamplingParams(n=1,temperature=0.,max_tokens=3894,
                      stop_token_ids=[t.eos_token_id],seed=3407)
records=[];start=time.time()
for offset in range(0,len(rows),16):
    batch=rows[offset:offset+16]
    prompts=[t.apply_chat_template(r['prompt'],tokenize=True,add_generation_prompt=True) for r in batch]
    assert all(len(x)<=202 for x in prompts)
    generated=llm.generate([{'prompt_token_ids':x} for x in prompts],params,use_tqdm=False)
    for row,result in zip(batch,generated):
        answer=result.outputs[0];ids=list(answer.token_ids)
        text=t.decode(ids,skip_special_tokens=True)
        eos=t.eos_token_id in ids or answer.stop_reason==t.eos_token_id
        ratio=repetition_ratio(text)
        record={'row_id':row['row_id'],'gold':row['answer'],'completion':text,'tokens':len(ids),
                'eos_finished':eos,'normal_end':eos and complete_format(text),
                'finish_reason':answer.finish_reason,'stop_reason':answer.stop_reason,
                'truncated':answer.finish_reason=='length','repetitive':ratio>.05,
                'repeated_12gram_ratio':ratio,'math_verification':verify_final_answer(text,row['answer'])}
        records.append(record)
    save_jsonl(out/'rollouts.jsonl',records)
    save_json(out/'progress.json',{'completed':len(records),'total':200,'seconds':time.time()-start})
    print('EVAL_PROGRESS',len(records),'/200',round(time.time()-start,1),flush=True)
assert file_hash(Path(a.adapter)/'adapter_model.safetensors')==before
save_json(out/'summary.json',{'questions':200,'math_verify':verification_summary(r['math_verification'] for r in records),
    **{k:sum(r[k] for r in records) for k in ['normal_end','eos_finished','truncated','repetitive']},
    'mean_tokens':sum(r['tokens'] for r in records)/200,'runtime_seconds':time.time()-start,
    'adapter_sha256':before,'data_sha256':file_hash(a.data),
    'generation_config':{'engine':'vllm','temperature':0,'max_tokens':3894,'seed':3407,'batch_size':16},
    'representation':'Frozen decoded NF4 plus LoRA streamed into BF16 vLLM; not bitwise HF mixed-dtype inference',
    'scope':'Fresh within-source P90-filtered held-out diagnostic, not external benchmark; no proof verification'})
llm.llm_engine.engine_core.shutdown()
from vllm.distributed.parallel_state import destroy_model_parallel
destroy_model_parallel()
if torch.distributed.is_initialized():torch.distributed.destroy_process_group()
os.sync();print('EVAL_SAVED',flush=True)
import sys
sys.stdout.flush();sys.stderr.flush();os._exit(0)
