"""Identical vLLM native BF16+LoRA deployment evaluation for all three arms."""
import argparse,json,time,hashlib,os
from pathlib import Path
from collections import Counter
from common import tokenizer,read_jsonl,save_json
from pilot_rewards_v3 import grade,format_ok,VERSION

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--adapter',required=True);p.add_argument('--data',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    if (out/'answers.jsonl').exists():raise RuntimeError('Refuse eval overwrite')
    from vllm import LLM,SamplingParams
    from vllm.lora.request import LoRARequest
    t=tokenizer(a.adapter);rows=read_jsonl(a.data)
    llm=LLM(model=a.model,tokenizer=a.adapter,dtype='bfloat16',enable_lora=True,max_lora_rank=32,max_model_len=8192,gpu_memory_utilization=.88,max_num_seqs=128,max_num_batched_tokens=8192,enable_prefix_caching=True,seed=3407)
    request=LoRARequest('pilot',1,a.adapter);samples=[];started=time.time()
    with (out/'answers.jsonl').open('w',encoding='utf-8') as f:
        for offset in range(0,len(rows),25):
            part=rows[offset:offset+25]
            prompts=[{'prompt_token_ids':t.apply_chat_template(r['prompt'],tokenize=True,add_generation_prompt=True)} for r in part]
            params=[SamplingParams(n=4,temperature=1.,top_p=1.,top_k=-1,max_tokens=3894,stop_token_ids=[t.eos_token_id],seed=int(hashlib.sha256(('eval3407:'+r['row_id']).encode()).hexdigest()[:8],16)) for r in part]
            outputs=llm.generate(prompts,params,lora_request=request,use_tqdm=False)
            for r,output in zip(part,outputs):
                for s in output.outputs:
                    ended=s.finish_reason=='stop' and s.stop_reason in (None,t.eos_token_id)
                    verdict=grade(s.text,r['answer'],r['source'])
                    record=dict(row_id=r['row_id'],source=r['source'],level=r['level'],gold=r['answer'],sample_index=s.index,completion=s.text,tokens=len(s.token_ids),correct=verdict['correct'],format_eos=format_ok(s.text,ended),eos=ended,truncated=s.finish_reason=='length',verdict=verdict)
                    f.write(json.dumps(record,ensure_ascii=False)+'\n');samples.append(record)
            f.flush();print('EVAL_PROGRESS',len(samples),'/400',flush=True)
    def metric(ss):
        byid={s['row_id'] for s in ss}
        return dict(questions=len(byid),answers=len(ss),correct=sum(s['correct'] for s in ss),format_eos=sum(s['format_eos'] for s in ss),normal_correct=sum(s['correct'] and s['format_eos'] for s in ss),eos=sum(s['eos'] for s in ss),truncated=sum(s['truncated'] for s in ss),observed_any_correct=sum(any(s['correct'] for s in ss if s['row_id']==rid) for rid in byid),mean_tokens=sum(s['tokens'] for s in ss)/len(ss))
    save_json(out/'summary.json',dict(status='complete',reward_version=VERSION,adapter=a.adapter,adapter_sha256=hashlib.sha256((Path(a.adapter)/'adapter_model.safetensors').read_bytes()).hexdigest(),data_sha256=hashlib.sha256(Path(a.data).read_bytes()).hexdigest(),sampling=dict(n=4,temperature=1.,top_p=1.,top_k=-1,max_tokens=3894),overall=metric(samples),by_source={src:metric([s for s in samples if s['source']==src]) for src in ['gsm8k','math']},runtime_seconds=time.time()-started))
    print('EVAL_COMPLETE',flush=True)
    llm.llm_engine.engine_core.shutdown()
    from vllm.distributed.parallel_state import destroy_model_parallel
    destroy_model_parallel()
    import sys
    os.sync();sys.stdout.flush();os._exit(0)

if __name__=='__main__':main()
