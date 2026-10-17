"""No optimizer, no training; compare warm generation on identical input tokens."""
import argparse,json,time,os,statistics
from pathlib import Path


def save(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')


def main():
    p=argparse.ArgumentParser();p.add_argument('--engine',choices=['hf','vllm'],required=True)
    for n in ['model','data','output']:p.add_argument('--'+n,required=True)
    a=p.parse_args();out=Path(a.output)
    if out.exists() and any(out.iterdir()):raise ValueError('Fresh benchmark output required')
    out.mkdir(parents=True)
    os.environ['HF_HUB_OFFLINE']='1';os.environ['HF_DATASETS_OFFLINE']='1'
    import torch
    from transformers import AutoTokenizer,set_seed
    set_seed(3407);t=AutoTokenizer.from_pretrained(a.model,local_files_only=True);t.padding_side='left'
    rows=[json.loads(x) for x in Path(a.data).read_text().splitlines()][:2]
    texts=[t.apply_chat_template(r['prompt'],tokenize=False,add_generation_prompt=True) for r in rows for _ in range(4)]
    prompts=t(texts,add_special_tokens=False)['input_ids'];start=time.perf_counter()
    if a.engine=='hf':
        from transformers import AutoModelForCausalLM,GenerationConfig
        model=AutoModelForCausalLM.from_pretrained(a.model,torch_dtype=torch.bfloat16,
            local_files_only=True,device_map={'':0},attn_implementation='sdpa').eval()
        def generate(cap,ignore_eos=False,warm=False):
            inputs=t.pad({'input_ids':prompts[:1] if warm else prompts},padding=True,return_tensors='pt').to('cuda')
            config=GenerationConfig(max_new_tokens=cap,do_sample=True,temperature=1.,top_p=1.,top_k=0,min_p=.1,
                eos_token_id=[] if ignore_eos else t.eos_token_id,pad_token_id=t.pad_token_id,
                use_cache=True,bos_token_id=None)
            torch.cuda.synchronize();begin=time.perf_counter()
            with torch.inference_mode():output=model.generate(**inputs,generation_config=config)
            torch.cuda.synchronize();elapsed=time.perf_counter()-begin
            records=[]
            for ids in output[:,inputs.input_ids.shape[1]:].cpu().tolist():
                if not ignore_eos and t.eos_token_id in ids:ids=ids[:ids.index(t.eos_token_id)+1]
                records.append(ids)
            return elapsed,records
    else:
        from vllm import LLM,SamplingParams
        # A bounded 32GB single-card benchmark; no training model resident.
        model=LLM(model=a.model,dtype='bfloat16',tensor_parallel_size=1,max_model_len=4096,
            gpu_memory_utilization=.75,max_num_seqs=8,enable_prefix_caching=False,
            seed=3407,disable_log_stats=True,enforce_eager=False)
        def generate(cap,ignore_eos=False,warm=False):
            params=SamplingParams(n=1,temperature=1.,top_p=1.,top_k=-1,min_p=.1,
                max_tokens=cap,ignore_eos=ignore_eos,seed=3407,
                stop_token_ids=[] if ignore_eos else [t.eos_token_id])
            begin=time.perf_counter()
            outputs=model.generate([{'prompt_token_ids':x} for x in (prompts[:1] if warm else prompts)],params,use_tqdm=False)
            elapsed=time.perf_counter()-begin
            return elapsed,[list(o.outputs[0].token_ids) for o in outputs]
    load_seconds=time.perf_counter()-start
    warm_seconds,_=generate(32,warm=True)
    results=[]
    # Equal work removes random output-length differences from speed comparison.
    for repeat in range(2):
        elapsed,ids=generate(512,ignore_eos=True)
        assert len(ids)==8 and all(len(x)==512 for x in ids)
        results.append({'case':'fixed_512','repeat':repeat,'seconds':elapsed,'tokens':4096,
            'tokens_per_second':4096/elapsed})
        print(json.dumps(results[-1]),flush=True)
    elapsed,ids=generate(3894)
    total=sum(len(x) for x in ids)
    results.append({'case':'natural_3894','seconds':elapsed,'tokens':total,
        'tokens_per_second':total/elapsed,'lengths':[len(x) for x in ids],
        'eos_count':sum(t.eos_token_id in x for x in ids)})
    save(out/'natural_outputs.json', [{'question_index':i//4,'sample_index':i%4,'tokens':len(x),
        'completion':t.decode(x,skip_special_tokens=True)} for i,x in enumerate(ids)])
    summary={'engine':a.engine,'model':a.model,'load_seconds':load_seconds,'warmup_seconds':warm_seconds,
        'sampling':{'temperature':1.,'top_p':1.,'min_p':.1,'seed':3407},'answers':8,
        'prompt_token_ids':prompts,'results':results,
        'fixed_mean_tokens_per_second':statistics.mean(x['tokens_per_second'] for x in results if x['case']=='fixed_512'),
        'scope':'Frozen generation benchmark. Does not include GRPO scoring, backward, reference model, optimizer, or weight synchronization.'}
    save(out/'summary.json',summary);print('BENCHMARK_COMPLETE',a.engine,flush=True)


if __name__=='__main__':main()
