"""Full official MATH L2/L3 evaluation, identical settings across checkpoints."""
import argparse,hashlib,json,os,time
from collections import Counter
from pathlib import Path

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,value):Path(p).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data-dir',required=True);p.add_argument('--output',required=True)
    p.add_argument('--model',default='/root/autodl-tmp/qwen3_hf/model')
    p.add_argument('--runs-root',default='/root/autodl-tmp/qwen3_hf/runs')
    p.add_argument('--arms',default='base,sft_v3,grpo1024,grpo1280,grpo1536')
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args();data=Path(a.data_dir);out=Path(a.output);runs=Path(a.runs_root)
    paths={'base':None,'sft_v3':runs/'E1_sft_v3/adapter',
           'grpo1024':runs/'E1_sft_v3_half_math1024_fast_v2/B_components/checkpoint-1024',
           'grpo1280':runs/'E1_sft_v3_expanded_continue1536_v1/B_components/checkpoint-256',
           'grpo1536':runs/'E1_sft_v3_expanded_continue1536_v1/B_components/checkpoint-512'}
    arms=a.arms.split(',');assert len(set(arms))==len(arms) and all(x in paths for x in arms)
    manifest=json.loads((data/'manifest.json').read_text(encoding='utf-8'))
    assert manifest['split']=='test' and manifest['independent_questions']==2024
    assert digest(data/'questions.jsonl')==manifest['data_sha256']
    rows=[json.loads(x) for x in (data/'questions.jsonl').read_bytes().splitlines() if x.strip()]
    assert len(rows)==2024 and len({x['row_id'] for x in rows})==2024
    assert Counter(x['level'] for x in rows)=={'Level 2':894,'Level 3':1130}
    assert all(x['split']=='test' and not x['overlaps_training_exact_normalized'] for x in rows)
    hashes={arm:digest(paths[arm]/'adapter_model.safetensors') for arm in arms if paths[arm]}
    if a.dry_run:
        print(json.dumps(dict(status='data_and_adapter_paths_verified',questions=len(rows),adapter_sha256=hashes),indent=2));return
    if out.exists():raise RuntimeError('Refuse overwrite of evaluation output')
    import torch
    if not torch.cuda.is_available():raise RuntimeError('CUDA GPU required; no evaluation started')
    from common import tokenizer,messages
    from pilot_rewards_v3 import grade,format_ok
    from math_verification import SCHEMA,require_math_verify
    from vllm import LLM,SamplingParams
    from vllm.lora.request import LoRARequest
    t=tokenizer(runs/'E1_sft_v3/adapter')
    prompts=[t.apply_chat_template(messages(x['problem']),tokenize=True,add_generation_prompt=True) for x in rows]
    assert max(map(len,prompts))+3894<=8192
    out.mkdir(parents=True)
    sampling=dict(n=1,temperature=0.,top_p=1.,top_k=-1,max_tokens=3894,eos_token_id=t.eos_token_id,ignore_eos=False)
    provenance=dict(data_sha256=manifest['data_sha256'],questions=len(rows),model=a.model,
                    adapter_sha256=hashes,prompt_tokens_sha256=hashlib.sha256(json.dumps(prompts).encode()).hexdigest(),
                    sampling=sampling,chat_template=t.chat_template,scorer_schema=SCHEMA,
                    math_verify_version=require_math_verify(),scorer_sha256=digest(Path(__file__).with_name('math_verification.py')))
    save(out/'evaluation_config.json',provenance)
    llm=LLM(model=a.model,tokenizer=str(runs/'E1_sft_v3/adapter'),dtype='bfloat16',
            enable_lora=True,max_lora_rank=32,max_loras=4,max_model_len=8192,
            gpu_memory_utilization=.88,max_num_seqs=128,max_num_batched_tokens=8192,
            enable_prefix_caching=True,seed=3407)
    params=SamplingParams(n=1,temperature=0.,top_p=1.,top_k=-1,max_tokens=3894,
                          stop_token_ids=[t.eos_token_id],seed=3407)
    comparisons={}
    for arm in arms:
        folder=out/arm;folder.mkdir();records=[];started=time.time()
        (out/'stage.txt').write_text(arm)
        request=None if arm=='base' else LoRARequest(arm,list(paths).index(arm),str(paths[arm]))
        with (folder/'answers.jsonl').open('w',encoding='utf-8') as f:
            for start in range(0,len(rows),64):
                part=rows[start:start+64]
                generated=llm.generate([{'prompt_token_ids':x} for x in prompts[start:start+64]],params,
                                       lora_request=request,use_tqdm=False)
                assert len(generated)==len(part)
                for row,result in zip(part,generated):
                    assert len(result.outputs)==1
                    s=result.outputs[0];ended=s.finish_reason=='stop' and s.stop_reason in (None,t.eos_token_id)
                    verdict=grade(s.text,row['answer'],'math')
                    record=dict(row_id=row['row_id'],question=row['problem'],gold=row['answer'],level=row['level'],
                                subject=row['type'],completion=s.text,tokens=len(s.token_ids),correct=bool(verdict['correct']),
                                prediction=verdict.get('prediction'),strict_verdict=verdict,
                                format_eos=format_ok(s.text,ended),eos=ended,truncated=s.finish_reason=='length',finish_reason=s.finish_reason)
                    records.append(record);f.write(json.dumps(record,ensure_ascii=False)+'\n')
                f.flush();save(folder/'progress.json',dict(completed=len(records),total=len(rows),correct=sum(x['correct'] for x in records)))
                print('EVAL_PROGRESS',arm,len(records),'/',len(rows),flush=True)
        def stats(rs):
            return dict(questions=len(rs),correct=sum(x['correct'] for x in rs),accuracy=sum(x['correct'] for x in rs)/len(rs),
                        format_complete=sum(x['format_eos'] for x in rs),truncated=sum(x['truncated'] for x in rs),
                        grading_statuses=dict(Counter(x['strict_verdict']['status'] for x in rs)))
        summary={**provenance,**stats(records),'status':'complete',
                 'by_level':{lv:stats([x for x in records if x['level']==lv]) for lv in ['Level 2','Level 3']},
                 'runtime_seconds':time.time()-started}
        # Existing keys in provenance agree with stats (same full question count).
        save(folder/'summary.json',summary);comparisons[arm]=summary;save(out/'comparison.json',comparisons)
        if paths[arm]:assert digest(paths[arm]/'adapter_model.safetensors')==hashes[arm]
    (out/'stage.txt').write_text('complete');print('ALL_COMPLETE',flush=True)
    llm.llm_engine.engine_core.shutdown()
    from vllm.distributed.parallel_state import destroy_model_parallel
    destroy_model_parallel()
    if torch.distributed.is_initialized():torch.distributed.destroy_process_group()
    os.sync()

if __name__=='__main__':main()
