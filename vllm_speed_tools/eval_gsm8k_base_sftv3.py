"""Paired single-answer GSM8K test evaluation; numeric and format metrics separate."""
import argparse,hashlib,json,re,time,os
from fractions import Fraction
from collections import Counter
from pathlib import Path
from common import tokenizer,messages
from pilot_rewards_v3 import grade,format_ok,numeric_answer
from math_verification import extract_final_answer

def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def numeric_grade(text,gold):
    # Keep the previous strict extraction score separately. For GSM8K accuracy,
    # permit the conventional last-number fallback when no structured final
    # answer can be extracted; never scan reasoning for a matching gold value.
    pred,origin,error=extract_final_answer(text)
    if not error:
        value=numeric_answer(pred)
        if value is not None:
            return value==numeric_answer(gold),str(pred),origin
    nums=re.findall(r'-?(?:\d+(?:\.\d+)?|\.\d+)',text.replace(',',''))
    if not nums:return False,None,'no_number'
    return Fraction(nums[-1])==Fraction(gold.replace(',','')),nums[-1],'last_number_fallback'

def main():
    p=argparse.ArgumentParser()
    for n in ('model','adapter','data','output'):p.add_argument('--'+n,required=True)
    p.add_argument('--dataset-kind',choices=['gsm8k','math'],default='gsm8k')
    p.add_argument('--arms',default='base,sft_v3')
    p.add_argument('--expected-adapter-sha256',default='5461e88ae8dba6a3fb536e2bd93bdd8e6f3098a3234943eaef28bc0196017b9b')
    a=p.parse_args();out=Path(a.output)
    assert not out.exists(),'Refuse output overwrite'
    assert digest(Path(a.adapter)/'adapter_model.safetensors')==a.expected_adapter_sha256
    arms=a.arms.split(',')
    assert arms and len(set(arms))==len(arms) and set(arms)<={'base','sft_v3','grpo1024','grpo1280','grpo1536'}
    rows=[json.loads(s) for s in Path(a.data).read_bytes().splitlines() if s.strip()]
    expected=1319 if a.dataset_kind=='gsm8k' else 200
    if a.dataset_kind=='math':
        rows=[dict(r,question=r['problem']) for r in rows]
        assert Counter(r['level'] for r in rows)=={'Level 2':100,'Level 3':100}
    assert len(rows)==expected and len({r['question'] for r in rows})==expected
    out.mkdir(parents=True)
    from vllm import LLM,SamplingParams
    from vllm.lora.request import LoRARequest
    t=tokenizer(a.adapter)
    prompts=[t.apply_chat_template(messages(r['question']),tokenize=True,add_generation_prompt=True) for r in rows]
    assert max(map(len,prompts))+3894<=8192
    config=dict(n=1,temperature=0.,top_p=1.,top_k=-1,max_tokens=3894,eos_token_id=t.eos_token_id,ignore_eos=False)
    provenance=dict(data_sha256=digest(a.data),adapter_sha256=digest(Path(a.adapter)/'adapter_model.safetensors'),
                    prompt_tokens_sha256=hashlib.sha256(json.dumps(prompts).encode()).hexdigest(),
                    sampling=config,chat_template=t.chat_template,prompt_max_tokens=max(map(len,prompts)),
                    main_metric=('Numeric exact match with structured-answer extraction then last-number fallback; independent of format score' if a.dataset_kind=='gsm8k' else 'Math-Verify final-answer equivalence; same extraction rules for both models; format reported separately'),
                    strict_metric='Previous pilot grader, requires an extractable final answer',
                    model=a.model,adapter=a.adapter)
    (out/'evaluation_config.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2))
    (out/'stage.txt').write_text('loading_model')
    llm=LLM(model=a.model,tokenizer=a.adapter,dtype='bfloat16',enable_lora=True,max_lora_rank=32,
            max_model_len=8192,gpu_memory_utilization=.88,max_num_seqs=128,max_num_batched_tokens=8192,
            enable_prefix_caching=True,seed=3407)
    summaries={}
    for arm in arms:
        request=None if arm=='base' else LoRARequest(arm,1,a.adapter)
        (out/'stage.txt').write_text(arm)
        folder=out/arm;folder.mkdir()
        samples=[];started=time.time()
        (folder/'progress.json').write_text(json.dumps(dict(completed=0,total=len(rows),correct=0)))
        with (folder/'answers.jsonl').open('w',encoding='utf-8') as f:
            for offset in range(0,len(rows),64):
                part=rows[offset:offset+64]
                inputs=[{'prompt_token_ids':ids} for ids in prompts[offset:offset+64]]
                params=SamplingParams(n=1,temperature=0.,top_p=1.,top_k=-1,max_tokens=3894,
                                      stop_token_ids=[t.eos_token_id],seed=3407)
                outputs=llm.generate(inputs,params,lora_request=request,use_tqdm=False)
                for index,(r,result) in enumerate(zip(part,outputs),start=offset):
                    assert len(result.outputs)==1
                    s=result.outputs[0];gold=r['answer'].rsplit('####',1)[1].strip() if a.dataset_kind=='gsm8k' else r['answer']
                    ended=s.finish_reason=='stop' and s.stop_reason in (None,t.eos_token_id)
                    strict=grade(s.text,gold,a.dataset_kind)
                    if a.dataset_kind=='gsm8k':correct,pred,method=numeric_grade(s.text,gold)
                    else:correct,pred,method=strict['correct'],strict.get('prediction'),'math_final_answer'
                    record=dict(row_id=r.get('row_id',f'gsm8k:main:test:{index}'),question=r['question'],gold=gold,level=r.get('level'),
                                completion=s.text,tokens=len(s.token_ids),correct=bool(correct),prediction=pred,
                                extraction_method=method,strict_correct=bool(strict['correct']),strict_verdict=strict,
                                format_eos=format_ok(s.text,ended),eos=ended,truncated=s.finish_reason=='length',finish_reason=s.finish_reason)
                    f.write(json.dumps(record,ensure_ascii=False)+'\n');samples.append(record)
                f.flush()
                (folder/'progress.json').write_text(json.dumps(dict(completed=len(samples),total=len(rows),correct=sum(s['correct'] for s in samples))))
                print('EVAL_PROGRESS',arm,len(samples),'/',len(rows),flush=True)
        assert len(samples)==expected and len({s['row_id'] for s in samples})==expected
        summary=dict(status='complete',questions=len(samples),correct=sum(s['correct'] for s in samples),
                     accuracy=sum(s['correct'] for s in samples)/len(samples),
                     strict_correct=sum(s['strict_correct'] for s in samples),
                     format_complete=sum(s['format_eos'] for s in samples),
                     correct_and_format_complete=sum(s['correct'] and s['format_eos'] for s in samples),
                     eos=sum(s['eos'] for s in samples),truncated=sum(s['truncated'] for s in samples),
                     mean_tokens=sum(s['tokens'] for s in samples)/len(samples),runtime_seconds=time.time()-started,
                     data_sha256=provenance['data_sha256'],prompt_tokens_sha256=provenance['prompt_tokens_sha256'],sampling=config)
        if a.dataset_kind=='math':
            summary['by_level']={level:dict(questions=sum(s['level']==level for s in samples),correct=sum(s['correct'] for s in samples if s['level']==level),format_complete=sum(s['format_eos'] for s in samples if s['level']==level),truncated=sum(s['truncated'] for s in samples if s['level']==level)) for level in ('Level 2','Level 3')}
        (folder/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2));summaries[arm]=summary
        print('EVAL_COMPLETE',arm,json.dumps(summary),flush=True)
    (out/'comparison.json').write_text(json.dumps(summaries,ensure_ascii=False,indent=2))
    (out/'stage.txt').write_text('complete')
    llm.llm_engine.engine_core.shutdown()
    from vllm.distributed.parallel_state import destroy_model_parallel
    destroy_model_parallel()
    import sys
    os.sync();sys.stdout.flush();os._exit(0)

if __name__=='__main__':main()
