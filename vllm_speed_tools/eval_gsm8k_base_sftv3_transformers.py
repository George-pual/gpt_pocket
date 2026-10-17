"""Ordinary BF16 Transformers generation, paired GSM8K test, one answer/question."""
import argparse,hashlib,json,os,time,gc,importlib.metadata
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM,GenerationConfig
from peft import PeftModel
from common import tokenizer,messages
from pilot_rewards_v3 import grade,format_ok
from eval_gsm8k_base_sftv3 import digest,numeric_grade

def main():
    p=argparse.ArgumentParser()
    for n in ('model','adapter','data','output'):p.add_argument('--'+n,required=True)
    p.add_argument('--batch-size',type=int,default=64)
    a=p.parse_args();out=Path(a.output)
    assert not out.exists(),'Refuse overwrite'
    assert digest(Path(a.adapter)/'adapter_model.safetensors')=='5461e88ae8dba6a3fb536e2bd93bdd8e6f3098a3234943eaef28bc0196017b9b'
    rows=[json.loads(s) for s in Path(a.data).read_bytes().splitlines() if s.strip()]
    assert len(rows)==1319 and len({r['question'] for r in rows})==1319
    out.mkdir(parents=True)
    t=tokenizer(a.adapter);t.padding_side='left'
    ids=[t.apply_chat_template(messages(r['question']),tokenize=True,add_generation_prompt=True) for r in rows]
    assert max(map(len,ids))+3894<=8192
    config=dict(n=1,do_sample=False,max_new_tokens=3894,eos_token_id=t.eos_token_id,pad_token_id=t.pad_token_id,use_cache=True)
    prompt_hash=hashlib.sha256(json.dumps(ids).encode()).hexdigest()
    provenance=dict(engine='Transformers; no vLLM',model=a.model,adapter=a.adapter,
                    data_sha256=digest(a.data),adapter_sha256=digest(Path(a.adapter)/'adapter_model.safetensors'),
                    prompt_tokens_sha256=prompt_hash,chat_template=t.chat_template,generation=config,
                    requested_batch_size=a.batch_size,oom_policy='Halve batch only; keep generation length and all questions',
                    main_metric='Numeric exact match: structured final answer then last-number fallback; independent of format',
                    strict_metric='Previous pilot final-answer extraction score',
                    versions={k:importlib.metadata.version(k) for k in ['torch','transformers','peft']})
    (out/'evaluation_config.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2))
    model=AutoModelForCausalLM.from_pretrained(a.model,local_files_only=True,torch_dtype=torch.bfloat16,
                                             device_map={'':'cuda:0'},attn_implementation='sdpa')
    model.eval()
    generation=GenerationConfig(**{k:v for k,v in config.items() if k!='n'})
    # Stable length sorting reduces left padding, shared by both arms.
    order=sorted(range(len(rows)),key=lambda i:len(ids[i]))
    summaries={};batch_size=a.batch_size
    for arm in ('base','sft_v3'):
        if arm=='sft_v3':
            model=PeftModel.from_pretrained(model,a.adapter,is_trainable=False,local_files_only=True)
            model.eval()
        (out/'stage.txt').write_text(arm)
        folder=out/arm;folder.mkdir();samples=[];offset=0;started=time.time()
        with (folder/'answers.jsonl').open('w',encoding='utf-8') as f:
            while offset<len(order):
                indices=order[offset:offset+batch_size]
                batch=t.pad({'input_ids':[ids[i] for i in indices]},padding=True,return_tensors='pt').to('cuda')
                width=batch['input_ids'].shape[1]
                try:
                    with torch.inference_mode():
                        generated=model.generate(**batch,generation_config=generation)
                    outputs=generated[:,width:].cpu().tolist()
                    del generated
                except torch.OutOfMemoryError:
                    del batch
                    gc.collect();torch.cuda.empty_cache()
                    if batch_size==1:raise
                    batch_size=max(1,batch_size//2)
                    print('OOM_RETRY',arm,'batch_size',batch_size,'offset',offset,flush=True)
                    continue
                del batch
                for index,token_ids in zip(indices,outputs):
                    r=rows[index];gold=r['answer'].rsplit('####',1)[1].strip()
                    ended=t.eos_token_id in token_ids
                    if ended:token_ids=token_ids[:token_ids.index(t.eos_token_id)+1]
                    text=t.decode(token_ids,skip_special_tokens=True)
                    correct,pred,method=numeric_grade(text,gold);strict=grade(text,gold,'gsm8k')
                    record=dict(row_id=f'gsm8k:main:test:{index}',question=r['question'],gold=gold,
                                completion=text,tokens=len(token_ids),correct=bool(correct),prediction=pred,
                                extraction_method=method,strict_correct=bool(strict['correct']),strict_verdict=strict,
                                format_eos=format_ok(text,ended),eos=ended,truncated=not ended,
                                batch_size=batch_size)
                    f.write(json.dumps(record,ensure_ascii=False)+'\n');samples.append(record)
                offset+=len(indices);f.flush()
                progress=dict(completed=len(samples),total=len(rows),correct=sum(s['correct'] for s in samples),
                              batch_size=batch_size,runtime_seconds=time.time()-started)
                (folder/'progress.json').write_text(json.dumps(progress))
                print('EVAL_PROGRESS',arm,json.dumps(progress),flush=True)
        assert len(samples)==1319 and len({s['row_id'] for s in samples})==1319
        summary=dict(status='complete',questions=len(samples),correct=sum(s['correct'] for s in samples),
                     accuracy=sum(s['correct'] for s in samples)/len(samples),strict_correct=sum(s['strict_correct'] for s in samples),
                     format_complete=sum(s['format_eos'] for s in samples),
                     correct_and_format_complete=sum(s['correct'] and s['format_eos'] for s in samples),
                     eos=sum(s['eos'] for s in samples),truncated=sum(s['truncated'] for s in samples),
                     mean_tokens=sum(s['tokens'] for s in samples)/len(samples),runtime_seconds=time.time()-started,
                     data_sha256=provenance['data_sha256'],prompt_tokens_sha256=prompt_hash,generation=config,
                     final_batch_size=batch_size)
        (folder/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2));summaries[arm]=summary
        print('EVAL_COMPLETE',arm,json.dumps(summary),flush=True)
    (out/'comparison.json').write_text(json.dumps(summaries,ensure_ascii=False,indent=2))
    (out/'stage.txt').write_text('complete')
    os.sync()

if __name__=='__main__':main()
