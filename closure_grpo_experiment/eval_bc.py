import argparse,collections,json,re,time
from pathlib import Path
import torch
from transformers import GenerationConfig,StoppingCriteriaList,set_seed
from common import tokenizer,read_jsonl,save_json,save_jsonl,file_hash
from dense_rollout import build_dense_rollout
from probe_completion_length import Progress
from closure_rewards import complete_format
from math_verification import verify_final_answer,verification_summary


def main():
    p=argparse.ArgumentParser()
    for x in ['model','adapter','data','output']:p.add_argument('--'+x,required=True)
    a=p.parse_args();out=Path(a.output)
    if out.exists() and any(out.iterdir()):raise ValueError('Use fresh eval directory')
    out.mkdir(parents=True);set_seed(3407)
    t=tokenizer(a.model);t.padding_side='left';rows=read_jsonl(a.data)
    assert len(rows)==8
    before=file_hash(Path(a.adapter)/'adapter_model.safetensors')
    model,dtype=build_dense_rollout(a.model,a.adapter)
    config=GenerationConfig(max_new_tokens=3894,do_sample=False,num_beams=1,
        eos_token_id=t.eos_token_id,pad_token_id=t.pad_token_id,use_cache=True,repetition_penalty=1.)
    prompts=[t.apply_chat_template(r['prompt'],tokenize=False,add_generation_prompt=True) for r in rows]
    x=t(prompts,padding=True,add_special_tokens=False,return_tensors='pt').to('cuda');width=x.input_ids.shape[1]
    start=time.perf_counter()
    with torch.inference_mode(),torch.autocast('cuda',dtype=dtype):
        generated=model.generate(**x,generation_config=config,stopping_criteria=StoppingCriteriaList([Progress(width)]))
    records=[]
    for row,ids in zip(rows,generated[:,width:].cpu().tolist()):
        eos=t.eos_token_id in ids
        if eos:ids=ids[:ids.index(t.eos_token_id)+1]
        text=t.decode(ids,skip_special_tokens=True);words=text.split()
        grams=collections.Counter(tuple(words[i:i+12]) for i in range(max(0,len(words)-11)))
        repetition=sum(v-1 for v in grams.values())/max(1,sum(grams.values()))
        records.append({'row_id':row['row_id'],'gold':row['answer'],'completion':text,'tokens':len(ids),
            'eos_finished':eos,'normal_end':eos and complete_format(text),'truncated':not eos and len(ids)>=3894,
            'repetitive':repetition>.05,'repeated_12gram_ratio':repetition,
            'math_verification':verify_final_answer(text,row['answer'])})
    assert before==file_hash(Path(a.adapter)/'adapter_model.safetensors')
    save_jsonl(out/'rollouts.jsonl',records)
    summary={'questions':len(records),'math_verify':verification_summary(r['math_verification'] for r in records),
        **{k:sum(r[k] for r in records) for k in ['normal_end','eos_finished','truncated','repetitive']},
        'mean_tokens':sum(r['tokens'] for r in records)/len(records),'runtime_seconds':time.perf_counter()-start,
        'data_sha256':file_hash(a.data),'adapter_sha256':before,'generation_config':config.to_dict(),
        'scope':'Eight-question within-source diagnostic, not an independent benchmark or proof verification'}
    save_json(out/'summary.json',summary);print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
