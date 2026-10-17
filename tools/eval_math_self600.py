"""Paired development generation for fresh Base and cold-start SFT snapshots."""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import time


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    for k in ('model','data','output'):
        p.add_argument('--'+k, required=True)
    p.add_argument('--training-dir')
    p.add_argument('--reuse-base', help='Reuse a matching previously completed Base development evaluation')
    a = p.parse_args()
    from common import tokenizer, messages
    from math_verification import verify_final_answer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    out=Path(a.output)
    out.mkdir(parents=True,exist_ok=True)
    with Path(a.data).open(encoding='utf-8') as f:
        rows=[json.loads(s) for s in f if s.strip()]
    assert len(rows)==len({r['row_id'] for r in rows})==200
    t=tokenizer(a.model)
    prompts=[t.apply_chat_template(messages(r['problem']),tokenize=True,add_generation_prompt=True) for r in rows]
    assert max(map(len,prompts))+3894<=8192
    arms={'base':None}
    if a.training_dir:
        root=Path(a.training_dir)
        for step in (8,16,32,64):
            path=root/'snapshots'/f'step-{step}'
            if (path/'adapter_model.safetensors').exists():arms[f'step-{step}']=path
    config=dict(model=a.model,data_sha256=sha(a.data),prompts_sha256=hashlib.sha256(json.dumps(prompts).encode()).hexdigest(),
                arms={k:sha(v/'adapter_model.safetensors') if v else None for k,v in arms.items()},
                sampling=dict(n=1,temperature=0.,top_p=1.,top_k=-1,max_tokens=3894,seed=3407,eos_token_id=t.eos_token_id),
                engine=dict(dtype='bfloat16',max_model_len=8192,enable_lora=True,max_lora_rank=64,max_num_seqs=64),
                grader_sha256=sha(Path(__file__).with_name('math_verification.py')))
    if a.reuse_base:
        previous=Path(a.reuse_base)
        prior=json.loads((previous/'evaluation_manifest.json').read_text())
        assert {k:v for k,v in prior.items() if k not in ('arms','base_reuse')}=={k:v for k,v in config.items() if k!='arms'}
        saved=previous/'base/answers.jsonl'
        with saved.open(encoding='utf-8') as f:base_records=[json.loads(s) for s in f if s.strip()]
        assert [r['row_id'] for r in base_records]==[r['row_id'] for r in rows]
        base_dest=out/'base/answers.jsonl'
        base_dest.parent.mkdir(exist_ok=True)
        if base_dest.exists():assert sha(base_dest)==sha(saved)
        else:base_dest.write_bytes(saved.read_bytes())
        config['base_reuse']={'path':str(saved),'sha256':sha(saved)}
    mp=out/'evaluation_manifest.json'
    if mp.exists():assert json.loads(mp.read_text())==config
    else:mp.write_text(json.dumps(config,indent=2),encoding='utf-8')
    engine=LLM(model=a.model,dtype='bfloat16',max_model_len=8192,enable_lora=True,max_lora_rank=64,
               gpu_memory_utilization=.85,max_num_seqs=64,seed=3407,enable_prefix_caching=True,disable_log_stats=True)
    all_records={}; summaries={}
    for arm_index,(arm,path) in enumerate(arms.items(),1):
        dest=out/arm
        dest.mkdir(exist_ok=True)
        result_path=dest/'answers.jsonl'
        records=[]
        if result_path.exists():
            with result_path.open(encoding='utf-8') as f:records=[json.loads(s) for s in f if s.strip()]
            assert [r['row_id'] for r in records]==[r['row_id'] for r in rows[:len(records)]]
        request=LoRARequest(arm,arm_index,str(path)) if path else None
        start=time.time()
        for offset in range(len(records),len(rows),64):
            batch=rows[offset:offset+64]
            results=engine.generate([{'prompt_token_ids':ids} for ids in prompts[offset:offset+64]],
                    SamplingParams(n=1,temperature=0.,top_p=1.,top_k=-1,max_tokens=3894,
                                   stop_token_ids=[t.eos_token_id],seed=3407),lora_request=request,use_tqdm=False)
            with result_path.open('a',encoding='utf-8') as f:
                for row,res in zip(batch,results):
                    s=res.outputs[0]
                    eos=s.finish_reason=='stop' and s.stop_reason in (None,t.eos_token_id)
                    verdict=verify_final_answer(s.text,row['answer'])
                    words=s.text.lower().split()
                    grams=[tuple(words[i:i+12]) for i in range(max(0,len(words)-11))]
                    repetition=1-len(set(grams))/len(grams) if grams else 0
                    format_eos=bool(eos and s.text.count('<end_working_out>')==1 and s.text.count('<SOLUTION>')==1
                        and s.text.count('</SOLUTION>')==1 and s.text.rstrip().endswith('</SOLUTION>')
                        and s.text.index('<end_working_out>')<s.text.index('<SOLUTION>') and '<start_working_out>' not in s.text)
                    rec=dict(row_id=row['row_id'],completion=s.text,gold=row['answer'],level=row['level'],type=row['type'],
                             tokens=len(s.token_ids),correct=bool(verdict['correct']),math_verification=verdict,
                             eos=eos,format_eos=format_eos,truncated=s.finish_reason=='length',repetitive=repetition>.05)
                    f.write(json.dumps(rec,ensure_ascii=False)+'\n');records.append(rec)
                f.flush()
            (out/'progress.json').write_text(json.dumps(dict(arm=arm,completed=len(records),total=200)),encoding='utf-8')
            print('DEV_EVAL_PROGRESS',arm,len(records),'/200',flush=True)
        assert len(records)==200
        summary=dict(questions=200,**{k:sum(r[k] for r in records) for k in ('correct','eos','format_eos','truncated','repetitive')},
                     accuracy=sum(r['correct'] for r in records)/200,mean_tokens=sum(r['tokens'] for r in records)/200,
                     runtime_this_invocation_seconds=time.time()-start)
        if arm!='base':
            base=all_records['base']
            summary.update(base_right_sft_wrong=sum(b['correct'] and not s['correct'] for b,s in zip(base,records)),
                           base_wrong_sft_right=sum(not b['correct'] and s['correct'] for b,s in zip(base,records)))
        (dest/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
        all_records[arm]=records;summaries[arm]=summary
    eligible={k:v for k,v in summaries.items() if k!='base' and v['accuracy']>=summaries['base']['accuracy']-.01 and v['format_eos']/200>=.9}
    choice=max(eligible,key=lambda k:(eligible[k]['correct'],eligible[k]['format_eos'],-eligible[k]['mean_tokens'])) if eligible else None
    final=dict(status='complete',models=summaries,recommended_grpo_cold_start=choice,
               gate='Exploratory development gate: accuracy loss <=1 percentage point and format+EOS >=90%; not proof of noninferiority.',
               boundary='Development set with possible historical exposure; no new blind-test claim.',grpo_started=False)
    (out/'comparison.json').write_text(json.dumps(final,indent=2),encoding='utf-8')
    (out/'progress.json').write_text(json.dumps(dict(status='complete')),encoding='utf-8')
    engine.llm_engine.engine_core.shutdown()
    print('DEV_EVAL_COMPLETE',json.dumps(final),flush=True)


if __name__=='__main__':main()
