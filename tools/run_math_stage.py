"""Resumable vLLM evaluation/calibration for the frozen MATH project."""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import time


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(p, value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8');tmp.replace(p)


def read(p):
    return [json.loads(s) for s in Path(p).read_text(encoding='utf-8').splitlines() if s.strip()]


def format_ok(text, eos):
    return bool(eos and text.count('<end_working_out>')==1 and text.count('<SOLUTION>')==1
        and text.count('</SOLUTION>')==1 and text.rstrip().endswith('</SOLUTION>')
        and text.index('<end_working_out>')<text.index('<SOLUTION>') and '<start_working_out>' not in text)


def stats(rows):
    n=len(rows)
    return dict(questions=n,correct=sum(r['correct'] for r in rows),
        accuracy=sum(r['correct'] for r in rows)/n if n else None,
        format_eos=sum(r['format_eos'] for r in rows),eos=sum(r['eos'] for r in rows),
        truncated=sum(r['truncated'] for r in rows),mean_tokens=sum(r['tokens'] for r in rows)/n if n else None,
        grading_statuses=dict(collections.Counter(r['verdict']['status'] for r in rows)))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--model',required=True);ap.add_argument('--data',required=True)
    ap.add_argument('--output',required=True);ap.add_argument('--arms',required=True,help='JSON object arm name: adapter path or null')
    ap.add_argument('--screen',action='store_true');ap.add_argument('--batch',type=int)
    ap.add_argument('--expected-grader-sha256')
    a=ap.parse_args()
    if a.batch is None:a.batch=16 if a.screen else 64
    from scorer_runtime_guard import load_verified_scorer
    scorer,grader_runtime=load_verified_scorer(a.expected_grader_sha256)
    verify_final_answer,require_math_verify,SCHEMA=scorer.verify_final_answer,scorer.require_math_verify,scorer.SCHEMA
    print('GRADER_VERIFIED',json.dumps(grader_runtime),flush=True)
    from common import tokenizer,messages
    from vllm import LLM,SamplingParams
    from vllm.lora.request import LoRARequest
    rows=read(a.data);assert len(rows)==len({r['row_id'] for r in rows})
    arms=json.loads(a.arms);assert arms
    if a.screen:assert len(arms)==1
    t=tokenizer(a.model)
    prompts=[t.apply_chat_template(messages(r['problem']),tokenize=True,add_generation_prompt=True) for r in rows]
    assert max(map(len,prompts))+(3894)<=8192
    overlong={i for i,ids in enumerate(prompts) if a.screen and len(ids)>1024}
    arm_hashes={k:sha(Path(p)/'adapter_model.safetensors') if p else None for k,p in arms.items()}
    cfg=dict(model=a.model,model_weights={p.name:sha(p) for p in Path(a.model).glob('model-*.safetensors')},
        data_sha256=sha(a.data),arms=arms,adapter_sha256=arm_hashes,
        prompt_ids_sha256=hashlib.sha256(json.dumps(prompts).encode()).hexdigest(),
        sampling=dict(n=4 if a.screen else 1,temperature=1. if a.screen else 0.,top_p=1.,top_k=-1,min_p=0.,
                      max_tokens=3894,eos_token_id=t.eos_token_id,repetition_penalty=1.),
        seed_policy='20261016 + frozen question index' if a.screen else '3407',
        grader_sha256=grader_runtime['sha256'],grading_schema=SCHEMA,math_verify_version=require_math_verify(),grader_runtime=grader_runtime,
        template=t.chat_template,scope='Final-answer equivalence and format only; no reasoning certification')
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    manifest=out/'manifest.json'
    if manifest.exists():assert json.loads(manifest.read_text())==cfg,'Resume provenance mismatch'
    else:save(manifest,cfg)
    engine=LLM(model=a.model,dtype='bfloat16',max_model_len=8192,enable_lora=True,max_lora_rank=64,
        gpu_memory_utilization=.85,max_num_seqs=64,seed=3407,enable_prefix_caching=True,disable_log_stats=True)
    summaries={}
    for aid,(arm,adapter) in enumerate(arms.items(),1):
        dest=out/arm;groups=dest/'groups';groups.mkdir(parents=True,exist_ok=True)
        request=LoRARequest(arm,aid,adapter) if adapter else None
        if a.screen:
            save(dest/'prompt_budget_exclusions.json',[dict(row_id=rows[i]['row_id'],prompt_tokens=len(prompts[i]),
                budget=1024,reason='prompt_exceeds_budget_no_truncation') for i in sorted(overlong)])
            for i in sorted(overlong):
                r=rows[i]
                if not (groups/f'{i:05d}.json').exists():
                    save(groups/f'{i:05d}.json',dict(row_id=r['row_id'],question_index=i,level=r['level'],type=r['type'],
                        gold=r['answer'],samples=[],excluded_reason='prompt_exceeds_budget_no_truncation',prompt_tokens=len(prompts[i])))
        gold_checks={}
        if a.screen:
            gold_file=dest/'gold_checks.json'
            if gold_file.exists():gold_checks=json.loads(gold_file.read_text())
            else:
                for i,r in enumerate(rows):
                    gold_checks[r['row_id']]=verify_final_answer('<SOLUTION>'+r['answer']+'</SOLUTION>',r['answer'],question=r['problem'])
                    if (i+1)%250==0:print('GOLD_PARSE_CHECK',i+1,len(rows),flush=True)
                save(gold_file,gold_checks)
        missing=[i for i in range(len(rows)) if not (groups/f'{i:05d}.json').exists()]
        for off in range(0,len(missing),a.batch):
            indices=missing[off:off+a.batch]
            params=[SamplingParams(n=4 if a.screen else 1,temperature=1. if a.screen else 0.,
                    top_p=1.,top_k=-1,min_p=0.,max_tokens=3894,stop_token_ids=[t.eos_token_id],
                    seed=20261016+i if a.screen else 3407) for i in indices]
            generated=engine.generate([{'prompt_token_ids':prompts[i]} for i in indices],params,
                lora_request=request,use_tqdm=False)
            for i,result in zip(indices,generated):
                r=rows[i];samples=[]
                assert len(result.outputs)==(4 if a.screen else 1)
                for s in result.outputs:
                    eos=s.finish_reason=='stop' and s.stop_reason in (None,t.eos_token_id)
                    v=verify_final_answer(s.text,r['answer'],question=r['problem'])
                    samples.append(dict(sample_index=s.index,completion=s.text,token_ids=list(s.token_ids),
                        tokens=len(s.token_ids),finish_reason=s.finish_reason,stop_reason=s.stop_reason,
                        correct=bool(v['correct']),verdict=v,eos=eos,truncated=s.finish_reason=='length',
                        format_eos=format_ok(s.text,eos)))
                save(groups/f'{i:05d}.json',dict(row_id=r['row_id'],question_index=i,level=r['level'],
                     type=r['type'],gold=r['answer'],samples=samples))
            complete=len(list(groups.glob('*.json')))
            save(out/'progress.json',dict(arm=arm,completed=complete,total=len(rows),screen=a.screen))
            print('STAGE_PROGRESS',arm,complete,'/',len(rows),flush=True)
        all_groups=[json.loads((groups/f'{i:05d}.json').read_text(encoding='utf-8')) for i in range(len(rows))]
        assert [r['row_id'] for r in all_groups]==[r['row_id'] for r in rows]
        if a.screen:
            scores=[]
            for g in all_groups:
                samples=g['samples'];k=None if g.get('excluded_reason') else sum(s['correct'] for s in samples)
                unknown=sum(s['verdict']['status'] not in ('correct','incorrect') for s in samples)
                bad_end=sum(not s['eos'] or s['truncated'] for s in samples)
                reliable_gold=gold_checks[g['row_id']]['correct']
                eligible=bool(not g.get('excluded_reason') and reliable_gold and unknown==0 and bad_end==0 and 0<k<4)
                scores.append(dict(row_id=g['row_id'],k=k,u=unknown,bad_end=bad_end,
                    eligible=eligible,gold_parser_supported=reliable_gold,
                    difficulty={3:'easy',2:'medium',1:'hard'}.get(k),level=g['level'],type=g['type'],
                    excluded_reason=g.get('excluded_reason')))
            with (dest/'scores.jsonl').open('w',encoding='utf-8') as f:
                for r in scores:f.write(json.dumps(r,ensure_ascii=False)+'\n')
            summaries[arm]=dict(questions=len(rows),eligible=sum(s['eligible'] for s in scores),
                k_counts=dict(collections.Counter(s['k'] for s in scores if s['k'] is not None)),
                eligible_difficulty=dict(collections.Counter(s['difficulty'] for s in scores if s['eligible'])),
                prompt_budget_exclusions=len(overlong),
                unknown_questions=sum(s['u']>0 for s in scores),abnormal_end_questions=sum(s['bad_end']>0 for s in scores))
        else:
            answers=[dict(row_id=g['row_id'],level=g['level'],type=g['type'],gold=g['gold'],**g['samples'][0]) for g in all_groups]
            with (dest/'answers.jsonl').open('w',encoding='utf-8') as f:
                for r in answers:f.write(json.dumps(r,ensure_ascii=False)+'\n')
            summaries[arm]=dict(**stats(answers),by_level={lv:stats([r for r in answers if r['level']==lv]) for lv in ['Level 2','Level 3']})
        save(dest/'summary.json',summaries[arm]);save(out/'comparison.json',summaries)
        if adapter:assert sha(Path(adapter)/'adapter_model.safetensors')==arm_hashes[arm]
    save(out/'progress.json',dict(status='complete',questions=len(rows)))
    engine.llm_engine.engine_core.shutdown()
    print('STAGE_COMPLETE',json.dumps(summaries),flush=True)


if __name__=='__main__':main()
