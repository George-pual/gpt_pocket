import argparse,json,random,re,unicodedata
from pathlib import Path
from common import read_jsonl,save_jsonl,save_json,file_hash,tokenizer
from math_verification import verify_final_answer


def canonical(s):
    s=unicodedata.normalize('NFKC',s).casefold()
    s=re.sub(r'\\(?:left|right|big|Big|,|;|!|\(|\)|\[|\])','',s)
    s=s.replace('\\ldots','...').replace('\\cdots','...')
    return re.sub(r'[^\w\\+*/=<>.^-]','',s)


def question(row):
    return row.get('problem') or next(m['content'] for m in row['prompt'] if m['role']=='user')


def main():
    p=argparse.ArgumentParser();p.add_argument('--storage',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();root,out=Path(a.storage),Path(a.output)
    if out.exists() and any(out.iterdir()):raise ValueError('Use fresh data directory')
    bundle=Path(__file__).parent/'inputs'
    excluded=set()
    for f in [bundle/'train.jsonl',bundle/'validation.jsonl',root/'prepared/sft.jsonl',bundle/'prior_eval_questions.jsonl']:
        for row in read_jsonl(f):excluded.add(canonical(question(row)))
    old_ids=set()
    for f in (root/'runs').rglob('*rollouts.jsonl'):
        for row in read_jsonl(f):
            if row.get('row_id') is not None:old_ids.add(row['row_id'])
    pool=root/'prepared/grpo_article_pilot10.jsonl'
    rows=read_jsonl(pool);random.Random(34117).shuffle(rows)
    chosen=[];seen=set(excluded)
    for row in rows:
        key=canonical(question(row))
        if key in seen or row['row_id'] in old_ids:continue
        if not verify_final_answer('<SOLUTION>'+row['answer']+'</SOLUTION>',row['answer'])['correct']:continue
        seen.add(key);chosen.append(row)
        if len(chosen)==72:break
    assert len(chosen)==72
    evaluation,training=chosen[:8],chosen[8:]
    save_jsonl(out/'train.jsonl',training);save_jsonl(out/'eval.jsonl',evaluation)
    meta=json.loads(pool.with_suffix('.metadata.json').read_text())
    meta.update(max_completion_length=3894,max_sequence_length=4096,retained_rows=64,
                prepared_sha256=file_hash(out/'train.jsonl'))
    save_json(out/'train.metadata.json',meta)
    old=root/'runs/E1_length_probe_1846_3894_b12/paired_rollouts.jsonl'
    audits=sorted([r for r in read_jsonl(old) if r['limit']==3894],key=lambda r:r['tokens'],reverse=True)
    save_jsonl(out/'preflight_rollouts.jsonl',audits)
    t=tokenizer(root/'model')
    for row in chosen:
        ids=t.apply_chat_template(row['prompt'],tokenize=True,add_generation_prompt=True)
        assert len(ids)<=meta['max_prompt_length']
    save_json(out/'split_manifest.json',{'seed':34117,'train_questions':64,'eval_questions':8,
        'train_ids':[r['row_id'] for r in training],'eval_ids':[r['row_id'] for r in evaluation],
        'train_sha256':file_hash(out/'train.jsonl'),'eval_sha256':file_hash(out/'eval.jsonl'),
        'excluded_sft_train_validation_original_and_prior_diagnostic':True,
        'prior_rollout_ids_excluded':len(old_ids),
        'scope':'Eight fresh held-out within-source diagnostic questions; not an external benchmark; no semantic dedup guarantee'})
    print('DATA_READY 64 train, 8 held-out diagnostic',flush=True)


if __name__=='__main__':main()
