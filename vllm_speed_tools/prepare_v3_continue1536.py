"""Select 1024 unseen question groups for 512 additional optimizer updates."""
import hashlib,json
from collections import Counter
from pathlib import Path

DATA=Path(__file__).resolve().parents[1]/'datasets/grpo_candidates_gsm8k_math_l2_l3'
POOL=DATA/'expanded_math_v3'
OUT=POOL/'continue1536';OUT.mkdir(exist_ok=True)
read=lambda p:[json.loads(x) for x in p.read_bytes().splitlines() if x.strip()]
seen=set(json.loads((POOL/'half1024_consumed_ids.json').read_text()))
assert len(seen)==2048
rows=read(POOL/'train_candidates.jsonl')
scores={q['row_id']:q for q in read(POOL/'train_candidate_scores.jsonl')}
added={q['row_id'] for q in read(POOL/'added_candidates.jsonl')}
bucket=lambda q:q['source'] if q['source']=='gsm8k' else q['level']
targets={'gsm8k':524,'Level 2':250,'Level 3':250}
selected=[]
for name,n in targets.items():
    candidates=[q for q in rows if q['row_id'] not in seen and bucket(q)==name]
    # For GSM8K, use newly added pool entries before other unused questions.
    candidates.sort(key=lambda q:(q['row_id'] not in added,hashlib.sha256(('continue1536:'+q['row_id']).encode()).hexdigest()))
    assert len(candidates)>=n,(name,len(candidates),n)
    selected.extend(candidates[:n])
selected.sort(key=lambda q:hashlib.sha256(('continue1536-order:'+q['row_id']).encode()).hexdigest())
assert len(selected)==1024 and not seen&{q['row_id'] for q in selected}
write=lambda p,rs:p.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rs),encoding='utf-8')
write(OUT/'train_candidates.jsonl',selected)
write(OUT/'train_candidate_scores.jsonl',[scores[q['row_id']] for q in selected])
(OUT/'validation.jsonl').write_bytes((POOL/'validation.jsonl').read_bytes())
manifest=dict(status='prepared',base_step=1024,additional_updates=512,cumulative_step_target=1536,questions=1024,
              counts=dict(Counter(bucket(q) for q in selected)),previous_run_overlap=0,
              new_pool_questions=sum(q['row_id'] in added for q in selected),
              train_sha256=hashlib.sha256((OUT/'train_candidates.jsonl').read_bytes()).hexdigest(),
              note='Selected from expanded pool; excludes 2048 actual prior-run rollout IDs; all 1024 selected groups will be used once with 2 groups/update.')
(OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(manifest,ensure_ascii=False,indent=2))
