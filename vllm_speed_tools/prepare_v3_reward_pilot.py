"""Deterministic pilot split; validation is internal, not benchmark test data."""
import json,tarfile,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'datasets/grpo_candidates_gsm8k_math_l2_l3'
OUT=DATA/'reward_pilot_v3'
OUT.mkdir(exist_ok=True)
read=lambda b:[json.loads(x) for x in b.splitlines() if x.strip()]
pool=read((DATA/'grpo_candidates.jsonl').read_bytes())
rows=read((DATA/'E1_sft_v3_grpo_screen_shard0_n2500/scored_questions.jsonl').read_bytes())
for name in ['first_snapshot.tar.gz','three_snapshot.tar.gz']:
 with tarfile.open(DATA/'combined_analysis'/name) as t:
  for m in t.getmembers():
   if m.name.endswith('scored_questions.jsonl'):rows.extend(read(t.extractfile(m).read()))
scored={r['row_id']:r for r in rows}
sort=lambda rs:sorted(rs,key=lambda r:hashlib.sha256(('pilot3407:'+r['row_id']).encode()).hexdigest())
available=sort([q for q in pool if not q['overlaps_sft505'] and not q['overlaps_eval100'] and scored[q['row_id']]['group']!='ungradable_gold'])
bucket=lambda q:'gsm8k' if q['source']=='gsm8k' else q['level']
val=[]
for source,n in [('gsm8k',60),('Level 2',20),('Level 3',20)]:val.extend([q for q in available if bucket(q)==source][:n])
vids={q['row_id'] for q in val}
train_pool=[q for q in available if q['row_id'] not in vids and scored[q['row_id']]['group']=='mixed' and scored[q['row_id']]['normal_correct_count']>0 and not any(s['truncated'] for s in scored[q['row_id']]['samples'])]
# Extra candidates allow reward-version regrading to reject newly all-right groups.
selected=[]
for source,n in [('gsm8k',120),('Level 2',40),('Level 3',24)]:selected.extend([q for q in train_pool if bucket(q)==source][:n])
write=lambda path,rs:path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rs),encoding='utf-8')
write(OUT/'train_candidates.jsonl',selected);write(OUT/'validation.jsonl',val)
write(OUT/'train_candidate_scores.jsonl',[scored[q['row_id']] for q in selected])
(OUT/'split_manifest.json').write_text(json.dumps(dict(seed=3407,train_target={'gsm8k':48,'Level 2':12,'Level 3':4},validation_questions=100,validation_scope='Internal holdout from screened training candidates; not a public test benchmark',no_train_validation_id_overlap=True),ensure_ascii=False,indent=2),encoding='utf-8')
print('PILOT_DATA',len(selected),len(val))
