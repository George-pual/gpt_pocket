"""Keep eligible screened MATH L2/L3; fill half of original mixed pool with GSM8K."""
import hashlib,json,tarfile
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'datasets/grpo_candidates_gsm8k_math_l2_l3'
OUT=DATA/'half_math_v3'
OUT.mkdir(exist_ok=True)
read=lambda b:[json.loads(x) for x in b.splitlines() if x.strip()]
pool=read((DATA/'grpo_candidates.jsonl').read_bytes())
rows=read((DATA/'E1_sft_v3_grpo_screen_shard0_n2500/scored_questions.jsonl').read_bytes())
for filename in ('first_snapshot.tar.gz','three_snapshot.tar.gz'):
    with tarfile.open(DATA/'combined_analysis'/filename) as archive:
        for member in archive.getmembers():
            if member.name.endswith('scored_questions.jsonl'):
                rows.extend(read(archive.extractfile(member).read()))
scores={r['row_id']:r for r in rows}
assert len(scores)==len(rows)==len(pool)
validation_path=DATA/'reward_scale128_v3/validation.jsonl'
validation=read(validation_path.read_bytes())
vids={r['row_id'] for r in validation}
mixed=[q for q in pool if scores[q['row_id']]['group']=='mixed']
assert len(mixed)==7631
target=(len(mixed)+1)//2
eligible=[];excluded=[]
for q in mixed:
    reasons=[]
    if q['row_id'] in vids: reasons.append('validation_overlap')
    if q['overlaps_sft505']: reasons.append('sft505_overlap')
    if q['overlaps_eval100']: reasons.append('previous_eval_overlap')
    if any(s['truncated'] for s in scores[q['row_id']]['samples']): reasons.append('screening_truncation')
    if reasons: excluded.append(dict(row_id=q['row_id'],source=q['source'],level=q.get('level'),reasons=reasons))
    else: eligible.append(q)
order=lambda q:hashlib.sha256(('halfmath3407:'+q['row_id']).encode()).hexdigest()
math=sorted([q for q in eligible if q['source']=='math'],key=order)
gsm=sorted([q for q in eligible if q['source']=='gsm8k'],key=order)
selected=sorted(math+gsm[:target-len(math)],key=order)
assert len(selected)==target and len({q['row_id'] for q in selected})==target
assert not {q['row_id'] for q in selected}&vids
assert {q['row_id'] for q in selected if q['source']=='math'}=={q['row_id'] for q in math}
write=lambda p,rs:p.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rs),encoding='utf-8')
write(OUT/'train_candidates.jsonl',selected)
write(OUT/'train_candidate_scores.jsonl',[scores[q['row_id']] for q in selected])
write(OUT/'excluded_questions.jsonl',excluded)
(OUT/'validation.jsonl').write_bytes(validation_path.read_bytes())
counts=dict(Counter('gsm8k' if q['source']=='gsm8k' else q['level'] for q in selected))
manifest=dict(status='data_prepared_training_not_started',original_mixed=7631,target_questions=target,selected_counts=counts,all_eligible_math_retained=True,selection_seed=3407,selection_hash_prefix='halfmath3407:',eligibility='Original screening mixed1-3/4, excluding validation/SFT/eval overlap and screening truncation; no extra format-success filter',grading='Original archived screening grades; regrade against training reward version before launch and audit any changed eligibility',validation_questions=100,validation_sha256=hashlib.sha256(validation_path.read_bytes()).hexdigest(),train_sha256=hashlib.sha256((OUT/'train_candidates.jsonl').read_bytes()).hexdigest(),excluded_questions=len(excluded),estimated_steps_one_epoch=(target+1)//2,estimated_hours_previous_rate=((target+1)//2)*2786.5746/128/3600,reward_selection='Wait for current format-components experiment; no full training launched')
(OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
(OUT/'README.md').write_text('# SFT v3 半量GRPO训练数据\n\n'+json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(manifest,ensure_ascii=False,indent=2))
