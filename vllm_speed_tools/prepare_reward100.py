"""Select 100 fresh unique questions from the existing article P90 pool."""
import json
import random
from pathlib import Path
from common import read_jsonl, save_json, save_jsonl, file_hash, tokenizer
from prepare_bc import canonical, question
from math_verification import verify_final_answer

root = Path('/root/autodl-tmp/qwen3_hf')
bundle = Path('/root/autodl-tmp/qwen3_hf_autodl/experiments/closure_grpo_experiment/inputs')
out = root / 'runs/C_reward_v2_100q_v1/data'
if out.exists():
    raise RuntimeError('Data directory exists; do not overwrite')
excluded = set()
for path in [bundle/'train.jsonl', bundle/'validation.jsonl', root/'prepared/sft.jsonl',
             bundle/'prior_eval_questions.jsonl', root/'runs/E2_closure_grpo_BC_v1/data/eval.jsonl']:
    for row in read_jsonl(path):
        excluded.add(canonical(question(row)))
old_ids = set()
for path in (root/'runs').rglob('*rollouts.jsonl'):
    for row in read_jsonl(path):
        if row.get('row_id') is not None:
            old_ids.add(row['row_id'])
pool = root/'prepared/grpo_article_pilot10.jsonl'
rows = read_jsonl(pool)
random.Random(3407).shuffle(rows)
chosen = []
for row in rows:
    key = canonical(question(row))
    if key in excluded or row['row_id'] in old_ids:
        continue
    if not verify_final_answer('<SOLUTION>'+row['answer']+'</SOLUTION>', row['answer'])['correct']:
        continue
    excluded.add(key)
    chosen.append(row)
    if len(chosen) == 100:
        break
assert len(chosen) == 100
meta = json.loads(pool.with_suffix('.metadata.json').read_text())
t = tokenizer(root/'model')
assert all(len(t.apply_chat_template(r['prompt'], tokenize=True, add_generation_prompt=True))
           <= meta['max_prompt_length'] for r in chosen)
save_jsonl(out/'train.jsonl', chosen)
meta.update(retained_rows=100, max_completion_length=3894, max_sequence_length=4096,
            prepared_sha256=file_hash(out/'train.jsonl'))
save_json(out/'train.metadata.json', meta)
save_json(out/'manifest.json', {'seed':3407, 'train_questions':100,
    'train_ids':[r['row_id'] for r in chosen], 'pool_rows':len(rows),
    'pool_sha256':file_hash(pool), 'prior_rollout_ids_excluded':len(old_ids),
    'canonical_dedup':True, 'semantic_dedup_guaranteed':False,
    'train_sha256':file_hash(out/'train.jsonl'),
    'eval_sha256':file_hash(root/'runs/E2_closure_grpo_BC_v1/data/eval.jsonl')})
print('DATA_READY 100 unique fresh train questions', flush=True)
