"""Expand the half-size pool to all eligible previously screened mixed questions."""
import hashlib
import json
import tarfile
from collections import Counter
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / 'datasets/grpo_candidates_gsm8k_math_l2_l3'
OUT = DATA / 'expanded_math_v3'
OUT.mkdir(exist_ok=True)
read = lambda b: [json.loads(x) for x in b.splitlines() if x.strip()]
pool = read((DATA / 'grpo_candidates.jsonl').read_bytes())
scores = read((DATA / 'E1_sft_v3_grpo_screen_shard0_n2500/scored_questions.jsonl').read_bytes())
for filename in ('first_snapshot.tar.gz', 'three_snapshot.tar.gz'):
    with tarfile.open(DATA / 'combined_analysis' / filename) as archive:
        for member in archive.getmembers():
            if member.name.endswith('scored_questions.jsonl'):
                scores.extend(read(archive.extractfile(member).read()))
by_id = {q['row_id']: q for q in scores}
assert len(by_id) == len(scores) == len(pool)
validation = DATA / 'half_math_v3/validation.jsonl'
vids = {q['row_id'] for q in read(validation.read_bytes())}
old = {q['row_id'] for q in read((DATA / 'half_math_v3/train_candidates.jsonl').read_bytes())}
eligible, excluded = [], []
for q in pool:
    if by_id[q['row_id']]['group'] != 'mixed':
        continue
    reasons = []
    if q['row_id'] in vids:
        reasons.append('validation_overlap')
    if q['overlaps_sft505']:
        reasons.append('sft505_overlap')
    if q['overlaps_eval100']:
        reasons.append('previous_eval_overlap')
    if any(s['truncated'] for s in by_id[q['row_id']]['samples']):
        reasons.append('screening_truncation')
    if reasons:
        excluded.append({'row_id': q['row_id'], 'reasons': reasons})
    else:
        eligible.append(q)
eligible.sort(key=lambda q: hashlib.sha256(('halfmath3407:' + q['row_id']).encode()).hexdigest())
assert old <= {q['row_id'] for q in eligible}
assert len({q['row_id'] for q in eligible}) == len(eligible)
added = [q for q in eligible if q['row_id'] not in old]
write = lambda p, rows: p.write_text(''.join(json.dumps(q, ensure_ascii=False) + '\n' for q in rows), encoding='utf-8')
write(OUT / 'train_candidates.jsonl', eligible)
write(OUT / 'train_candidate_scores.jsonl', [by_id[q['row_id']] for q in eligible])
write(OUT / 'added_candidates.jsonl', added)
write(OUT / 'excluded_questions.jsonl', excluded)
(OUT / 'validation.jsonl').write_bytes(validation.read_bytes())
counts = lambda rows: dict(Counter(q['source'] if q['source'] == 'gsm8k' else q['level'] for q in rows))
manifest = {'status': 'prepared_not_launched', 'questions': len(eligible), 'counts': counts(eligible),
            'previous_pool': len(old), 'added_questions': len(added), 'added_counts': counts(added),
            'excluded_questions': len(excluded), 'train_sha256': hashlib.sha256((OUT / 'train_candidates.jsonl').read_bytes()).hexdigest(),
            'validation_sha256': hashlib.sha256(validation.read_bytes()).hexdigest(),
            'scope': 'All eligible archived mixed 1-3/4 screening groups; same exclusions as half pool. Regrade before training. Added means absent from previous candidate pool, not an audited training exposure ledger.',
            'resume_policy': 'Audit checkpoint-1024 optimizer state and actual consumed row IDs before launching; prioritize unseen questions.'}
(OUT / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(manifest, ensure_ascii=False, indent=2))
