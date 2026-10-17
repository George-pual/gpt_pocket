"""Freeze disjoint, stratified MATH partitions for fresh-Base self trajectories."""
import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'datasets/math/base_self600_20261015'
SEED = 20261015


def read(p):
    with p.open(encoding='utf-8-sig') as f:
        return [json.loads(s) for s in f if s.strip()]


def canonical(s):
    return re.sub(r'\s+', '', unicodedata.normalize('NFKC', s)).replace('$', '').casefold()


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def rank(r, stage):
    return hashlib.sha256(f'{SEED}:{stage}:{r["row_id"]}'.encode()).hexdigest()


def partition(rows, targets, stage):
    chosen = []
    for level, n in targets.items():
        groups = defaultdict(list)
        for r in rows:
            if r['level'] == level:
                groups[r['type']].append(r)
        total = sum(map(len, groups.values()))
        quotas = {s: n * len(g) // total for s, g in groups.items()}
        remainder = sorted(groups, key=lambda s: (-(n * len(groups[s]) % total), s))
        for s in remainder[:n - sum(quotas.values())]:
            quotas[s] += 1
        for s in sorted(groups):
            chosen.extend(sorted(groups[s], key=lambda r: rank(r, stage))[:quotas[s]])
    ids = {r['row_id'] for r in chosen}
    return sorted(chosen, key=lambda r: rank(r, stage + ':order')), [r for r in rows if r['row_id'] not in ids]


def main():
    inputs = {
        'source': ROOT / 'datasets/grpo_candidates_gsm8k_math_l2_l3/candidates_all.jsonl',
        'original_train': ROOT / 'datasets/math/train.jsonl',
        'sft505': ROOT / 'sft_quality_audit_v2/merged_supplied_answers/train_merged_505.jsonl',
        'test': ROOT / 'datasets/math/test_l2_l3_full/questions_all_official.jsonl',
    }
    if OUT.exists():
        raise RuntimeError('Refuse to overwrite frozen partition')
    train = {r['id']: r for r in read(inputs['original_train'])}
    excluded_keys = {canonical(r['problem']) for k in ('sft505', 'test') for r in read(inputs[k])}
    pool, excluded, seen = [], [], set()
    for r in read(inputs['source']):
        if r['source'] != 'math' or r['level'] not in ('Level 2', 'Level 3'):
            continue
        original = train[r['row_id']]
        assert original['split'] == r['split'] == 'train'
        assert original['problem'].strip() == r['problem'] and original['solution'] == r['reference_solution']
        key = canonical(r['problem'])
        reason = 'sft505_or_test_exact_overlap' if key in excluded_keys else 'canonical_duplicate' if key in seen else None
        if reason:
            excluded.append({'row_id': r['row_id'], 'reason': reason})
            continue
        seen.add(key)
        pool.append({k: r[k] for k in ('row_id', 'dataset', 'split', 'level', 'type', 'problem', 'answer', 'reference_solution', 'prompt')})
    dev, rest = partition(pool, {'Level 2': 90, 'Level 3': 110}, 'dev')
    sft, grpo = partition(rest, {'Level 2': 270, 'Level 3': 330}, 'sft')
    assert len(dev) == 200 and len(sft) == 600
    assert len({canonical(r['problem']) for r in dev + sft + grpo}) == len(pool)
    assert all(canonical(r['problem']) not in excluded_keys for r in pool)
    OUT.mkdir(parents=True)
    files = {'questions.jsonl': sft, 'development.jsonl': dev, 'grpo_reserved.jsonl': grpo, 'exclusions.jsonl': excluded}
    for name, rows in files.items():
        (OUT / name).write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8', newline='\n')
    meta = {
        'status': 'frozen_generation_not_started', 'seed': SEED,
        'source': 'MATH official train L2-L3 only',
        'counts': {k: len(v) for k, v in files.items()},
        'sft_strata': dict(Counter(r['level'] + '/' + r['type'] for r in sft)),
        'source_sha256': {k: sha(p) for k, p in inputs.items()},
        'output_sha256': {k: sha(OUT / k) for k in files},
        'selection': 'Deterministic SHA256 ranking and proportional largest-remainder subject allocation within each Level; development first, SFT next, GRPO remainder.',
        'overlap_scope': 'NFKC/case/whitespace/dollar normalized exact matching; semantic duplicates and pretraining contamination not excluded.',
        'development_boundary': 'New partition for a fresh Base run. Historical project exposure is not excluded; this is not a new blind confirmation set.',
        'generation': {'n': 4, 'temperature': 1.0, 'top_p': 1.0, 'top_k': -1, 'max_tokens': 3894, 'engine': 'vLLM', 'base_only': True},
    }
    (OUT / 'selection_manifest.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
